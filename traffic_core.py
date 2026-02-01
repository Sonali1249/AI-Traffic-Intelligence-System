import cv2
import math
import time
import os
import numpy as np
from collections import deque
from ultralytics import YOLO

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load YOLOv8 model
model_path = os.path.join(BASE_DIR, "yolov8n.pt")
model = YOLO(model_path)

# Load video - fallback to webcam if video file doesn't exist
video_path = os.path.join(BASE_DIR, "traffic_2.mp4")
cap = None
is_video_file = False

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        is_video_file = True
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"✓ Using video file: {video_path}")
    else:
        cap = None
        print(f"✗ Could not open video file: {video_path}")

if cap is None:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Using webcam instead.")
    else:
        print("✗ Error: Could not open webcam or video file!")

# Vehicle classes we care about
Vehicle_Classes = ["car", "motorcycle", "bus", "truck"]

# Shared stats for Flask - ALL REAL-TIME DATA
latest_stats = {
    "vehicles": 0,
    "traffic": "LOW",
    "avg_speed": 0,
    "max_speed": 0,
    "min_speed": 0,
    "speed_distribution": {
        "slow": 0,
        "medium": 0,
        "fast": 0
    },
    "vehicle_types": {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    },
    "total_detected": 0,
    "fps": 0,
    "detection_confidence": 0,
    "direction_stats": {
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0
    }
}

# Tracking data with trajectory history - REAL TRACKING
vehicles = {}
vehicle_id = 0
trajectories = {}  # Store trajectory points for each vehicle
vehicle_directions = {}  # Store confirmed direction for each vehicle
MIN_MOVEMENT_THRESHOLD = 15  # Minimum pixels moved to determine direction

# Constants
DIST_THRESHOLD = 70
PIXELS_PER_METER = 8
LOW_TRAFFIC = 10
MEDIUM_TRAFFIC = 25
MAX_TRAJECTORY_LENGTH = 40


def get_centroid(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def calculate_real_direction(trajectory):
    """Calculate REAL direction based on actual trajectory movement"""
    if len(trajectory) < 3:  # Need at least 3 points for accurate direction
        return "unknown"
    
    # Get first and last points
    start_point = trajectory[0]
    end_point = trajectory[-1]
    
    # Calculate total movement
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    
    # Calculate total distance moved
    total_distance = math.hypot(dx, dy)
    
    # Only determine direction if vehicle has moved significantly
    if total_distance < MIN_MOVEMENT_THRESHOLD:
        return "unknown"
    
    # Determine primary direction based on movement
    if abs(dx) > abs(dy):
        # Horizontal movement
        if dx > 0:
            return "right"
        else:
            return "left"
    else:
        # Vertical movement
        if dy > 0:
            return "down"
        else:
            return "up"


def draw_trajectory(frame, trajectory, color):
    """Draw trajectory line for a vehicle"""
    if len(trajectory) < 2:
        return
    
    points = np.array(trajectory, dtype=np.int32)
    for i in range(1, len(points)):
        alpha = i / len(points)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)


def draw_info_panel(frame, stats, fps):
    """Draw REAL-TIME information panel on video"""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Traffic status with color
    traffic = stats.get("traffic", "LOW")
    if traffic == "LOW":
        traffic_color = (0, 255, 0)
    elif traffic == "MEDIUM":
        traffic_color = (0, 255, 255)
    else:
        traffic_color = (0, 0, 255)
    
    # Draw REAL-TIME information
    y_offset = 35
    cv2.putText(frame, f"LIVE TRAFFIC ANALYSIS - REAL-TIME", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Traffic: {traffic}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, traffic_color, 2)
    
    y_offset += 25
    cv2.putText(frame, f"Vehicles: {stats.get('vehicles', 0)}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 25
    cv2.putText(frame, f"Avg Speed: {stats.get('avg_speed', 0):.1f} km/h", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 25
    directions = stats.get("direction_stats", {})
    dir_text = f"L:{directions.get('left', 0)} R:{directions.get('right', 0)} U:{directions.get('up', 0)} D:{directions.get('down', 0)}"
    cv2.putText(frame, dir_text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {stats.get('detection_confidence', 0):.1f}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def generate_frames():
    global vehicles, vehicle_id, trajectories, vehicle_directions
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        if not cap.isOpened():
            print("Error: Video capture is not opened")
            break
            
        ret, frame = cap.read()
        if not ret:
            # If video file ended, restart it for looping
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not restart video")
                    break
                # Reset ALL tracking when video restarts
                vehicles = {}
                trajectories = {}
                vehicle_directions = {}
                vehicle_id = 0
                # Reset stats
                latest_stats["direction_stats"] = {"left": 0, "right": 0, "up": 0, "down": 0}
            else:
                print("Error: Could not read from webcam")
                break

        # Resize frame for processing
        try:
            frame = cv2.resize(frame, (1280, 720))
        except Exception as e:
            print(f"Error resizing frame: {e}")
            continue
            
        current_time = time.time()

        # Run YOLO detection - REAL DETECTION
        try:
            results = model(frame, verbose=False, conf=0.35, imgsz=640, device='cpu', half=False)
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            continue
            
        detections = []
        confidences = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name in Vehicle_Classes and confidence > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = get_centroid(x1, y1, x2, y2)
                    confidences.append(confidence)
                    detections.append((cx, cy, class_name, x1, y1, x2, y2, confidence))

        ##REAL VEHICLE TRACKING & SPEED CALCULATION
        new_vehicles = {}
        # Reset direction counts - count only ACTIVE vehicles with confirmed direction
        direction_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

        for cx, cy, class_name, x1, y1, x2, y2, conf in detections:
            matched = False
            best_match_id = None
            best_distance = DIST_THRESHOLD

            # Find best matching vehicle based on REAL position
            for vid, data in vehicles.items():
                px, py = data["pos"]
                distance = math.hypot(cx - px, cy - py)

                if distance < best_distance:
                    best_distance = distance
                    best_match_id = vid

            # Update existing vehicle or create new
            if best_match_id is not None:
                vid = best_match_id
                prev_data = vehicles[vid]
                prev_pos = prev_data["pos"]
                
                time_diff = current_time - prev_data["time"]
                
                # REAL speed calculation
                if time_diff > 0 and best_distance > 0:
                    meters = best_distance / PIXELS_PER_METER
                    speed = (meters / time_diff) * 3.6
                    # Smooth speed with exponential moving average
                    prev_speed = prev_data.get("speed", 0)
                    speed = 0.7 * speed + 0.3 * prev_speed if prev_speed > 0 else speed
                else:
                    speed = prev_data.get("speed", 0)
                
                # Update trajectory with REAL position
                if vid not in trajectories:
                    trajectories[vid] = deque(maxlen=MAX_TRAJECTORY_LENGTH)
                trajectories[vid].append((cx, cy))
                
                # Calculate REAL direction from trajectory
                if len(trajectories[vid]) >= 3:
                    real_direction = calculate_real_direction(list(trajectories[vid]))
                    if real_direction != "unknown":
                        vehicle_directions[vid] = real_direction
                
                # Get confirmed direction
                direction = vehicle_directions.get(vid, "unknown")
                
                # Count direction only if confirmed
                if direction != "unknown" and direction in direction_counts:
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                
                new_vehicles[vid] = {
                    "pos": (cx, cy),
                    "time": current_time,
                    "speed": speed,
                    "type": class_name,
                    "confidence": conf,
                    "direction": direction
                }
                
                # Draw REAL trajectory
                if len(trajectories[vid]) > 1:
                    if speed > 60:
                        color = (0, 0, 255)  # Red for fast
                    elif speed > 30:
                        color = (0, 255, 255)  # Yellow for medium
                    else:
                        color = (0, 255, 0)  # Green for slow
                    draw_trajectory(frame, list(trajectories[vid]), color)
                
                # Draw bounding box with REAL speed-based color
                if speed > 60:
                    box_color = (0, 0, 255)  # Red for fast
                elif speed > 30:
                    box_color = (0, 255, 255)  # Yellow for medium
                else:
                    box_color = (0, 255, 0)  # Green for slow
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Enhanced label with REAL data
                direction_symbol = {"left": "←", "right": "→", "up": "↑", "down": "↓"}.get(direction, "")
                label = f"ID:{vid} {class_name} {int(speed)}km/h {direction_symbol}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1 - 5, label_size[1] + 5)
                
                # Label background
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                            (x1 + label_size[0] + 5, label_y + 5), box_color, -1)
                cv2.putText(frame, label, (x1 + 2, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                matched = True
            else:
                # New vehicle - REAL new detection
                vehicle_id += 1
                vid = vehicle_id
                new_vehicles[vid] = {
                    "pos": (cx, cy),
                    "time": current_time,
                    "speed": 0,
                    "type": class_name,
                    "confidence": conf,
                    "direction": "unknown"
                }
                
                trajectories[vid] = deque(maxlen=MAX_TRAJECTORY_LENGTH)
                trajectories[vid].append((cx, cy))
                
                # Draw bounding box for new vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                label = f"ID:{vid} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Clean up old trajectories and directions for vehicles no longer detected
        active_ids = set(new_vehicles.keys())
        trajectories = {vid: traj for vid, traj in trajectories.items() if vid in active_ids}
        vehicle_directions = {vid: dir for vid, dir in vehicle_directions.items() if vid in active_ids}
        
        vehicles = new_vehicles

        ##REAL TRAFFIC ANALYSIS
        total_vehicles = len(vehicles)

        if total_vehicles <= LOW_TRAFFIC:
            traffic_status = "LOW"
            color = (0, 255, 0)
        elif total_vehicles <= MEDIUM_TRAFFIC:
            traffic_status = "MEDIUM"
            color = (0, 255, 255)
        else:
            traffic_status = "HEAVY"
            color = (0, 0, 255)

        ##REAL SPEED ANALYSIS - Only from vehicles with actual speed
        speeds = [data["speed"] for data in vehicles.values() if data["speed"] > 0]
        
        if speeds:
            latest_stats["avg_speed"] = round(sum(speeds) / len(speeds), 1)
            latest_stats["max_speed"] = round(max(speeds), 1)
            latest_stats["min_speed"] = round(min(speeds), 1)
            
            slow_count = sum(1 for s in speeds if 0 < s <= 30)
            medium_count = sum(1 for s in speeds if 30 < s <= 60)
            fast_count = sum(1 for s in speeds if s > 60)
            
            latest_stats["speed_distribution"] = {
                "slow": slow_count,
                "medium": medium_count,
                "fast": fast_count
            }
        else:
            latest_stats["avg_speed"] = 0
            latest_stats["max_speed"] = 0
            latest_stats["min_speed"] = 0
            latest_stats["speed_distribution"] = {"slow": 0, "medium": 0, "fast": 0}

        ##REAL VEHICLE TYPE ANALYSIS
        vehicle_types_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for vid, data in vehicles.items():
            vtype = data.get("type", "car")
            if vtype in vehicle_types_count:
                vehicle_types_count[vtype] += 1
        
        ##REAL FPS CALCULATION
        fps_frame_count += 1
        if fps_frame_count >= 15:
            fps_elapsed = time.time() - fps_start_time
            fps = round(fps_frame_count / fps_elapsed, 1) if fps_elapsed > 0 else 0
            fps_frame_count = 0
            fps_start_time = time.time()
        
        ##REAL AVERAGE CONFIDENCE
        avg_confidence = round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0

        # Update ALL stats with REAL data
        latest_stats["vehicles"] = total_vehicles
        latest_stats["traffic"] = traffic_status
        latest_stats["vehicle_types"] = vehicle_types_count
        latest_stats["total_detected"] = latest_stats.get("total_detected", 0) + len(detections)
        latest_stats["fps"] = fps
        latest_stats["detection_confidence"] = avg_confidence
        latest_stats["direction_stats"] = direction_counts  # REAL direction counts

        # Draw REAL-TIME info panel
        draw_info_panel(frame, latest_stats, fps)

        ##STREAM FRAME TO FLASK
        try:
            # Resize for display
            display_frame = cv2.resize(frame, (1080, 608), interpolation=cv2.INTER_LINEAR)
            
            # Encode with optimized quality
            ret, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("Error encoding frame")
                continue
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error processing frame for stream: {e}")
            continue
