import cv2
import math
import time
import os
import numpy as np
import sqlite3
import easyocr
from collections import deque
from ultralytics import YOLO
from datetime import datetime
import threading

# Get directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "train", "images")
DB_PATH = os.path.join(BASE_DIR, "traffic_data.db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize EasyOCR (English) - auto-detect GPU
reader = easyocr.Reader(['en'], gpu=True) # gpu=True will fallback to CPU if no GPU/CUDA found

# Load YOLOv8 model
model_path = os.path.join(BASE_DIR, "yolov8n.pt")
model = YOLO(model_path)

# Video State Control
class VideoState:
    def __init__(self):
        self.paused = False
        self.frame_position = 0
        self.total_frames = 0
        self.seek_requested = False
        self.seek_frame = 0

video_state = VideoState()

# Database Setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  vehicle_id INTEGER,
                  type TEXT,
                  confidence REAL,
                  speed REAL,
                  lane INTEGER,
                  plate_text TEXT,
                  timestamp DATETIME,
                  is_speeding BOOLEAN)''')
    conn.commit()
    conn.close()

init_db()

def migrate_db():
    """Add new columns for Indian context if they don't exist"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("ALTER TABLE vehicle_logs ADD COLUMN is_helmet_missing BOOLEAN")
            c.execute("ALTER TABLE vehicle_logs ADD COLUMN rider_count INTEGER")
        except sqlite3.OperationalError:
            pass # Columns likely exist
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Migration Error: {e}")

migrate_db()

def log_vehicle_to_db(data):
    """Log vehicle data to SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO vehicle_logs (vehicle_id, type, confidence, speed, lane, plate_text, timestamp, is_speeding, is_helmet_missing, rider_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (data['id'], data['type'], data['confidence'], data['speed'], data['lane'], data['plate_text'], datetime.now(), data['is_speeding'], data.get('is_helmet_missing', False), data.get('rider_count', 1)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# Global Stats
latest_stats = {
    "vehicles": 0,
    "traffic": "LOW",
    "avg_speed": 0,
    "max_speed": 0,
    "min_speed": 0,
    "lane_distribution": {1: 0, 2: 0},
    "speeding_count": 0,
    "vehicle_types": {},
    "total_detected": 0,
    "fps": 0,
    "helmet_violations": 0,
    "triple_riding_violations": 0
}

# Tracking Data
vehicles = {}
vehicle_id_counter = 0
trajectories = {}

# Constants
# Constants
DIST_THRESHOLD = 70
PIXELS_PER_METER = 8
SPEED_LIMIT = 60  # km/h

def detect_visual_violations(crop, cls_name):
    """Heuristic check for helmets and triple riding using color analysis."""
    is_helmet_missing = False
    rider_count = 1

    if cls_name == "motorcycle":
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return is_helmet_missing, rider_count

        # --- Rider Count via aspect ratio ---
        # Taller crops (relative to width) suggest a pillion rider
        if h > w * 1.6:
            rider_count = 2
        if h > w * 2.2:
            rider_count = 3  # Triple riding

        # --- Helmet Detection via HSV color heuristic ---
        # Check the top 40% of the crop for skin-tone colors.
        # Skin tones in HSV: Hue 0-25, Sat 30-170, Val 80-255
        # A helmet would NOT show significant skin tones on top.
        top_region = crop[:int(h * 0.4), :]
        hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 80], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / (top_region.shape[0] * top_region.shape[1] + 1e-5)
        if skin_ratio > 0.25:  # >25% skin visible on top of rider -> likely no helmet
            is_helmet_missing = True

    return is_helmet_missing, rider_count

def get_centroid(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def determine_lane(cx, frame_width):
    if cx < frame_width / 2:
        return 1
    return 2

def recognize_plate(frame, x1, y1, x2, y2):
    """Attempt to read license plate from vehicle crop"""
    try:
        # Crop vehicle
        vehicle_crop = frame[max(0,y1):min(y2,frame.shape[0]), max(0,x1):min(x2,frame.shape[1])]
        if vehicle_crop.size == 0: return ""
        
        # Simple processing for OCR
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        
        # Return first valid text
        for (bbox, text, prob) in results:
            if prob > 0.4 and len(text) > 3:
                return text
    except Exception as e:
        print(f"OCR Error: {e}")
    return "Unknown"

def save_training_data(frame, x1, y1, x2, y2, class_name, vid):
    """Save crop for training AI"""
    try:
        vehicle_crop = frame[max(0,y1):min(y2,frame.shape[0]), max(0,x1):min(x2,frame.shape[1])]
        if vehicle_crop.size > 0:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{class_name}_{vid}.jpg"
            cv2.imwrite(os.path.join(DATA_DIR, filename), vehicle_crop)
    except:
        pass

def generate_frames():
    global vehicles, vehicle_id_counter, trajectories, latest_stats, video_state
    
    video_path = os.path.join(BASE_DIR, "traffic_2.mp4")
    cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)
    
    if cap.isOpened():
        video_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    while True:
        # Handle Video Controls
        if video_state.seek_requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_state.seek_frame)
            video_state.seek_requested = False
            # Reset tracking on seek
            vehicles = {}
            trajectories = {}
            
        if video_state.paused:
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            vehicles = {}
            trajectories = {}
            continue
            
        # Update progress
        video_state.frame_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Resize
        frame = cv2.resize(frame, (1280, 720))
        H, W = frame.shape[:2]
        
        # Lanes Overlay
        cv2.line(frame, (W//2, 0), (W//2, H), (255, 255, 255), 2)  # Lane divider
        cv2.putText(frame, "LANE 1", (W//4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "LANE 2", (3*W//4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        current_time = time.time()
        
        # YOLO Detection
        results = model(frame, verbose=False, conf=0.35, imgsz=640)
        
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]: # car, motorcycle, bus, truck (COCO indices)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = get_centroid(x1, y1, x2, y2)
                    conf = float(box.conf[0])
                    cls_name = model.names[int(box.cls[0])]
                    detections.append((cx, cy, cls_name, x1, y1, x2, y2, conf))
        
        # Tracking Logic
        new_vehicles = {}
        lane_counts = {1: 0, 2: 0}
        vehicle_type_counts = {}
        speeding_count = 0
        helmet_violation_count = 0
        triple_riding_count = 0
        
        for cx, cy, cls_name, x1, y1, x2, y2, conf in detections:
            matched_vid = None
            min_dist = DIST_THRESHOLD
            
            for vid, vdata in vehicles.items():
                px, py = vdata['pos']
                dist = math.hypot(cx - px, cy - py)
                if dist < min_dist:
                    min_dist = dist
                    matched_vid = vid
            
            lane = determine_lane(cx, W)
            lane_counts[lane] += 1
            
            speed = 0
            is_speeding = False
            
            if matched_vid is not None:
                # Update existing
                prev = vehicles[matched_vid]
                time_diff = current_time - prev['time']
                if time_diff > 0:
                    pixel_dist = min_dist
                    meters = pixel_dist / PIXELS_PER_METER
                    curr_speed = (meters / time_diff) * 3.6
                    speed = 0.7 * curr_speed + 0.3 * prev['speed'] # Smooth speed
                
                vid = matched_vid
                
                # Check for updates (only calculate OCR/Database occasionally to save perf)
                if speed > SPEED_LIMIT: is_speeding = True
                    
                # Store Data
                new_vehicles[vid] = {
                    "pos": (cx, cy),
                    "time": current_time,
                    "speed": speed,
                    "type": cls_name,
                    "lane": lane,
                    "plate": prev.get("plate", "Scanning..."),
                    "saved": prev.get("saved", False),
                    "confidence": conf
                }
                
                # DB & OCR Trigger (Once per vehicle when stable)
                if not prev.get("saved") and 0.5 < speed < 120 and conf > 0.6: 
                    # Try OCR
                    plate = recognize_plate(frame, x1, y1, x2, y2)
                    new_vehicles[vid]["plate"] = plate if plate else "Unknown"
                    new_vehicles[vid]["saved"] = True
                    
                    # Save Training Data
                    save_training_data(frame, x1, y1, x2, y2, cls_name, vid)
                    
                    # Log to DB
                    log_data = {
                        "id": vid,
                        "type": cls_name,
                        "confidence": conf,
                        "speed": speed,
                        "lane": lane,
                        "plate_text": new_vehicles[vid]["plate"],
                        "is_speeding": is_speeding,
                        "is_helmet_missing": prev.get("helmet_violation", False),
                        "rider_count": prev.get("rider_count", 1)
                    }
                    threading.Thread(target=log_vehicle_to_db, args=(log_data,)).start()
                
            else:
                # New Vehicle
                vehicle_id_counter += 1
                vid = vehicle_id_counter
                new_vehicles[vid] = {
                    "pos": (cx, cy),
                    "time": current_time,
                    "speed": 0,
                    "type": cls_name,
                    "lane": lane,
                    "plate": "Scanning...",
                    "saved": False,
                    "confidence": conf,
                    "helmet_violation": False,
                    "rider_count": 1
                }
                
                 # Check Indian Context Violations
                if cls_name == "motorcycle":
                    # Crop logic
                    vehicle_crop = frame[max(0,y1):min(y2,H), max(0,x1):min(x2,W)]
                    if vehicle_crop.size > 0:
                        no_helmet, riders = detect_visual_violations(vehicle_crop, cls_name)
                        new_vehicles[vid]["helmet_violation"] = no_helmet
                        new_vehicles[vid]["rider_count"] = riders
            
            if speed > SPEED_LIMIT: speeding_count += 1
            
            # Drawing
            color = (0, 0, 255) if speed > SPEED_LIMIT else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{vid} | {int(speed)}km/h | {new_vehicles[vid]['plate']}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Speeding Alert
                
            # Indian Context Alerts
            if new_vehicles[vid].get("helmet_violation"):
                helmet_violation_count += 1
                cv2.putText(frame, "NO HELMET", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            if new_vehicles[vid].get("rider_count", 1) > 2:
                triple_riding_count += 1
                cv2.putText(frame, "TRIPLE RIDING", (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
                
        vehicles = new_vehicles

        # --- Aggregate vehicle type counts ---
        for v in vehicles.values():
            vt = v.get('type', 'unknown')
            vehicle_type_counts[vt] = vehicle_type_counts.get(vt, 0) + 1

        # Stats Update
        speeds = [v['speed'] for v in vehicles.values() if v['speed'] > 0]
        avg_spd = sum(speeds) / len(speeds) if speeds else 0
        max_spd = max(speeds) if speeds else 0
        min_spd = min(speeds) if speeds else 0

        # Traffic density classification
        num_vehicles = len(vehicles)
        if num_vehicles == 0:
            traffic_level = "CLEAR"
        elif num_vehicles <= 4:
            traffic_level = "LOW"
        elif num_vehicles <= 9:
            traffic_level = "MEDIUM"
        else:
            traffic_level = "HIGH"

        fps_frame_count += 1
        if fps_frame_count >= 15:
            fps = 15 / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()

        latest_stats["total_detected"] += len(detections)
        latest_stats.update({
            "vehicles": num_vehicles,
            "traffic": traffic_level,
            "avg_speed": round(avg_spd, 1),
            "max_speed": round(max_spd, 1),
            "min_speed": round(min_spd, 1),
            "lane_distribution": lane_counts,
            "speeding_count": speeding_count,
            "vehicle_types": vehicle_type_counts,
            "helmet_violations": helmet_violation_count,
            "triple_riding_violations": triple_riding_count,
            "fps": round(fps, 1),
            "total_frames": video_state.total_frames,
            "current_frame": video_state.frame_position,
            "is_paused": video_state.paused
        })
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
