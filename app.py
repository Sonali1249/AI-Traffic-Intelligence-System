from flask import Flask, render_template, Response, jsonify, request
from traffic_core import generate_frames, latest_stats, video_state, DB_PATH
from traffic_agent import TrafficAgent
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

agent = TrafficAgent(DB_PATH)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/stats")
def stats():
    return jsonify(latest_stats)

@app.route("/api/history")
def history():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Get last 20 records
        c.execute("SELECT vehicle_id, type, speed, lane, plate_text, is_speeding, timestamp FROM vehicle_logs ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        
        data = []
        for r in rows:
            data.append({
                "id": r[0],
                "type": r[1],
                "speed": round(r[2], 1),
                "lane": r[3],
                "plate": r[4],
                "is_speeding": bool(r[5]),
                "time": r[6]
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/video_control", methods=["POST"])
def video_control():
    action = request.json.get("action")
    
    if action == "pause":
        video_state.paused = True
    elif action == "play":
        video_state.paused = False
    elif action == "seek_forward":
        video_state.seek_frame = min(video_state.total_frames, video_state.frame_position + 300) # +10s (approx 30fps)
        video_state.seek_requested = True
    elif action == "seek_backward":
        video_state.seek_frame = max(0, video_state.frame_position - 300) # -10s
        video_state.seek_requested = True
        
    return jsonify({"success": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"response": "Please ask a question."})
    
    result = agent.analyze(user_query)
    return jsonify(result)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AI Traffic Intelligence System v3.0")
    print("="*50)
    print("📊 Dashboard: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
