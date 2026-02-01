from flask import Flask, render_template, Response, jsonify
from traffic_core import generate_frames, latest_stats

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

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AI Traffic Intelligence System")
    print("="*50)
    print("📊 Dashboard: http://localhost:5000")
    print("📹 Video Stream: http://localhost:5000/video")
    print("📈 Stats API: http://localhost:5000/stats")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
