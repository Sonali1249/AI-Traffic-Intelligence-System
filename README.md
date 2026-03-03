# 🚦 AI Traffic Intelligence System V4.0 (Indian Enterprise Edition)

**Professional Real-Time Traffic Analysis & Enforcement System**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red)
![Flask](https://img.shields.io/badge/Backend-Flask-black)
![SQLite](https://img.shields.io/badge/Data-SQLite-blue)

A complete, production-ready solution for traffic monitoring, detecting vehicle attributes, identifying speeding violations, and collecting data for AI training.

## 🚀 Key Features

### 1. 📷 Advanced Computer Vision
- **Live Video Analysis**: Supports MP4 files, Webcams, and RTSP/IP Camera streams.
- **Multi-Class Detection**: Detects **Cars, Motorcycles, Buses, and Trucks** using `YOLOv8`.
- **Indian Context Safety**:
  - **Helmet Detection**: Flags motorcyclists without safety gear.
  - **Triple Riding Detection**: Identifies overloaded two-wheelers.
- **Lane Detection**: Automatically classifies vehicles into **Lane 1** or **Lane 2**.
- **License Plate Recognition (OCR)**: Integrates **EasyOCR** to read license plates.
- **Speed Estimation**: Real-time speed calculation (km/h).

### 2. 🛡️ Enforcement & Safety
- **Speed Enforcement**:
  - Automatically flags vehicles exceeding **60 km/h**.
  - Highlights violators with a **RED** bounding box and "OVERSPEED" tag.
- **Traffic Flow Analysis**: Monitors traffic density (Low/Medium/High) and directional flow.

### 3. 📊 Professional Dashboard (V3.0)
- **Live Video Player**: Centralized feed with **Pause, Play, and Seek** controls.
- **🛡️ AI Analyst Agent**: Integrated Chatbot to query the traffic database in natural language.
- **Violation Badges**: Custom tags for Speeding, No Helmet, and Triple Riding.
- **Real-Time Data Table**: Live scrolling log with rich metadata.
- **Responsive Design**: Dark-themed, high-contrast UI optimized for control centers.

### 4. 💾 Data Engineering & AI Training
- **SQL Database**: Automatically logs every detected vehicle to `traffic_data.db` (SQLite).
- **Automated Dataset Creation**:
  - Automatically crops vehicle images.
  - Saves high-confidence samples to `data/train/images/` for future model fine-tuning.

---

## 🛠️ Technical Architecture

### Backend (`app.py` & `traffic_core.py`)
- **Framework**: Flask (Python)
- **AI Core**: Ultralytics YOLOv8n + EasyOCR
- **Database**: SQLite3 (Schema: `vehicle_logs`)
- **API Endpoints**:
  - `GET /stats`: Real-time telemetry.
  - `GET /api/history`: Last 20 vehicle logs.
  - `POST /video_control`: DVR controls (Pause/Seek).

### Frontend
- **HTML5/CSS3**: Custom "Inter" font typography, Grid Layout, Glassmorphism elements.
- **JavaScript**: Fetch API for seamless non-blocking updates.

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sonali1249/AI-Traffic-Intelligence-System.git
   cd AI-Traffic-Intelligence-System
   ```

2. **Install Dependencies**
   (Requires Python 3.8+)
   ```bash
   pip install -r requirements.txt
   pip install easyocr  # For License Plate Recognition
   ```

3. **Run the System**
   ```bash
   python app.py
   ```
   *The system will automatically download the `yolov8n.pt` model on the first run.*

4. **Access Dashboard**
   - Open browser: `http://localhost:5000`

---

## 🎥 Connecting Live Cameras (RTSP)

To use a **Real Live Traffic IP Camera** instead of the demo video:
1. Open `traffic_core.py`.
2. Find the `generate_frames()` function.
3. Replace the `video_path` with your camera's RTSP URL:
   ```python
   # Example for an IP Camera
   cap = cv2.VideoCapture("rtsp://admin:password@192.168.1.100:554/stream1")
   ```

---

## 👤 Author
**Sonali Tiwari**
*AI Engineer & Full Stack Developer*

## ☁️ Deploy to Cloud Run

You can containerize and deploy the Streamlit dashboard to Google Cloud Run.

### 1. Build the Docker image
```bash
# From the project root
docker build -t gcr.io/<YOUR_PROJECT_ID>/traffic-streamlit .
```

### 2. Push the image to Artifact Registry (or Container Registry)
```bash
# Enable the Artifact Registry API first if needed
gcloud artifacts repositories create my-repo --repository-format=docker --location=europe-west1

gcloud auth configure-docker europe-west1-docker.pkg.dev

docker tag gcr.io/<YOUR_PROJECT_ID>/traffic-streamlit europe-west1-docker.pkg.dev/<YOUR_PROJECT_ID>/my-repo/traffic-streamlit:latest

docker push europe-west1-docker.pkg.dev/<YOUR_PROJECT_ID>/my-repo/traffic-streamlit:latest
```

### 3. Deploy to Cloud Run
```bash
gcloud run deploy traffic-streamlit \
  --image europe-west1-docker.pkg.dev/<YOUR_PROJECT_ID>/my-repo/traffic-streamlit:latest \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=${GEMINI_API_KEY:-}
```

After deployment, Cloud Run will provide a secure HTTPS URL where the dashboard is accessible.

### 4. Verify
Open the URL in a browser; you should see the Streamlit UI with live traffic analytics.

> **Note**: Ensure the `traffic_2.mp4` video file is included in the Docker image (it is copied by the Dockerfile). If you want to use a live camera feed, set the `VIDEO_PATH` environment variable accordingly.

---