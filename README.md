# AI Traffic Intelligence System

An AI-based traffic monitoring system that detects, tracks, and analyzes vehicles from traffic video and displays real-time results on a web dashboard.

## 🚀 Features
- **Real-time Vehicle Detection**: Powered by YOLOv8 to detect cars, motorcycles, buses, and trucks.
- **Vehicle Tracking**: Unique ID assignment and trajectory tracking using centroid matching.
- **Speed Estimation**: Calculates vehicle speed based on pixel movement.
- **Traffic Analysis**: Classifies traffic density (Low, Medium, Heavy) and analyzes directional flow.
- **Web Dashboard**: Interactive Flask-based dashboard with live video feed and real-time statistics.
- **Responsive Design**: Modern, dark-themed UI.

## 🛠️ Tech Stack
- **Python** (Core Logic)
- **OpenCV** (Video Processing)
- **YOLOv8** (Object Detection)
- **Flask** (Web Backend)
- **HTML/CSS/JS** (Frontend Dashboard)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Traffic-Intelligence-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Setup**
   The system uses the `yolov8n.pt` model. It will automaticallly download on the first run.

## ▶️ How to Run

1. **Start the Flask Server**
   ```bash
   python app.py
   ```

2. **Access the Dashboard**
   Open your browser and go to:
   ```
   http://localhost:5000
   ```

## 📋 Usage Notes
- **Video Source**: The system looks for `traffic_2.mp4` in the root directory by default. If not found, it attempts to use the **webcam**.
- **Dashboard**: The web interface updates statistics in real-time every second.

## 👤 Author
**Sonali Tiwari**