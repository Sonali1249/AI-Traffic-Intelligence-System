"""
AI Traffic Intelligence System — Streamlit Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Page config — must be FIRST Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "traffic_data.db")
VIDEO_PATH = os.path.join(BASE_DIR, "traffic_2.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    [data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.78rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 2rem !important; font-weight: 800 !important; }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* Main background */
    .main .block-container {
        background: #0a0f1e;
        padding: 1.5rem 2rem;
    }

    /* Tables */
    .dataframe thead th {
        background: #1e293b !important;
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .dataframe tbody tr { background: #0f172a !important; color: #e2e8f0 !important; }
    .dataframe tbody tr:hover { background: #1e293b !important; }

    /* Violation badges */
    .badge-speed  { background:#ef4444; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.72rem; margin:1px; }
    .badge-helmet { background:#f59e0b; color:#000; padding:2px 8px; border-radius:10px; font-size:0.72rem; margin:1px; }
    .badge-triple { background:#d946ef; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.72rem; margin:1px; }
    .badge-ok     { background:#22c55e; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.72rem; margin:1px; }

    /* Chat messages */
    .chat-msg-user { background:#3b82f6; color:#fff; padding:10px 14px; border-radius:12px 12px 0 12px; margin:6px 0; max-width:80%; float:right; clear:both; font-size:0.88rem; }
    .chat-msg-bot  { background:#1e293b; color:#e2e8f0; border:1px solid #334155; padding:10px 14px; border-radius:12px 12px 12px 0; margin:6px 0; max-width:80%; float:left; clear:both; font-size:0.88rem; }
    .chat-clearfix { clear:both; }

    /* Section headers */
    .section-title { font-size:1.1rem; font-weight:700; color:#60a5fa; margin-bottom:12px; border-bottom:1px solid #1e3a5f; padding-bottom:6px; }

    /* Status pill */
    .status-live   { background:#22c55e; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .status-paused { background:#f59e0b; color:#000; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .status-clear  { background:#64748b; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .status-low    { background:#22c55e; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .status-medium { background:#f59e0b; color:#000; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
    .status-high   { background:#ef4444; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }

    h1, h2, h3 { color: #f1f5f9 !important; }
    hr { border-color: #1e293b; }
    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white; border-radius: 8px; border: none;
        font-weight: 600; padding: 8px 20px;
    }
    .stButton button:hover { background: linear-gradient(135deg, #60a5fa, #3b82f6); }
    .stTextInput input { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_recent_logs(limit=50):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(
            f"SELECT * FROM vehicle_logs ORDER BY id DESC LIMIT {limit}", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def fetch_stats():
    """Aggregate stats from DB for charts."""
    try:
        conn = get_db_connection()
        stats = {}
        stats['total']    = pd.read_sql_query("SELECT COUNT(*) as c FROM vehicle_logs", conn)['c'][0]
        stats['speeding'] = pd.read_sql_query("SELECT COUNT(*) as c FROM vehicle_logs WHERE is_speeding=1", conn)['c'][0]
        stats['avg_speed']= pd.read_sql_query("SELECT ROUND(AVG(speed),1) as s FROM vehicle_logs", conn)['s'][0] or 0
        stats['max_speed']= pd.read_sql_query("SELECT ROUND(MAX(speed),1) as s FROM vehicle_logs", conn)['s'][0] or 0
        stats['min_speed']= pd.read_sql_query("SELECT ROUND(MIN(speed),1) as s FROM vehicle_logs WHERE speed > 0", conn)['s'][0] or 0
        stats['helmet']   = pd.read_sql_query("SELECT COUNT(*) as c FROM vehicle_logs WHERE is_helmet_missing=1", conn)['c'][0]
        stats['triple']   = pd.read_sql_query("SELECT COUNT(*) as c FROM vehicle_logs WHERE rider_count > 2", conn)['c'][0]
        stats['by_type']  = pd.read_sql_query("SELECT type, COUNT(*) as count FROM vehicle_logs GROUP BY type", conn)
        stats['by_lane']  = pd.read_sql_query("SELECT lane, COUNT(*) as count FROM vehicle_logs GROUP BY lane", conn)
        stats['speed_ts'] = pd.read_sql_query(
            "SELECT timestamp, ROUND(AVG(speed),1) as avg_speed FROM vehicle_logs GROUP BY strftime('%H:%M', timestamp) ORDER BY timestamp DESC LIMIT 60", conn
        )
        conn.close()
        return stats
    except Exception as e:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Video Frame Capture (cached for performance)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    """Load YOLO model once (cached across reruns)."""
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            return YOLO(MODEL_PATH)
    except Exception:
        pass
    return None

def capture_frame_with_detection(frame_number=None):
    """Grab one frame from the video and run YOLO detection on it."""
    if not os.path.exists(VIDEO_PATH):
        return None, "Video file not found."

    model = load_yolo_model()
    cap = cv2.VideoCapture(VIDEO_PATH)

    if frame_number is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    else:
        # jump to a random middle frame so it's interesting
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 3)

    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not ret or frame is None:
        return None, "Could not read frame."

    frame = cv2.resize(frame, (1280, 720))
    H, W = frame.shape[:2]

    # Lane divider
    cv2.line(frame, (W//2, 0), (W//2, H), (255,255,255), 2)
    cv2.putText(frame, "LANE 1", (W//4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "LANE 2", (3*W//4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if model:
        results = model(frame, verbose=False, conf=0.35, imgsz=640)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    # BGR → RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, total_frames


# ──────────────────────────────────────────────────────────────────────────────
# AI Agent
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    from traffic_agent import TrafficAgent
    return TrafficAgent(DB_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 AI Traffic System")
    st.markdown("**v4.0 — Indian Enterprise**")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Live Dashboard", "📈 Analytics", "🗄️ Vehicle Logs", "🤖 AI Chat Agent", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()

    # Gemini key input
    st.markdown("### 🔑 Gemini API Key")
    gemini_key = st.text_input("Enter key (optional)", type="password",
                               value=os.getenv("GEMINI_API_KEY", ""),
                               help="Enables intelligent NL queries via Google Gemini")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.success("Key set ✓")

    st.divider()
    st.markdown("**System Info**")
    db_exists = os.path.exists(DB_PATH)
    video_exists = os.path.exists(VIDEO_PATH)
    st.markdown(f"{'✅' if db_exists else '❌'} Database")
    st.markdown(f"{'✅' if video_exists else '❌'} Video file")
    st.markdown(f"{'✅' if os.path.exists(MODEL_PATH) else '❌'} YOLOv8 model")
    st.divider()
    st.caption("Built by **Sonali Tiwari**\nAI Engineer & Full Stack Dev")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Live Dashboard
# ──────────────────────────────────────────────────────────────────────────────
if "📊 Live Dashboard" in page:
    st.markdown("# 🚦 AI Traffic Intelligence System")
    st.markdown("*Real-time vehicle detection • Speed enforcement • Violation tracking*")
    st.divider()

    stats = fetch_stats()
    if stats:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Vehicles", f"{stats['total']:,}")
        c2.metric("Speeding Violations", f"{stats['speeding']:,}", delta=f"+{stats['speeding']}", delta_color="inverse")
        c3.metric("Avg Speed", f"{stats['avg_speed']} km/h")
        c4.metric("Max Speed", f"{stats['max_speed']} km/h", delta="⚡")
        c5.metric("No Helmet", f"{stats['helmet']:,}", delta_color="inverse")
        c6.metric("Triple Riding", f"{stats['triple']:,}", delta_color="inverse")
    else:
        st.info("No vehicle data yet. Run `python app.py` to start the live analysis engine, then refresh.")

    st.divider()

    # ── Video Frame Preview ──────────────────────────────────────────────────
    col_vid, col_ctrl = st.columns([3, 1])
    with col_vid:
        st.markdown('<div class="section-title">📷 Video Preview (Snapshot with Detection)</div>', unsafe_allow_html=True)

        if "frame_number" not in st.session_state:
            st.session_state.frame_number = None

        if os.path.exists(VIDEO_PATH):
            frame_img, total_or_err = capture_frame_with_detection(st.session_state.frame_number)
            if frame_img is not None:
                st.image(frame_img, use_column_width=True, caption=f"YOLOv8 Detection Preview | Total frames: {total_or_err}")
            else:
                st.error(total_or_err)
        else:
            st.warning(f"Video not found at `{VIDEO_PATH}`. Please add `traffic_2.mp4` to run detection preview.")

    with col_ctrl:
        st.markdown('<div class="section-title">🎮 Controls</div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Frame"):
            st.session_state.frame_number = None
            st.rerun()

        if os.path.exists(VIDEO_PATH):
            cap_temp = cv2.VideoCapture(VIDEO_PATH)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_temp.release()
            frame_seek = st.slider("Seek Frame", 0, max(total_frames-1, 1),
                                   st.session_state.frame_number or total_frames//3)
            if st.button("📍 Go to Frame"):
                st.session_state.frame_number = frame_seek
                st.rerun()

        st.divider()
        st.markdown("**🔴 Live Flask Server**")
        st.markdown("For the full live MJPEG stream, run:")
        st.code("python app.py", language="bash")
        st.markdown("Then open: [http://localhost:5000](http://localhost:5000)")

        st.divider()
        st.markdown("**Traffic Density**")
        if stats:
            total_v = stats['total']
            density = "CLEAR" if total_v == 0 else "LOW" if total_v <= 100 else "MEDIUM" if total_v <= 500 else "HIGH"
            color_map = {"CLEAR": "status-clear","LOW":"status-low","MEDIUM":"status-medium","HIGH":"status-high"}
            st.markdown(f'<span class="{color_map[density]}">{density}</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Analytics
# ──────────────────────────────────────────────────────────────────────────────
elif "📈 Analytics" in page:
    st.markdown("# 📈 Traffic Analytics")
    st.divider()

    stats = fetch_stats()
    if not stats or stats['total'] == 0:
        st.info("No data yet. Start the live engine with `python app.py` and wait for vehicles to be detected.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🚗 Vehicle Type Breakdown</div>', unsafe_allow_html=True)
        if not stats['by_type'].empty:
            fig = px.pie(stats['by_type'], names='type', values='count',
                         color_discrete_sequence=px.colors.sequential.Blues_r,
                         hole=0.45)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🛣️ Lane Distribution</div>', unsafe_allow_html=True)
        if not stats['by_lane'].empty:
            lane_df = stats['by_lane'].copy()
            lane_df['lane'] = lane_df['lane'].apply(lambda x: f"Lane {x}")
            fig2 = px.bar(lane_df, x='lane', y='count',
                          color='count',
                          color_continuous_scale='Blues',
                          text='count')
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
                font_color='#e2e8f0', margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b')
            )
            fig2.update_traces(textposition='outside', marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">📉 Speed Over Time</div>', unsafe_allow_html=True)
    if not stats['speed_ts'].empty:
        fig3 = px.line(stats['speed_ts'], x='timestamp', y='avg_speed',
                       labels={'avg_speed': 'Avg Speed (km/h)', 'timestamp': 'Time'},
                       color_discrete_sequence=['#3b82f6'])
        fig3.add_hline(y=60, line_dash="dash", line_color="#ef4444",
                       annotation_text="Speed Limit 60 km/h", annotation_font_color="#ef4444")
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
            font_color='#e2e8f0', margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b')
        )
        fig3.update_traces(line_width=2.5)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Speed Recorded", f"{stats['min_speed']} km/h")
    c2.metric("Max Speed Recorded", f"{stats['max_speed']} km/h")
    c3.metric("Avg Speed Overall",  f"{stats['avg_speed']} km/h")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Vehicle Logs
# ──────────────────────────────────────────────────────────────────────────────
elif "🗄️ Vehicle Logs" in page:
    st.markdown("# 🗄️ Vehicle Logs Database")
    st.divider()

    limit = st.slider("Show last N records", 10, 200, 50, step=10)
    df = fetch_recent_logs(limit)

    if df.empty:
        st.info("No logs yet. Start the Flask server to begin recording vehicle data.")
    else:
        # Filter controls
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            vtype_filter = st.multiselect("Vehicle Type", options=sorted(df['type'].unique().tolist()), default=None)
        with col_f2:
            speed_min, speed_max = st.slider("Speed Range (km/h)", 0.0, float(df['speed'].max() or 200), (0.0, float(df['speed'].max() or 200)))
        with col_f3:
            viol_only = st.checkbox("Violations Only", value=False)

        filtered = df.copy()
        if vtype_filter:
            filtered = filtered[filtered['type'].isin(vtype_filter)]
        filtered = filtered[(filtered['speed'] >= speed_min) & (filtered['speed'] <= speed_max)]
        if viol_only:
            filtered = filtered[(filtered['is_speeding'] == 1) | (filtered['is_helmet_missing'] == 1) | (filtered['rider_count'] > 2)]

        st.markdown(f"**Showing {len(filtered)} records**")

        # Render a nice HTML table
        rows_html = ""
        for _, row in filtered.iterrows():
            badges = ""
            if row.get('is_speeding'):    badges += '<span class="badge-speed">SPEED</span> '
            if row.get('is_helmet_missing'): badges += '<span class="badge-helmet">NO HELMET</span> '
            if row.get('rider_count', 1) > 2: badges += '<span class="badge-triple">TRIPLE</span> '
            if not badges:                badges = '<span class="badge-ok">OK</span>'
            spd_color = "color:#ef4444;font-weight:700" if row.get('is_speeding') else "color:#22c55e"
            rows_html += f"""
            <tr>
              <td>#{row['vehicle_id']}</td>
              <td style="text-transform:capitalize">{row['type']}</td>
              <td>Lane {row['lane']}</td>
              <td style="{spd_color}">{round(row['speed'],1)}</td>
              <td><code style="background:#1e293b;padding:2px 6px;border-radius:4px">{row.get('plate_text','—')}</code></td>
              <td>{badges}</td>
              <td style="color:#64748b;font-size:0.8rem">{str(row.get('timestamp',''))[:19]}</td>
            </tr>"""

        table_html = f"""
        <style>
          .vlog-table {{ width:100%; border-collapse:collapse; background:#0f172a; }}
          .vlog-table th {{ background:#1e293b; color:#94a3b8; padding:10px 12px; text-align:left; font-size:0.8rem; font-weight:600; border-bottom:1px solid #334155; }}
          .vlog-table td {{ padding:9px 12px; color:#e2e8f0; font-size:0.85rem; border-bottom:1px solid #1e293b; }}
          .vlog-table tr:hover td {{ background:#1a2744; }}
        </style>
        <div style="overflow-x:auto; max-height:520px; overflow-y:auto; border-radius:10px; border:1px solid #1e293b;">
          <table class="vlog-table">
            <thead><tr>
              <th>ID</th><th>Type</th><th>Lane</th><th>Speed (km/h)</th><th>Plate</th><th>Violations</th><th>Timestamp</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""
        st.markdown(table_html, unsafe_allow_html=True)

        st.divider()
        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Export as CSV", csv, "traffic_logs.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: AI Chat Agent
# ──────────────────────────────────────────────────────────────────────────────
elif "🤖 AI Chat Agent" in page:
    st.markdown("# 🤖 AI Traffic Analyst Chat")
    st.markdown("Ask questions about traffic data in natural language. Powered by **Gemini AI** (if key provided) or **keyword fallback**.")
    st.divider()

    agent = get_agent()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "bot", "text": "👋 Hello! I'm the Traffic AI Analyst. Ask me anything about traffic data.\n\n"
             "Try: *'How many speeding vehicles?'*, *'What is the average speed?'*, *'Show vehicle type breakdown'*, *'What is the max speed recorded?'*"}
        ]

    # Render chat
    chat_html = '<div style="max-height:420px;overflow-y:auto;padding:10px;background:#0f172a;border-radius:12px;border:1px solid #1e293b;margin-bottom:12px">'
    for msg in st.session_state.chat_history:
        cls = "chat-msg-user" if msg["role"] == "user" else "chat-msg-bot"
        chat_html += f'<div class="{cls}">{msg["text"]}</div><div class="chat-clearfix"></div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input("Ask a question:", label_visibility="collapsed",
                                   placeholder="e.g. How many vehicles are speeding?",
                                   key="chat_input")
    with col_btn:
        send = st.button("Send 🚀")

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.spinner("Analyzing..."):
            result = agent.analyze(user_input)
        bot_reply = result.get("response", "Sorry, I couldn't process that.")
        st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
        st.rerun()

    with st.expander("💡 Example Questions"):
        samples = [
            "How many vehicles have been detected?",
            "How many speeding violations?",
            "What is the average speed?",
            "Show me the fastest car",
            "Show slowest vehicle",
            "Lane traffic breakdown",
            "Count cars vs trucks",
            "How many helmet violations?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(samples):
            if cols[i % 2].button(q, key=f"sample_{i}"):
                st.session_state.chat_history.append({"role": "user", "text": q})
                result = agent.analyze(q)
                st.session_state.chat_history.append({"role": "bot", "text": result.get("response", "—")})
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Settings
# ──────────────────────────────────────────────────────────────────────────────
elif "⚙️ Settings" in page:
    st.markdown("# ⚙️ System Settings")
    st.divider()

    st.markdown("### 🎯 Detection Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Speed Limit (km/h)", value=60, min_value=20, max_value=200, step=5,
                        help="Vehicles exceeding this are flagged as speeding")
        st.number_input("YOLO Confidence Threshold", value=0.35, min_value=0.1, max_value=0.95, step=0.05,
                        help="Minimum confidence for a detection to be counted")
    with col2:
        st.number_input("Pixels Per Meter", value=8, min_value=1, max_value=50,
                        help="Calibration factor for speed estimation")
        st.number_input("Tracking Distance Threshold (px)", value=70, min_value=10, max_value=300,
                        help="Max pixel distance to consider same vehicle across frames")

    st.info("⚠️ These settings are for reference. To change them, edit the constants at the top of `traffic_core.py`.")
    st.divider()

    st.markdown("### 📁 File Paths")
    st.code(f"Video:    {VIDEO_PATH}\nDatabase: {DB_PATH}\nModel:    {MODEL_PATH}", language="text")

    st.divider()
    st.markdown("### 🗑️ Database Actions")
    if st.button("🗑️ Clear All Vehicle Logs", type="secondary"):
        try:
            conn = get_db_connection()
            conn.execute("DELETE FROM vehicle_logs")
            conn.commit()
            conn.close()
            st.success("All vehicle logs cleared.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.markdown("### ℹ️ How to Run")
    st.markdown("""
**Option 1: Streamlit (this dashboard)**
```bash
streamlit run streamlit_app.py
```

**Option 2: Flask (full live MJPEG stream)**
```bash
python app.py
# Open http://localhost:5000
```

**For Gemini AI Agent**, set your API key:
```bash
# Windows
set GEMINI_API_KEY=your_key_here

# Linux/Mac  
export GEMINI_API_KEY=your_key_here
```
Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)
""")
