import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import sqlite3
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="Physico Risk Detection", layout="wide")

DB_PATH = "physico_history.db"


# DATABASE SETUP


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            knee REAL,
            hip REAL,
            elbow REAL,
            symmetry REAL,
            risk TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_record(knee, hip, elbow, symmetry, risk):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO session_data (timestamp, knee, hip, elbow, symmetry, risk)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), knee, hip, elbow, symmetry, risk))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM session_data ORDER BY timestamp", conn)
    conn.close()
    return df

init_db()


# LOAD ML MODELS


@st.cache_resource
def load_models():
    model = joblib.load("physico_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()


# LOAD MEDIAPIPE TASK


@st.cache_resource
def create_landmarker():
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)

landmarker = create_landmarker()


# HELPER FUNCTIONS


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_symmetry(l, r):
    return 1.0 if l + r == 0 else 1 - abs(l - r) / (l + r)

def calculate_streak(dates):
    if len(dates) == 0:
        return 0
    unique_days = sorted(set(dates))
    streak = 1
    max_streak = 1
    for i in range(1, len(unique_days)):
        delta = (unique_days[i] - unique_days[i-1]).days
        if delta == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    return max_streak

# FRAME PROCESSING


def process_frame(img, ts_ms):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, ts_ms)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        def pt(i): return [lm[i].x, lm[i].y]

        lh, lk, la = pt(23), pt(25), pt(27)
        rh, rk, ra = pt(24), pt(26), pt(28)
        ls, le, lw = pt(11), pt(13), pt(15)
        rs, re, rw = pt(12), pt(14), pt(16)

        lk_ang = calculate_angle(lh, lk, la)
        rk_ang = calculate_angle(rh, rk, ra)
        lh_ang = calculate_angle(ls, lh, lk)
        rh_ang = calculate_angle(rs, rh, rk)
        le_ang = calculate_angle(ls, le, lw)
        re_ang = calculate_angle(rs, re, rw)

        knee = (lk_ang + rk_ang) / 2
        hip = (lh_ang + rh_ang) / 2
        elbow = (le_ang + re_ang) / 2
        sym = np.mean([
            get_symmetry(lk_ang, rk_ang),
            get_symmetry(lh_ang, rh_ang),
            get_symmetry(le_ang, re_ang)
        ])

        X = np.array([[knee, hip, elbow, sym]])
        Xs = scaler.transform(X)
        pred = model.predict(Xs)
        risk = label_encoder.inverse_transform(pred)[0]

        # Save once per second
        if ts_ms % 1000 < 40:
            insert_record(knee, hip, elbow, sym, risk)

        # Overlay
        cv2.putText(img, f"Knee: {int(knee)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, f"Hip: {int(hip)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, f"Elbow: {int(elbow)}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, f"Symmetry: {sym:.2f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        color = (0,255,0) if risk=="Low" else (0,255,255) if risk=="Medium" else (0,0,255)
        cv2.putText(img, f"Risk: {risk}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        return img, risk

    return img, None

# LIVE VIDEO PROCESSOR


class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.monotonic()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ts_ms = int((time.monotonic() - self.start_time) * 1000)
        img, _ = process_frame(img, ts_ms)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# UI HEADER


st.title(" Physico Risk Detection")
st.write("Live webcam + uploaded video posture risk prediction")

tab1, tab2 = st.tabs(["📷 Live Camera", "🎥 Upload Video"])


# LIVE CAMERA


with tab1:
    webrtc_streamer(
        key="physico",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# UPLOAD VIDEO


with tab2:
    uploaded_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi"])
    if uploaded_file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        landmarker.close()
        landmarker = create_landmarker()

        cap = cv2.VideoCapture(temp_path)
        frame_slot = st.empty()
        ts_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms += 33
            frame, risk = process_frame(frame, ts_ms)
            frame_slot.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processed and saved to history!")

# DASHBOARD


st.divider()
st.subheader(" Dashboard")

if st.button("Load History"):
    hist = load_history()

    if len(hist):
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        hist["date"] = hist["timestamp"].dt.date

        
        # FILTER PANEL
        
        st.subheader("🎛 Filters")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            start_date = st.date_input("Start Date", hist["date"].min())
        with c2:
            end_date = st.date_input("End Date", hist["date"].max())
        with c3:
            risk_filter = st.multiselect(
                "Risk Level",
                options=hist["risk"].unique().tolist(),
                default=hist["risk"].unique().tolist()
            )
        with c4:
            view_mode = st.selectbox("View Mode", ["Sessions", "Daily Average"])

        filtered = hist[
            (hist["date"] >= start_date) &
            (hist["date"] <= end_date) &
            (hist["risk"].isin(risk_filter))
        ]

        
        # METRICS
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", len(filtered))
        m2.metric("Avg Symmetry", round(filtered["symmetry"].mean(), 2))
        m3.metric("High Risk %", round((filtered["risk"]=="High").mean()*100, 1))
        streak = calculate_streak(filtered["date"].tolist())
        m4.metric("🔥 Max Streak", streak)

        
        # TREND CHART
       
        st.subheader("Performance Trend")
        metric = st.selectbox("Metric", ["knee","hip","elbow","symmetry"])

        if view_mode == "Daily Average":
            chart_data = filtered.groupby("date")[metric].mean()
        else:
            chart_data = filtered[metric]

        st.line_chart(chart_data)

       
        # RISK DISTRIBUTION
       
        st.subheader("⚠️ Risk Distribution")
        st.bar_chart(filtered["risk"].value_counts())

      
        # ACTIVITY CHART
       
        st.subheader("📅 Activity by Date")
        daily_counts = filtered.groupby("date").size()
        st.bar_chart(daily_counts)

       
        # SCORE TUNING
       
        st.subheader("🏆 Performance Score Tuning")
        c1, c2, c3 = st.columns(3)
        ideal_knee = c1.slider("Ideal Knee", 90, 160, 120)
        ideal_hip = c2.slider("Ideal Hip", 80, 140, 100)
        ideal_elbow = c3.slider("Ideal Elbow", 60, 140, 90)

        filtered["score"] = (
            filtered["symmetry"] * 100
            - abs(filtered["knee"] - ideal_knee)
            - abs(filtered["hip"] - ideal_hip)
            - abs(filtered["elbow"] - ideal_elbow)
        )

        daily_score = filtered.groupby("date")["score"].mean()
        st.line_chart(daily_score)

       
        # EXPORT BUTTON
        
        if st.button("Export Filtered CSV"):
            filtered.to_csv("physico_filtered_history.csv", index=False)
            st.success("Downloaded physico_filtered_history.csv")

    else:
        st.info("No history data yet. Start a session.")
