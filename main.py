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

        # Save once per second max
        if ts_ms % 1000 < 40:
            insert_record(knee, hip, elbow, sym, risk)

        # Overlay text
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


# VIDEO PROCESSOR (LIVE)

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.monotonic()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ts_ms = int((time.monotonic() - self.start_time) * 1000)
        img, _ = process_frame(img, ts_ms)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# STREAMLIT UI

st.title("🧍 Physico Risk Detection")
st.write("Live webcam + uploaded video posture risk prediction")

tab1, tab2 = st.tabs(["📷 Live Camera", "🎥 Upload Video"])

with tab1:
    webrtc_streamer(
        key="physico",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

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

            ts_ms += 33  # ~30 FPS monotonic
            frame, risk = process_frame(frame, ts_ms)
            frame_slot.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processed and saved to history!")


# DASHBOARD

st.divider()
st.subheader("Dashboard")

if st.button("Load History"):
    hist = load_history()

    if len(hist):
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        hist["date"] = hist["timestamp"].dt.date
        hist["day"] = hist["timestamp"].dt.day_name()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(hist))
        c2.metric("Avg Symmetry", round(hist["symmetry"].mean(), 2))
        c3.metric("High Risk %", round((hist["risk"]=="High").mean()*100, 1))

        r1, r2 = st.columns(2)
        with r1:
            st.subheader("📈 Daily Angles")
            daily = hist.groupby("date")[["knee","hip","elbow","symmetry"]].mean()
            st.line_chart(daily)

        with r2:
            st.subheader("⚠️ Risk Distribution")
            st.bar_chart(hist["risk"].value_counts())

        r3, r4 = st.columns(2)
        with r3:
            st.subheader("📅 Weekly Activity")
            day_counts = hist.groupby("day").size().reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                fill_value=0
            )
            st.bar_chart(day_counts)

        with r4:
            st.subheader("🏆 Performance Score")
            hist["score"] = (
                hist["symmetry"] * 100
                - abs(hist["knee"] - 120)
                - abs(hist["hip"] - 100)
                - abs(hist["elbow"] - 90)
            )
            daily_score = hist.groupby("date")["score"].mean()
            st.line_chart(daily_score)

        streak = calculate_streak(hist["date"].tolist())
        st.metric("🔥 Max Streak (Days)", streak)

        if st.button("Export CSV"):
            hist.to_csv("physico_history.csv", index=False)
            st.success("Downloaded physico_history.csv")

    else:
        st.info("No history data yet. Start a session.")



# import streamlit as st
# import cv2
# import numpy as np
# import joblib
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import av
# import sqlite3
# import pandas as pd
# from datetime import datetime, date

# st.set_page_config(page_title="Physico Risk Detection", layout="wide")

# # =========================
# # DATABASE SETUP
# # =========================

# DB_PATH = "physico_history.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS session_data (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT,
#             knee REAL,
#             hip REAL,
#             elbow REAL,
#             symmetry REAL,
#             risk TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# def insert_record(knee, hip, elbow, symmetry, risk):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute("""
#         INSERT INTO session_data (timestamp, knee, hip, elbow, symmetry, risk)
#         VALUES (?, ?, ?, ?, ?, ?)
#     """, (datetime.now().isoformat(), knee, hip, elbow, symmetry, risk))
#     conn.commit()
#     conn.close()

# def load_history():
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql("SELECT * FROM session_data ORDER BY timestamp", conn)
#     conn.close()
#     return df

# init_db()

# # =========================
# # LOAD ML MODELS
# # =========================

# @st.cache_resource
# def load_models():
#     model = joblib.load("physico_risk_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     label_encoder = joblib.load("label_encoder.pkl")
#     return model, scaler, label_encoder

# model, scaler, label_encoder = load_models()

# # =========================
# # LOAD MEDIAPIPE TASK
# # =========================

# @st.cache_resource
# def load_landmarker():
#     base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         running_mode=vision.RunningMode.VIDEO,
#         num_poses=1
#     )
#     return vision.PoseLandmarker.create_from_options(options)

# landmarker = load_landmarker()

# # =========================
# # HELPER FUNCTIONS
# # =========================

# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def get_symmetry(l, r):
#     return 1.0 if l+r == 0 else 1 - abs(l-r)/(l+r)

# def calculate_streak(dates):
#     if len(dates) == 0:
#         return 0
#     unique_days = sorted(set(dates))
#     streak = 1
#     max_streak = 1
#     for i in range(1, len(unique_days)):
#         delta = (unique_days[i] - unique_days[i-1]).days
#         if delta == 1:
#             streak += 1
#             max_streak = max(max_streak, streak)
#         else:
#             streak = 1
#     return max_streak

# # =========================
# # VIDEO PROCESSOR
# # =========================

# class PoseProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.timestamp = 0

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#         result = landmarker.detect_for_video(mp_image, self.timestamp)
#         self.timestamp += 1

#         if result.pose_landmarks:
#             lm = result.pose_landmarks[0]

#             def pt(i): return [lm[i].x, lm[i].y]

#             lh, lk, la = pt(23), pt(25), pt(27)
#             rh, rk, ra = pt(24), pt(26), pt(28)
#             ls, le, lw = pt(11), pt(13), pt(15)
#             rs, re, rw = pt(12), pt(14), pt(16)

#             lk_ang = calculate_angle(lh, lk, la)
#             rk_ang = calculate_angle(rh, rk, ra)
#             lh_ang = calculate_angle(ls, lh, lk)
#             rh_ang = calculate_angle(rs, rh, rk)
#             le_ang = calculate_angle(ls, le, lw)
#             re_ang = calculate_angle(rs, re, rw)

#             knee = (lk_ang + rk_ang) / 2
#             hip = (lh_ang + rh_ang) / 2
#             elbow = (le_ang + re_ang) / 2
#             sym = np.mean([
#                 get_symmetry(lk_ang, rk_ang),
#                 get_symmetry(lh_ang, rh_ang),
#                 get_symmetry(le_ang, re_ang)
#             ])

#             X = np.array([[knee, hip, elbow, sym]])
#             Xs = scaler.transform(X)
#             pred = model.predict(Xs)
#             risk = label_encoder.inverse_transform(pred)[0]

#             # SAVE TO DATABASE every 15 frames
#             if self.timestamp % 15 == 0:
#                 insert_record(knee, hip, elbow, sym, risk)

#             # Overlay text
#             cv2.putText(img, f"Knee: {int(knee)}", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.putText(img, f"Hip: {int(hip)}", (20, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.putText(img, f"Elbow: {int(elbow)}", (20, 100),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.putText(img, f"Symmetry: {sym:.2f}", (20, 130),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#             color = (0,255,0) if risk=="Low" else (0,255,255) if risk=="Medium" else (0,0,255)
#             cv2.putText(img, f"Risk: {risk}", (20, 180),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # =========================
# # STREAMLIT UI
# # =========================

# st.title("🧍 Physico Risk Detection (MediaPipe + ML)")
# st.write("Live posture risk prediction with history tracking and streaks.")

# webrtc_streamer(
#     key="physico",
#     video_processor_factory=PoseProcessor,
#     media_stream_constraints={"video": True, "audio": False}
# )

# st.divider()
# st.subheader("📈 Session History Dashboard")

# if st.button("Load History"):
#     hist = load_history()

#     if len(hist) > 0:
#         hist["timestamp"] = pd.to_datetime(hist["timestamp"])
#         hist["date"] = hist["timestamp"].dt.date
#         hist["day"] = hist["timestamp"].dt.day_name()

#         st.write("Total Records:", len(hist))

#         # DAILY SUMMARY
#         daily = hist.groupby("date").agg({
#             "knee": "mean",
#             "hip": "mean",
#             "elbow": "mean",
#             "symmetry": "mean"
#         }).reset_index()

#         st.subheader("📅 Daily Performance")
#         st.line_chart(daily.set_index("date")[["knee","hip","elbow","symmetry"]])

#         # RISK DISTRIBUTION
#         st.subheader("⚠️ Risk Distribution")
#         st.bar_chart(hist["risk"].value_counts())

#         # WEEKLY STREAK
#         st.subheader("🔥 Weekly Consistency (Mon–Sun)")
#         day_counts = hist.groupby("day").size().reindex(
#             ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
#             fill_value=0
#         )
#         st.bar_chart(day_counts)

#         # PERFORMANCE SCORE
#         hist["score"] = (
#             hist["symmetry"] * 100
#             - abs(hist["knee"] - 120)
#             - abs(hist["hip"] - 100)
#             - abs(hist["elbow"] - 90)
#         )

#         daily_score = hist.groupby("date")["score"].mean().reset_index()

#         st.subheader("🏆 Daily Performance Score")
#         st.line_chart(daily_score.set_index("date")["score"])

#         streak = calculate_streak(daily["date"].tolist())
#         st.metric("🔥 Max Streak (Days)", streak)

#         # EXPORT BUTTON
#         if st.button("Export History CSV"):
#             hist.to_csv("physico_history.csv", index=False)
#             st.success("Saved as physico_history.csv")

#     else:
#         st.info("No history data yet. Start a session to collect data.")



# import streamlit as st
# import cv2
# import numpy as np
# import joblib
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import av
# import sqlite3
# import pandas as pd
# from datetime import datetime
# from collections import deque

# st.set_page_config(page_title="Physico AI Trainer", layout="wide")

# DB_PATH = "physico_history.db"

# # =========================
# # DATABASE
# # =========================

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()

#     c.execute("""
#         CREATE TABLE IF NOT EXISTS session_data (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT,
#             knee REAL,
#             hip REAL,
#             elbow REAL,
#             symmetry REAL,
#             risk TEXT,
#             injury_prob REAL,
#             reps INTEGER
#         )
#     """)

#     conn.commit()
#     conn.close()

# def insert_record(knee, hip, elbow, symmetry, risk, injury_prob, reps):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute("""
#         INSERT INTO session_data
#         (timestamp, knee, hip, elbow, symmetry, risk, injury_prob, reps)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#     """, (
#         datetime.now().isoformat(),
#         float(knee), float(hip), float(elbow),
#         float(symmetry), str(risk),
#         float(injury_prob), int(reps)
#     ))
#     conn.commit()
#     conn.close()

# def load_history():
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql("SELECT * FROM session_data ORDER BY timestamp", conn)
#     conn.close()

#     for col in ["knee","hip","elbow","symmetry","risk","injury_prob","reps"]:
#         if col not in df.columns:
#             df[col] = 0.0 if col != "reps" else 0
#     return df

# init_db()

# # =========================
# # ML MODELS
# # =========================

# @st.cache_resource
# def load_models():
#     model = joblib.load("physico_risk_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     label_encoder = joblib.load("label_encoder.pkl")
#     return model, scaler, label_encoder

# model, scaler, label_encoder = load_models()

# # =========================
# # MEDIAPIPE
# # =========================

# @st.cache_resource
# def load_landmarker():
#     base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         running_mode=vision.RunningMode.VIDEO,
#         num_poses=1
#     )
#     return vision.PoseLandmarker.create_from_options(options)

# landmarker = load_landmarker()

# # =========================
# # HELPERS
# # =========================

# FEATURE_NAMES = ["knee", "hip", "elbow", "symmetry"]

# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# def symmetry_score(l, r):
#     return 1.0 if (l+r)==0 else 1 - abs(l-r)/(l+r)

# def injury_probability(knee, hip, elbow, symmetry):
#     prob = 0
#     if knee < 100: prob += 0.25
#     if hip < 90: prob += 0.25
#     if elbow < 80: prob += 0.20
#     if symmetry < 0.8: prob += 0.30
#     return min(prob,1.0) * 100

# def calculate_streak(dates):
#     if not dates: return 0
#     unique = sorted(set(dates))
#     streak = max_streak = 1
#     for i in range(1,len(unique)):
#         if (unique[i]-unique[i-1]).days == 1:
#             streak += 1
#             max_streak = max(max_streak, streak)
#         else:
#             streak = 1
#     return max_streak

# # =========================
# # VIDEO PROCESSOR
# # =========================

# class PoseProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.timestamp = 0
#         self.rep_state = "down"
#         self.rep_count = 0
#         self.knee_buffer = deque(maxlen=10)

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#         result = landmarker.detect_for_video(mp_image, self.timestamp)
#         self.timestamp += 1

#         if result.pose_landmarks:
#             lm = result.pose_landmarks[0]
#             def pt(i): return [lm[i].x, lm[i].y]

#             lh, lk, la = pt(23), pt(25), pt(27)
#             rh, rk, ra = pt(24), pt(26), pt(28)
#             ls, le, lw = pt(11), pt(13), pt(15)
#             rs, re, rw = pt(12), pt(14), pt(16)

#             knee = (calculate_angle(lh, lk, la) + calculate_angle(rh, rk, ra))/2
#             hip = (calculate_angle(ls, lh, lk) + calculate_angle(rs, rh, rk))/2
#             elbow = (calculate_angle(ls, le, lw) + calculate_angle(rs, re, rw))/2

#             sym = np.mean([
#                 symmetry_score(lk[1], rk[1]),
#                 symmetry_score(lh[1], rh[1]),
#                 symmetry_score(le[1], re[1])
#             ])

#             self.knee_buffer.append(knee)
#             smooth_knee = np.mean(self.knee_buffer)

#             if smooth_knee < 95:
#                 self.rep_state = "down"
#             if smooth_knee > 140 and self.rep_state=="down":
#                 self.rep_count += 1
#                 self.rep_state = "up"

#             X = pd.DataFrame([[knee, hip, elbow, sym]], columns=FEATURE_NAMES)
#             Xs = scaler.transform(X)
#             pred = model.predict(Xs)
#             risk = label_encoder.inverse_transform(pred)[0]

#             injury_prob = injury_probability(knee, hip, elbow, sym)

#             if self.timestamp % 20 == 0:
#                 insert_record(knee, hip, elbow, sym, risk, injury_prob, self.rep_count)

#             cv2.putText(img, f"Knee: {int(knee)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
#             cv2.putText(img, f"Hip: {int(hip)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
#             cv2.putText(img, f"Elbow: {int(elbow)}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
#             cv2.putText(img, f"Symmetry: {sym:.2f}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

#             color = (0,255,0) if risk=="Low" else (0,255,255) if risk=="Medium" else (0,0,255)
#             cv2.putText(img, f"Risk: {risk}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color,3)
#             cv2.putText(img, f"Injury: {int(injury_prob)}%", (20,210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255),2)
#             cv2.putText(img, f"Reps: {self.rep_count}", (20,250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0),3)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # =========================
# # STREAMLIT UI
# # =========================

# st.title("🧠 Physico AI Trainer")

# webrtc_streamer(
#     key="physico",
#     video_processor_factory=PoseProcessor,
#     media_stream_constraints={"video": True, "audio": False}
# )

# st.divider()
# st.subheader("📊 Performance History")

# if st.button("Load History"):
#     hist = load_history()

#     if len(hist) > 0:
#         hist["timestamp"] = pd.to_datetime(hist["timestamp"])
#         hist["date"] = hist["timestamp"].dt.date
#         hist["day"] = hist["timestamp"].dt.day_name()

#         agg_cols = {}
#         for col, agg in [("knee","mean"),("hip","mean"),("elbow","mean"),
#                          ("symmetry","mean"),("injury_prob","mean"),("reps","max")]:
#             if col in hist.columns:
#                 agg_cols[col] = agg

#         daily = hist.groupby("date").agg(agg_cols).reset_index()

#         st.subheader("📈 Daily Trends")
#         st.line_chart(daily.set_index("date")[["knee","hip","elbow","symmetry"]])

#         st.subheader("🗓️ Activity Heatmap")
#         heat = hist.groupby("date").size()
#         st.bar_chart(heat)

#         st.subheader("⚠️ Risk Distribution")
#         st.bar_chart(hist["risk"].value_counts())

#         streak = calculate_streak(daily["date"].tolist())
#         st.metric("🔥 Max Streak (Days)", streak)

#         hist["score"] = hist["symmetry"]*100 - abs(hist["knee"]-120)-abs(hist["hip"]-100)-abs(hist["elbow"]-90)
#         daily_score = hist.groupby("date")["score"].mean()
#         st.subheader("🏆 Performance Score")
#         st.line_chart(daily_score)

#         if "injury_prob" in daily.columns:
#             st.subheader("🧠 Injury Probability Trend")
#             st.line_chart(daily.set_index("date")["injury_prob"])

#         if "reps" in daily.columns:
#             st.subheader("📊 Reps Trend")
#             st.bar_chart(daily.set_index("date")["reps"])

#         if st.button("Export History CSV"):
#             hist.to_csv("physico_history.csv", index=False)
#             st.success("Saved as physico_history.csv")
#     else:
#         st.info("No history yet. Start a session first.")

