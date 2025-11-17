import streamlit as st
from ultralytics import YOLO
import cv2
import yt_dlp
import threading
import queue
import numpy as np
import time

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="🎥 YOLO Live Stream Detection", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #181818;
        color: white;
    }
    .stButton>button {
        background-color: #34A853;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        border: none;
        margin-right: 10px;
    }
    .stButton>button:hover {
        background-color: #2c8c45;
        color: white;
    }
    h1, h2, h3, p {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎥 YOLO Real-Time Object Detection")
st.caption("Detect people, cars, dogs, and more directly from a YouTube live or recorded stream.")

# -------------------- User Input --------------------
yt_url = st.text_input(
    "📺 Enter YouTube Video or Live Stream URL:",
    "https://www.youtube.com/watch?v=6dp-bvQ7RWo"
)

start_btn = st.button("▶ Start Detection")
stop_btn = st.button("⏹ Stop Detection")

frame_placeholder = st.empty()
status_text = st.empty()

# -------------------- YOLO Model --------------------
model = YOLO("yolov8n.pt")

# -------------------- Function to Get Stream URL --------------------
def get_youtube_stream(url):
    """Extracts direct video stream URL from YouTube link"""
    ydl_opts = {
        'format': 'best[height<=720]',
        'quiet': True,
        'extractor_args': {'youtube': ['player_client=default']}
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# -------------------- Detection Logic --------------------
frame_queue = queue.Queue(maxsize=1)
stop_flag = threading.Event()

def read_frames(cap):
    """Continuously read frames and store only latest one."""
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame)

def run_detection(stream_url):
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error("❌ Could not open YouTube stream.")
        return

    status_text.info("🟢 Detecting objects... Press Stop to end.")
    reader_thread = threading.Thread(target=read_frames, args=(cap,), daemon=True)
    reader_thread.start()

    target_fps = 25
    frame_interval = 1.0 / target_fps

    while not stop_flag.is_set():
        start_time = time.time()
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.resize(frame, (960, 540))

            # YOLO detection
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    cap.release()
    reader_thread.join()
    status_text.warning("⏹ Detection stopped.")

# -------------------- Stream Control --------------------
if start_btn:
    try:
        status_text.info("🔄 Connecting to stream...")
        stream_url = get_youtube_stream(yt_url)
        stop_flag.clear()
        run_detection(stream_url)
    except Exception as e:
        st.error(f"⚠️ Failed to start detection: {e}")

if stop_btn:
    stop_flag.set()
