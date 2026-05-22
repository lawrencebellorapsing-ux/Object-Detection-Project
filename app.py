import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

# Cache the model so it doesn't reload every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# =========================
# OPTION 1: Object Counting
# =========================
enable_counting = st.checkbox("🔢 Enable Object Counting")

# =========================
# OPTION 3: Save Detected Frames
# =========================
save_detected = st.checkbox("📸 Save Detected Frames")

# Create folder for saved images
os.makedirs("saved_frames", exist_ok=True)

# Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    # =========================
    # OBJECT COUNTING
    # =========================
    if enable_counting:
        counts = {}

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name in counts:
                    counts[class_name] += 1
                else:
                    counts[class_name] = 1

        y = 30
        for obj, count in counts.items():
            text = f"{obj}: {count}"

            cv2.putText(
                annotated_frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            y += 30

    # =========================
    # SAVE DETECTED FRAMES
    # =========================
    if save_detected:
        timestamp = int(time.time())
        filename = f"saved_frames/detected_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,  # smoother performance
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)