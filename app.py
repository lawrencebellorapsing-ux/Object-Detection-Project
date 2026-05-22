import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.title("Real-Time Object Detection and Tracking")

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

run = st.checkbox("Start Camera")

frame_window = st.image([])

while run:
    success, frame = camera.read()

    if not success:
        st.write("Failed to access camera")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    frame_window.image(
        annotated_frame,
        channels="BGR"
    )

camera.release()
