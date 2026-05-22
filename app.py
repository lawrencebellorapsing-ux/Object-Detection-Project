import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("Real-Time Object Detection and Tracking")

model = YOLO("yolov8n.pt")

camera = st.camera_input("Open Camera")

if camera is not None:
    image = Image.open(camera)
    frame = np.array(image)

    results = model(frame)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, channels="BGR")
