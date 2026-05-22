import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Object Detection Project")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    st.image(image, caption="Original Image")
    st.image(gray, caption="Processed Image")
