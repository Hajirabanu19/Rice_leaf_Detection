import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
from keras.models import load_model
import tensorflow as tf
from PIL import Image

# ---------------------------------
# Load CNN model with Data Augmentation
# ---------------------------------
model = load_model("cnn_model_aug.keras")

# Class names (must match training order)
class_names = ['Leaf Blast', 'Bacterial Blight', 'Brown Spot']

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")

st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image to predict the disease")

uploaded_file = st.file_uploader(
    "Choose a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Predicted Disease: **{predicted_class}**")
    st.info(f"Prediction Confidence: **{confidence:.2f}%**")
