import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# =====================
# Base directory
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "classes.txt")

IMAGE_SIZE = (224, 224)

# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at:\n{MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

# =====================
# Load classes
# =====================
def load_classes():
    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, "r") as f:
            return f.read().splitlines()
    return []

# =====================
# Image preprocessing
# =====================
def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================
# Streamlit UI
# =====================
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to predict the disease")

model = load_model()
class_names = load_classes()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    label = class_names[idx] if class_names else f"Class {idx}"

    st.success(f"🦠 Disease: **{label}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

