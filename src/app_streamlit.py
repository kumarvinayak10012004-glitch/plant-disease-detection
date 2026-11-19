
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras import layers
from PIL import Image


# ----------------------------------
# APP CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "plant_disease_v2")

WEIGHTS_PATH = os.path.join(
    MODEL_DIR, "best_model_finetuned.weights.h5"
)

CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMAGE_SIZE = (224, 224)


# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(WEIGHTS_PATH):
        st.error("Model weights file not found!")
        st.stop()

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(38, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.load_weights(WEIGHTS_PATH)

    return model


# ----------------------------------
# LOAD CLASSES
# ----------------------------------

@st.cache_data
def load_classes():

    if not os.path.exists(CLASS_PATH):
        st.error("Class names file not found!")
        st.stop()

    with open(CLASS_PATH, "r") as file:
        return json.load(file)


# ----------------------------------
# IMAGE PREPROCESSING
# ----------------------------------

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    # EfficientNetB0 handles preprocessing internally.
    # Do not divide by 255.

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ----------------------------------
# LOAD RESOURCES
# ----------------------------------

model = load_model()
class_names = load_classes()


# ----------------------------------
# APP TITLE
# ----------------------------------

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image to predict "
    "its disease using Deep Learning."
)

st.info(
    "Model: EfficientNetB0 | "
    "Classes: 38 | "
    "Validation Accuracy: 95.86%"
)


# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader(
    "Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        if st.button("Predict Disease", type="primary"):

            processed_image = preprocess_image(image)

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            predicted_index = int(np.argmax(predictions[0]))

            predicted_class = class_names[predicted_index]

            score = float(predictions[0][predicted_index])

            st.success("Prediction completed!")

            st.write("### Predicted Disease")
            st.write(f"**{predicted_class}**")

            st.metric(
                "Model Score",
                f"{score * 100:.2f}%"
            )

            st.caption(
                "Model score is based on softmax output. "
                "It is not a confirmed plant disease diagnosis."
            )


# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "Automated Plant Disease Detection "
    "Using Deep Learning and EfficientNetB0"
)
