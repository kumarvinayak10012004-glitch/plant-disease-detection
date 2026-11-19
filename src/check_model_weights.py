
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "plant_disease_v2")

WEIGHTS_PATH = os.path.join(
    MODEL_DIR,
    "best_model_finetuned.weights.h5"
)

CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

# Load class names
with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

num_classes = len(class_names)

# Recreate EfficientNetB0 architecture
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(224, 224, 3)
)

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Load trained weights
model.load_weights(WEIGHTS_PATH)

print("Model weights loaded successfully!")
print("Total classes:", num_classes)

# Test with a dummy image
dummy_image = tf.zeros((1, 224, 224, 3))
prediction = model.predict(dummy_image, verbose=0)

print("Prediction output shape:", prediction.shape)
print("Model test successful!")
