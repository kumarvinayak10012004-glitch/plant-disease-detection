import streamlit as st
import numpy as np
from PIL import Image
import json
import io
import os
import tensorflow as tf
from src.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image


@st.cache_resource
def load_model_and_assets(model_path='models/best_model.h5'):
model = tf.keras.models.load_model(model_path)
class_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
if os.path.exists(class_path):
with open(class_path) as f:
class_names = json.load(f)
else:
class_names = []
# recommendations.json optional
rec_path = os.path.join(os.path.dirname(model_path), 'recommendations.json')
if os.path.exists(rec_path):
with open(rec_path) as f:
recs = json.load(f)
else:
recs = {}
return model, class_names, recs


model, class_names, recs = load_model_and_assets()


st.title('Plant Disease Detection')
st.write('Upload a leaf image and the model will predict the disease.')


uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg'])
if uploaded is not None:
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
st.image(img, caption='Uploaded', use_column_width=True)


# Preprocess
img_resized = img.resize((224,224))
x = np.array(img_resized)[None, ...]
x = tf.keras.applications.efficientnet.preprocess_input(x.astype('float32'))


preds = model.predict(x)
top_idx = np.argmax(preds[0])
top_conf = float(preds[0][top_idx])
label = class_names[top_idx] if class_names else str(top_idx)


st.markdown(f'**Prediction:** {label}
**Confidence:** {top_conf:.2%}')


# Recommendation
rec = recs.get(label, {}).get('action', 'No recommendation available.')
st.write('**Recommendation:**')
st.write(rec)


# Grad-CAM
try:
last_conv = None
# heuristics to find last conv layer name
for layer in reversed(model.layers):
if 'conv' in layer.name:
last_conv = layer.name
break
if last_conv is None:
last_conv = model.layers[-3].name


heatmap = make_gradcam_heatmap(x, model, last_conv)
overlay = overlay_heatmap_on_image(img_resized, heatmap)
st.image(overlay, caption='Grad-CAM Overlay', use_column_width=True)
except Exception as e:
st.write('Grad-CAM failed:', e)