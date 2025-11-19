import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed batch (1, H, W, 3) - same preprocessing used during training
    model: tf.keras Model
    last_conv_layer_name: name of last conv layer in base model
    returns heatmap resized to image size (H, W) with values 0..1
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap_on_pil(img_pil, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    img_pil: PIL.Image (RGB)
    heatmap: 2D numpy array (0..1) same aspect ratio as img_pil
    returns: PIL.Image overlay
    """
    img = np.array(img_pil)
    h, w = img.shape[:2]
    heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (w, h)))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = heatmap_color * alpha + img
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return Image.fromarray(overlay)
