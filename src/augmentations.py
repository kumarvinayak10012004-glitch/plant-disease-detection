import tensorflow as tf

def get_augmentation_layer(img_size=(224,224)):
    """
    Returns a tf.keras.Sequential that performs common on-the-fly augmentations.
    This runs on the GPU if available and is efficient.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        # Random contrast/brightness can be added:
        tf.keras.layers.RandomContrast(0.08),
    ], name="data_augmentation")
