import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes, img_size=(224,224), base_name="EfficientNetB0", dropout=0.3):
    """
    Build transfer learning model with pre-processing built-in for EfficientNet.
    Returns: model, base_model (so you can unfreeze later)
    """
    inputs = tf.keras.Input(shape=(*img_size, 3))

    # Use EfficientNetB0 preprocessing and base model
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model
