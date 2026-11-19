
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(
    num_classes,
    img_size=(224, 224),
    dropout=0.3
):
    """
    Build an EfficientNetB0 transfer learning model.

    Returns:
        model
        base_model
    """

    inputs = tf.keras.Input(
        shape=(*img_size, 3),
        name="input_image"
    )

    # EfficientNetB0 with ImageNet pretrained weights
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3)
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    # Feature extraction
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        dropout,
        name="dropout"
    )(x)

    # Output layer for all dataset classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="PlantDisease_EfficientNetB0"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model
