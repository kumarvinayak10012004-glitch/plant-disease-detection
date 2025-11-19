import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

# Default dataset folder used by the rest of the scripts:
DEFAULT_DATA_PATH = os.path.join("data", "plant_village", "color")

def get_datasets(data_dir=DEFAULT_DATA_PATH, img_size=(224, 224), batch_size=32, val_split=0.2, seed=123):
    """
    Returns: train_ds, val_ds, class_names
    Uses tf.keras.preprocessing.image_dataset_from_directory for simplicity and performance.
    """
    # Validate path
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}. Please extract PlantVillage into this folder.")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"  # integer labels (0..N-1)
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    class_names = train_ds.class_names

    # Prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
