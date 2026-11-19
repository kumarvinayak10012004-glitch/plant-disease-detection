
import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

DEFAULT_DATA_PATH = os.path.join(
    "data", "plant_village", "color"
)


def get_datasets(
    data_dir=DEFAULT_DATA_PATH,
    img_size=(224, 224),
    batch_size=16,
    val_split=0.2,
    seed=123
):
    """
    Load PlantVillage dataset.

    Returns:
        train_ds
        val_ds
        class_names
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset path not found: {data_dir}"
        )

    print("Loading training dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    print("Loading validation dataset...")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    class_names = train_ds.class_names

    # Prefetch improves data loading efficiency.
    # Cache is intentionally omitted to reduce RAM usage.
    train_ds = train_ds.prefetch(
        buffer_size=AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        buffer_size=AUTOTUNE
    )

    print("Total classes:", len(class_names))
    print("Class names:", class_names)

    return train_ds, val_ds, class_names
