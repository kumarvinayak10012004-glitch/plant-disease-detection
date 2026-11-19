
import argparse
import os
import json
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

from src.data_loader import get_datasets, DEFAULT_DATA_PATH
from src.model import build_model
from src.augmentations import get_augmentation_layer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine_tune_epochs", type=int, default=3)

    # New folder: old model files will not be overwritten
    parser.add_argument(
        "--out_dir",
        default="models/plant_disease_v2"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    print("\nLoading PlantVillage dataset...")

    train_ds, val_ds, class_names = get_datasets(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    num_classes = len(class_names)

    print("\nTotal classes:", num_classes)

    # Save exact class order
    class_path = os.path.join(
        args.out_dir,
        "class_names.json"
    )

    with open(class_path, "w") as file:
        json.dump(class_names, file, indent=4)

    print("Class names saved:", class_path)

    # Data augmentation
    augmentation = get_augmentation_layer(
        img_size=(args.img_size, args.img_size)
    )

    def augment(images, labels):
        images = augmentation(images, training=True)
        return images, labels

    train_ds_aug = train_ds.map(
        augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Build model
    print("\nBuilding EfficientNetB0 model...")

    model, base_model = build_model(
        num_classes=num_classes,
        img_size=(args.img_size, args.img_size)
    )

    model.summary()

    # Training callbacks
    best_model_path = os.path.join(
        args.out_dir,
        "best_model.keras"
    )

    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Stage 1: Train classification head
    print("\nStage 1: Training classification head...")

    model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning
    print("\nStage 2: Fine-tuning EfficientNetB0...")

    base_model.trainable = True

    # Freeze all except last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen
    for layer in base_model.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_path = os.path.join(
        args.out_dir,
        "best_model_finetuned.keras"
    )

    fine_tune_callbacks = [
        ModelCheckpoint(
            fine_tune_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=fine_tune_callbacks
    )

    # Save final model
    final_path = os.path.join(
        args.out_dir,
        "final_model.keras"
    )

    model.save(final_path)

    print("\nTraining completed successfully!")
    print("Number of classes:", num_classes)
    print("Final model:", final_path)
    print("Class names:", class_path)


if __name__ == "__main__":
    main()
