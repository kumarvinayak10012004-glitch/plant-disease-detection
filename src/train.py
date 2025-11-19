import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from src.data_loader import get_datasets, DEFAULT_DATA_PATH
from src.model import build_model
from src.augmentations import get_augmentation_layer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_PATH)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--out_dir", default="models")
    p.add_argument("--fine_tune_epochs", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, class_names = get_datasets(data_dir=args.data_dir,
                                                img_size=(args.img_size, args.img_size),
                                                batch_size=args.batch_size)

    print("Classes:", len(class_names))

    # Data augmentation layer inserted as first layer in the model pipeline
    augmentation = get_augmentation_layer(img_size=(args.img_size, args.img_size))

    # Prefetch and map augmentation
    def augment(images, labels):
        images = augmentation(images)
        return images, labels

    train_ds_aug = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    num_classes = len(class_names)

    model, base_model = build_model(num_classes, img_size=(args.img_size, args.img_size))

    # Callbacks
    ckpt1 = os.path.join(args.out_dir, "best_model.h5")
    callbacks = [
        ModelCheckpoint(ckpt1, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    ]

    print("Training top layers...")
    history = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save class names
    with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Fine-tuning: unfreeze last blocks
    print("Fine-tuning...")
    base_model.trainable = True
    # Freeze first layers, unfreeze last ~20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt2 = os.path.join(args.out_dir, "best_model_finetuned.h5")
    ft_callbacks = [
        ModelCheckpoint(ckpt2, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=ft_callbacks
    )

    final_path = os.path.join(args.out_dir, "final_model.h5")
    model.save(final_path)
    print("Training complete. Models saved to:", args.out_dir)
    print("Final model saved as:", final_path)

if __name__ == "__main__":
    main()
