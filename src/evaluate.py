import argparse
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import get_datasets, DEFAULT_DATA_PATH

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default=DEFAULT_DATA_PATH)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    model = load_model(args.model)
    _, val_ds, class_names = get_datasets(data_dir=args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)

    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix shape:", cm.shape)

if __name__ == "__main__":
    main()
