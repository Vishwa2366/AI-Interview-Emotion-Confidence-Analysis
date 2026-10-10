"""Train a MobileNetV2 transfer learning model for FER2013."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from config import DATA_DIR, EMOTION_CLASSES, MODELS_DIR, RANDOM_SEED
from src.model_utils import configure_logging, save_json

logger = configure_logging("train_mobilenet")


def load_images(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths, labels = [], []
    for label, class_name in enumerate(EMOTION_CLASSES):
        directory = data_dir / split / class_name
        if not directory.exists():
            raise FileNotFoundError(f"Missing directory: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(path); labels.append(label)
    if not paths:
        raise ValueError(f"No images found in {data_dir / split}.")
    images = np.stack([
        cv2.resize(cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (224, 224))
        for path in paths
    ])
    return images.astype(np.float32) / 255.0, np.asarray(labels), paths


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(7, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(data_dir: Path = DATA_DIR, epochs: int = 15, batch_size: int = 32) -> dict:
    train_images, train_labels, _ = load_images(data_dir, "train")
    test_images, test_labels, _ = load_images(data_dir, "test")
    tf.random.set_seed(RANDOM_SEED)
    model = build_model()
    checkpoint = MODELS_DIR / "mobilenetv2_emotion.keras"
    callbacks = [
        ModelCheckpoint(checkpoint, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
    ]
    started = time.perf_counter()
    history = model.fit(train_images, train_labels, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    training_time = time.perf_counter() - started
    predictions = np.argmax(model.predict(test_images, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision_macro": float(precision_score(test_labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=range(7)).tolist(),
        "class_names": EMOTION_CLASSES,
        "training_time_seconds": round(training_time, 3),
        "epochs": int(len(history.history["loss"])),
    }
    save_json(MODELS_DIR / "class_names.json", EMOTION_CLASSES)
    metadata = {
        "model": "MobileNetV2 transfer learning",
        "input_shape": [224, 224, 3],
        "classes": EMOTION_CLASSES,
        "base_model": "MobileNetV2",
        "base_model_trainable": False,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": metrics,
    }
    save_json(MODELS_DIR / "model_metadata.json", metadata)
    save_json(MODELS_DIR / "mobilenet_metrics.json", metrics)
    model.save(checkpoint)
    logger.info("MobileNetV2 training finished with accuracy %.4f", metrics["accuracy"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(train(args.data_dir, args.epochs, args.batch_size))


if __name__ == "__main__":
    main()
