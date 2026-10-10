
"""Train and evaluate a CNN for FER2013 emotion classification."""

import json
import time
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

from config import (
    DATA_DIR, MODELS_DIR, GRAPHS_DIR, CONFUSION_DIR,
    EMOTION_CLASSES, IMAGE_SIZE, RANDOM_SEED,
)

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BATCH_SIZE = 64
EPOCHS = 30


def load_images(split):
    """Load grayscale 48x48 images from class subfolders."""
    images, labels = [], []
    split_dir = DATA_DIR / split

    for label, emotion in enumerate(EMOTION_CLASSES):
        class_dir = split_dir / emotion
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                image = tf.keras.utils.load_img(
                    path,
                    color_mode="grayscale",
                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                )
                array = tf.keras.utils.img_to_array(image)
                images.append(array)
                labels.append(label)
            except Exception as exc:
                print(f"Skipping unreadable image {path}: {exc}")

    if not images:
        raise RuntimeError(f"No images found in {split_dir}")

    X = np.asarray(images, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int64)
    print(f"{split}: {len(X)} images loaded; shape={X.shape}")
    return X, y


def build_model():
    """Build a compact CNN for seven emotion classes."""
    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Conv2D(128, 3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(EMOTION_CLASSES), activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FER2013 training data...")
    X_all, y_all = load_images("train")
    X_test, y_test = load_images("test")

    # Keep a validation split separate from the final test set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
    del X_all, y_all

    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_values)
    }

    model = build_model()
    model.summary()

    best_path = MODELS_DIR / "cnn_emotion_best.keras"
    final_path = MODELS_DIR / "cnn_emotion.keras"
    metrics_path = MODELS_DIR / "cnn_metrics.json"

    callbacks = [
        ModelCheckpoint(
            str(best_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting CNN training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the best validation checkpoint on the held-out test set.
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)

    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    test_accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(EMOTION_CLASSES))),
        target_names=EMOTION_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test, predictions, labels=list(range(len(EMOTION_CLASSES)))
    )

    # Save final model and evaluation metrics.
    model.save(final_path)

    metrics = {
        "model": "CNN",
        "dataset": "FER2013",
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
        "training_seconds": round(time.time() - start, 2),
        "epochs_completed": len(history.history["loss"]),
        "class_names": EMOTION_CLASSES,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "prediction_counts": {
            emotion: int(np.sum(predictions == index))
            for index, emotion in enumerate(EMOTION_CLASSES)
        },
        "note": (
            "FER2013 measures emotion categories, not interview confidence. "
            "Metrics describe this held-out dataset and are not a guarantee "
            "of real-world performance."
        ),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save confusion matrix as a PNG.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(
            xticks=range(len(EMOTION_CLASSES)),
            yticks=range(len(EMOTION_CLASSES)),
            xticklabels=EMOTION_CLASSES,
            yticklabels=EMOTION_CLASSES,
            xlabel="Predicted emotion",
            ylabel="True emotion",
            title="CNN — FER2013 Test Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, str(matrix[row, col]),
                        ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(CONFUSION_DIR / "cnn_confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not save confusion-matrix plot: {exc}")

    print("\n========== CNN TEST RESULTS ==========")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_test, predictions,
        labels=list(range(len(EMOTION_CLASSES))),
        target_names=EMOTION_CLASSES,
        zero_division=0,
    ))
    print("\nPrediction counts:")
    print(metrics["prediction_counts"])
    print(f"\nSaved model: {final_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Elapsed time: {metrics['training_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()