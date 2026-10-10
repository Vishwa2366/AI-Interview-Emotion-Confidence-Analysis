
"""Diagnose FER2013 data and test whether a small CNN can learn."""

import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import DATA_DIR, EMOTION_CLASSES, RANDOM_SEED


# Make the experiment reproducible.
tf.keras.utils.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def load_balanced_samples(samples_per_class=20):
    """Load an equal number of images for each emotion."""
    images = []
    labels = []

    for label, emotion in enumerate(EMOTION_CLASSES):
        folder = Path(DATA_DIR) / "train" / emotion

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(folder.glob(pattern))

        paths = sorted(paths)
        random.shuffle(paths)

        loaded = 0

        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: unreadable image: {path}")
                continue

            if image.shape != (48, 48):
                image = cv2.resize(image, (48, 48))

            images.append(image.astype(np.float32) / 255.0)
            labels.append(label)
            loaded += 1

            if loaded >= samples_per_class:
                break

        if loaded < samples_per_class:
            raise ValueError(
                f"Only {loaded} readable images found for {emotion}"
            )

    x = np.asarray(images, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int32)

    return x, y


def verify_data(x, y):
    """Verify image shape, pixel range, and labels."""
    print("\n--- DATA VERIFICATION ---")
    print("Class order:", list(EMOTION_CLASSES))
    print("Image shape:", x.shape)
    print("Label shape:", y.shape)
    print("Pixel range:", float(x.min()), "to", float(x.max()))
    print("Label range:", int(y.min()), "to", int(y.max()))

    print("\nImages per class:")
    for label, emotion in enumerate(EMOTION_CLASSES):
        print(f"{label} - {emotion}: {int(np.sum(y == label))}")

    assert x.shape == (len(EMOTION_CLASSES) * 20, 48, 48, 1)
    assert len(x) == len(y)
    assert np.isfinite(x).all()
    assert 0.0 <= float(x.min()) <= float(x.max()) <= 1.0
    assert set(np.unique(y)) == set(range(len(EMOTION_CLASSES)))

    print("\nData verification passed.")


def build_diagnostic_model():
    """Build a simple CNN for a controlled learning test."""
    model = Sequential([
        Input(shape=(48, 48, 1)),

        Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),

        Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),

        Conv2D(128, kernel_size=3, padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),

        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(EMOTION_CLASSES), activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def test_learning(x, y):
    """Train a temporary model without saving any model files."""
    print("\n--- SMALL CNN LEARNING TEST ---")
    print("Training images:", len(x))
    print("Epochs: 30")
    print("Batch size: 16")

    model = build_diagnostic_model()

    initial_probabilities = model.predict(x, batch_size=16, verbose=0)
    initial_predictions = np.argmax(initial_probabilities, axis=1)
    initial_accuracy = float(np.mean(initial_predictions == y))

    print(f"Initial training-sample accuracy: {initial_accuracy:.4f}")

    history = model.fit(
        x,
        y,
        epochs=30,
        batch_size=16,
        shuffle=True,
        verbose=1,
    )

    probabilities = model.predict(x, batch_size=16, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    final_accuracy = float(np.mean(predictions == y))

    print("\n--- RESULTS ---")
    print(f"Initial accuracy: {initial_accuracy:.4f}")
    print(f"Final training-sample accuracy: {final_accuracy:.4f}")

    print("\nAccuracy and loss by epoch:")
    for epoch, (accuracy, loss) in enumerate(
        zip(history.history["accuracy"], history.history["loss"]),
        start=1,
    ):
        print(
            f"Epoch {epoch:02d}: "
            f"accuracy={accuracy:.4f}, loss={loss:.4f}"
        )

    print("\nPredicted counts:")
    for label, emotion in enumerate(EMOTION_CLASSES):
        count = int(np.sum(predictions == label))
        print(f"{emotion}: {count}")

    print(
        "\nMean predicted probabilities:",
        np.round(probabilities.mean(axis=0), 3),
    )

    print(
        "\nDiagnostic complete. This temporary model was not saved, "
        "and existing model files were not overwritten."
    )


def main():
    print("Loading a small, balanced FER2013 sample...")

    x, y = load_balanced_samples(samples_per_class=20)
    verify_data(x, y)
    test_learning(x, y)


if __name__ == "__main__":
    main()