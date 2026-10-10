import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from config import DATA_DIR, EMOTION_CLASSES, RANDOM_SEED

tf.keras.utils.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def main():
    images, labels = [], []

    for label, emotion in enumerate(EMOTION_CLASSES):
        folder = Path(DATA_DIR) / "train" / emotion
        paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(folder.glob(pattern))
        random.shuffle(paths)

        count = 0
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if image.shape != (48, 48):
                image = cv2.resize(image, (48, 48))
            images.append(image.astype(np.float32) / 255.0)
            labels.append(label)
            count += 1
            if count == 20:
                break

        if count != 20:
            raise ValueError(f"Could not load 20 images for {emotion}")

    x = np.asarray(images, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int32)

    print("Image shape:", x.shape)
    print("Pixel range:", float(x.min()), "to", float(x.max()))
    print("Class counts:", [int(np.sum(y == i)) for i in range(7)])

    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Conv2D(128, 3, padding="same", activation="relu"),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(EMOTION_CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Starting diagnostic: 30 epochs")
    model.fit(x, y, epochs=30, batch_size=16, shuffle=True, verbose=1)

    probabilities = model.predict(x, batch_size=16, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("Final training-sample accuracy:", float(np.mean(predictions == y)))
    print("Predicted counts:", [int(np.sum(predictions == i)) for i in range(7)])
    print("Diagnostic complete. No model file was saved.")


if __name__ == "__main__":
    main()
