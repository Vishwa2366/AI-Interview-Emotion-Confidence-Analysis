
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import DATA_DIR, MODELS_DIR, EMOTION_CLASSES, RANDOM_SEED


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15


def make_dataset(data_dir, split, batch_size=BATCH_SIZE):
    """Load images in batches instead of keeping the whole dataset in RAM."""
    split_dir = Path(data_dir) / split

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=list(EMOTION_CLASSES),
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_training_datasets(data_dir, batch_size=BATCH_SIZE):
    """Create separate training and validation datasets."""
    train_dir = Path(data_dir) / "train"

    common_args = {
        "labels": "inferred",
        "label_mode": "int",
        "class_names": list(EMOTION_CLASSES),
        "color_mode": "rgb",
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "validation_split": VALIDATION_SPLIT,
        "seed": RANDOM_SEED,
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset="training",
        shuffle=True,
        **common_args,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset="validation",
        shuffle=False,
        **common_args,
    )

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    train_ds = train_ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model(num_classes=len(EMOTION_CLASSES)):
    """Build a transfer-learning model using MobileNetV2."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # First train the new classification head.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(
    data_dir=DATA_DIR,
    epochs=15,
    batch_size=BATCH_SIZE,
):
    """Train, evaluate and save the MobileNetV2 emotion classifier."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    data_dir = Path(data_dir)
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = models_dir / "mobilenetv2_emotion.keras"
    best_model_path = models_dir / "mobilenetv2_emotion_best.keras"
    metrics_path = models_dir / "mobilenet_metrics.json"

    print("Loading training and validation datasets...")
    train_ds, val_ds = make_training_datasets(
        data_dir,
        batch_size=batch_size,
    )

    print("Loading test dataset...")
    test_ds = make_dataset(
        data_dir,
        split="test",
        batch_size=batch_size,
    )

    print("Building MobileNetV2 model...")
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Starting training for up to {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    training_time = time.time() - start_time

    # Load the best validation-loss checkpoint before final evaluation.
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    print("Evaluating on the separate test dataset...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    y_true = np.concatenate(
        [labels.numpy() for _, labels in test_ds],
        axis=0,
    )
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "model": "MobileNetV2",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_loss": float(test_loss),
        "test_accuracy_from_evaluate": float(test_accuracy),
        "training_time_seconds": round(training_time, 2),
        "epochs_requested": int(epochs),
        "epochs_completed": len(history.history["loss"]),
        "batch_size": int(batch_size),
        "image_size": list(IMAGE_SIZE),
        "validation_split": VALIDATION_SPLIT,
        "class_names": list(EMOTION_CLASSES),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(EMOTION_CLASSES))),
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(EMOTION_CLASSES))),
            target_names=list(EMOTION_CLASSES),
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save the best model under the filename expected by the app.
    model.save(final_model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(models_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(list(EMOTION_CLASSES), file, indent=4)

    metadata = {
        "model": "MobileNetV2",
        "input_shape": [224, 224, 3],
        "preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input",
        "class_names": list(EMOTION_CLASSES),
        "weights": "imagenet",
    }

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed.")
    print(f"Best model saved to: {final_model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['f1_macro']:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate the MobileNetV2 emotion classifier."
    )
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

