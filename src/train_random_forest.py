
"""Train a Random Forest baseline using HOG features."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from skimage.feature import hog

from config import DATA_DIR, EMOTION_CLASSES, MODELS_DIR, RANDOM_SEED
from src.model_utils import configure_logging, save_json

logger = configure_logging("train_random_forest")


def load_data(
    data_dir: Path,
    split: str,
) -> tuple[list[Path], np.ndarray]:
    """Load image paths and their numeric emotion labels."""
    paths: list[Path] = []
    labels: list[int] = []

    for label, class_name in enumerate(EMOTION_CLASSES):
        directory = data_dir / split / class_name

        if not directory.exists():
            raise FileNotFoundError(
                f"Missing directory: {directory}"
            )

        for path in sorted(directory.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ):
                paths.append(path)
                labels.append(label)

    if not paths:
        raise ValueError(
            f"No supported images found in {data_dir / split}"
        )

    return paths, np.asarray(labels, dtype=np.int64)


def extract_hog(image_path: Path) -> np.ndarray:
    """Extract HOG (Histogram of Oriented Gradients) features."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    # The feature_type argument is not supported by skimage.feature.hog.
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )

    return np.asarray(features, dtype=np.float32).reshape(-1)


def extract_features(
    paths: list[Path],
    split_name: str,
) -> np.ndarray:
    """Extract HOG features for all images in a dataset split."""
    features = []

    total = len(paths)

    for index, path in enumerate(paths, start=1):
        features.append(extract_hog(path))

        if index % 500 == 0 or index == total:
            logger.info(
                "%s features: %d/%d images processed",
                split_name,
                index,
                total,
            )

    return np.stack(features).astype(np.float32)


def train(
    data_dir: Path = DATA_DIR,
    n_estimators: int = 200,
    max_depth: int = 20,
) -> dict:
    """Train the Random Forest and evaluate it on the test split."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    data_dir = Path(data_dir)

    logger.info("Loading training images from %s", data_dir)

    train_paths, train_labels = load_data(data_dir, "train")
    test_paths, test_labels = load_data(data_dir, "test")

    logger.info("Training images: %d", len(train_paths))
    logger.info("Test images: %d", len(test_paths))

    logger.info("Extracting HOG features for training images...")
    train_features = extract_features(train_paths, "Training")

    logger.info("Extracting HOG features for test images...")
    test_features = extract_features(test_paths, "Test")

    logger.info("Training Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    started = time.perf_counter()
    model.fit(train_features, train_labels)
    training_time = time.perf_counter() - started

    logger.info("Evaluating model on the test split...")

    predictions = model.predict(test_features)

    metrics = {
        "model": "RandomForestClassifier",
        "feature_extractor": "HOG",
        "accuracy": float(
            accuracy_score(test_labels, predictions)
        ),
        "precision_macro": float(
            precision_score(
                test_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                test_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                test_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=list(range(len(EMOTION_CLASSES))),
        ).tolist(),
        "class_names": list(EMOTION_CLASSES),
        "training_samples": int(len(train_labels)),
        "test_samples": int(len(test_labels)),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "training_time_seconds": round(training_time, 3),
    }

    # Ensure the output directory exists before saving.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "random_forest_hog.joblib"
    metrics_path = MODELS_DIR / "random_forest_metrics.json"

    dump(model, model_path)
    save_json(metrics_path, metrics)

    logger.info("Model saved to %s", model_path)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Test accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1 score: %.4f", metrics["f1_macro"])

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest emotion classifier using HOG features."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the FER2013 dataset directory.",
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest.",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=20,
        help="Maximum depth of each tree.",
    )

    args = parser.parse_args()

    metrics = train(
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    print("\nRandom Forest evaluation results")
    print("-" * 35)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['f1_macro']:.4f}")
    print(f"Macro precision: {metrics['precision_macro']:.4f}")
    print(f"Macro recall:    {metrics['recall_macro']:.4f}")
    print(f"Training time: {metrics['training_time_seconds']:.3f} seconds")


if __name__ == "__main__":
    main()

