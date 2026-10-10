
"""Train an SVM baseline using HOG and PCA."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skimage.feature import hog

from config import DATA_DIR, EMOTION_CLASSES, MODELS_DIR, RANDOM_SEED
from src.model_utils import configure_logging, save_json

logger = configure_logging("train_svm")


def load_data(
    data_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image paths and their numeric emotion labels."""
    paths = []
    labels = []

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
            f"No images found in {data_dir / split}."
        )

    return np.asarray(paths), np.asarray(labels, dtype=np.int64)


def extract_hog(image_path: Path) -> np.ndarray:
    """Extract HOG features from an image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )

    return np.asarray(features, dtype=np.float32).reshape(-1)


def train(data_dir: Path = DATA_DIR) -> dict:
    """Train the SVM pipeline and evaluate it on the test split."""
    train_paths, train_labels = load_data(data_dir, "train")
    test_paths, test_labels = load_data(data_dir, "test")

    logger.info("Training images: %d", len(train_paths))
    logger.info("Test images: %d", len(test_paths))

    logger.info("Extracting training HOG features...")
    train_features = np.stack(
        [extract_hog(path) for path in train_paths]
    )

    logger.info("Extracting test HOG features...")
    test_features = np.stack(
        [extract_hog(path) for path in test_paths]
    )

    # PCA cannot use more components than the available samples
    # or input features. This also supports the small smoke test.
    n_components = min(
        128,
        train_features.shape[0] - 1,
        train_features.shape[1],
    )

    if n_components < 1:
        raise ValueError(
            "Not enough training samples to perform PCA."
        )

    logger.info("Using %d PCA components", n_components)

    model = Pipeline([
        (
            "pca",
            PCA(
                n_components=n_components,
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "svm",
            SVC(
                C=2.0,
                gamma="scale",
                kernel="rbf",
                probability=False,
                random_state=RANDOM_SEED,
            ),
        ),
    ])

    logger.info("Training SVM...")
    started = time.perf_counter()
    model.fit(train_features, train_labels)
    training_time = time.perf_counter() - started

    logger.info("Evaluating SVM...")
    predictions = model.predict(test_features)

    metrics = {
        "model": "SVM",
        "feature_extractor": "HOG",
        "dimensionality_reduction": "PCA",
        "pca_components": int(n_components),
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
        "training_time_seconds": round(training_time, 3),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "svm_hog_pca.joblib"
    metrics_path = MODELS_DIR / "svm_metrics.json"

    dump(model, model_path)
    save_json(metrics_path, metrics)

    logger.info("Model saved to %s", model_path)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1: %.4f", metrics["f1_macro"])

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an SVM emotion classifier using HOG and PCA."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the FER2013 dataset directory.",
    )

    args = parser.parse_args()
    metrics = train(args.data_dir)

    print("\nSVM evaluation results")
    print("-" * 30)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro F1:       {metrics['f1_macro']:.4f}")
    print(f"Macro precision: {metrics['precision_macro']:.4f}")
    print(f"Macro recall:    {metrics['recall_macro']:.4f}")
    print(f"PCA components: {metrics['pca_components']}")
    print(f"Training time:  {metrics['training_time_seconds']:.3f} seconds")


if __name__ == "__main__":
    main()

