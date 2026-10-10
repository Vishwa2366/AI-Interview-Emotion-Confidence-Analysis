"""Reproduce model comparison using actual training and test results."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATA_DIR, EMOTION_CLASSES, OUTPUTS_DIR
from src.train_cnn import train as train_cnn
from src.train_random_forest import train as train_random_forest
from src.train_svm import train as train_svm
from src.train_mobilenet import train as train_mobilenet
from src.model_utils import save_json


def save_comparison(metrics_by_model: dict[str, dict], output_dir: Path) -> Path:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision_macro"],
            "recall": metrics["recall_macro"],
            "f1": metrics["f1_macro"],
            "training_time": metrics["training_time_seconds"],
        }
        rows.append(row)
    dataframe = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("graphs").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("confusion_matrices").mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_dir / "model_comparison.csv", index=False)
    chart = sns.barplot(data=dataframe, x="model", y="f1", ax=None)
    fig = chart.figure
    fig.savefig(output_dir / "graphs" / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for model_name, metrics in metrics_by_model.items():
        matrix = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(f"Confusion matrix: {model_name}")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrices" / f"{model_name.replace(' ', '_').replace('+', '_').lower()}.png", dpi=150)
        plt.close(fig)
    return output_dir / "model_comparison.csv"


def create_subset(data_dir: Path, subset_per_class: int) -> Path:
    """Create a temporary, class-balanced development dataset using real images."""
    temp_dir = Path(tempfile.mkdtemp(prefix="fer2013_subset_"))
    for split in ("train", "test"):
        for class_name in EMOTION_CLASSES:
            source = data_dir / split / class_name
            target = temp_dir / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            for image in images[:subset_per_class]:
                shutil.copy2(image, target / image.name)
    return temp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--subset_per_class", type=int, default=None)
    args = parser.parse_args()
    if args.subset_per_class is not None and args.subset_per_class < 1:
        raise SystemExit("--subset_per_class must be positive.")
    if args.subset_per_class is not None:
        print(f"Development mode: limiting each class to {args.subset_per_class} images.")
        data_dir = create_subset(args.data_dir, args.subset_per_class)
    else:
        data_dir = args.data_dir
    metrics = {
        "CNN from scratch": train_cnn(data_dir, epochs=args.epochs),
        "SVM + HOG + PCA": train_svm(data_dir),
        "Random Forest + HOG": train_random_forest(data_dir),
        "MobileNetV2 transfer learning": train_mobilenet(data_dir, epochs=args.epochs),
    }
    comparison = save_comparison(metrics, OUTPUTS_DIR)
    print(json.dumps(metrics, indent=2))
    print(f"Comparison saved to {comparison}")


if __name__ == "__main__":
    main()
