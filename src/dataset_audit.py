"""Audit FER2013 dataset structure and image quality."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, EMOTION_CLASSES, GRAPHS_DIR, LOGS_DIR
from src.model_utils import configure_logging

logger = configure_logging("dataset_audit")
SPLITS = ("train", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset(data_dir: Path) -> tuple[dict, list[Path]]:
    """Validate train/test class folders and return split counts and image paths."""
    missing = [
        f"{split}/{class_name}"
        for split in SPLITS
        for class_name in EMOTION_CLASSES
        if not (data_dir / split / class_name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"Dataset is incomplete. Missing folders: {', '.join(missing)}. "
            f"Expected all emotion folders under train and test in {data_dir}."
        )

    split_counts: dict[str, dict[str, int]] = {}
    paths: list[Path] = []
    empty_classes: list[str] = []
    for split in SPLITS:
        counts: dict[str, int] = {}
        for class_name in EMOTION_CLASSES:
            class_paths = sorted(
                path for path in (data_dir / split / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            counts[class_name] = len(class_paths)
            paths.extend(class_paths)
            if not class_paths:
                empty_classes.append(f"{split}/{class_name}")
        split_counts[split] = counts

    if empty_classes:
        raise ValueError(
            f"Empty class folders (no supported image files): {', '.join(empty_classes)}."
        )

    return {
        "counts": split_counts["train"],
        "split_counts": split_counts,
        "missing_classes": [],
        "classes_found": {split: list(EMOTION_CLASSES) for split in SPLITS},
        "paths": paths,
    }


def corrupt_images(paths: Iterable[Path], max_samples: int = 25) -> list[Path]:
    """Identify images that OpenCV cannot decode as a practical quality audit."""
    unreadable: list[Path] = []
    for path in list(paths)[:max_samples]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable.append(path)
    return unreadable


def audit_dataset(data_dir: Path = DATA_DIR, graph_dir: Path = GRAPHS_DIR) -> dict:
    """Run the complete dataset audit and save a distribution graph."""
    data_dir = Path(data_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    data = validate_dataset(data_dir)
    split_counts = data["split_counts"]
    counts = pd.DataFrame([
        {"class": class_name, "split": split, "images": split_counts[split][class_name]}
        for split in SPLITS
        for class_name in EMOTION_CLASSES
    ])
    total = int(counts["images"].sum())
    unreadable = corrupt_images(data["paths"] if "paths" in data else [])
    graph_path = graph_dir / "class_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    sns = __import__("seaborn")
    sns.barplot(data=counts, x="class", y="images", hue="split", ax=ax)
    ax.set_title("FER2013 Training and Test Class Distribution")
    ax.set_xlabel("Emotion class")
    ax.set_ylabel("Image count")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    result = {
        "data_dir": str(data_dir),
        "total_images": total,
        "class_counts": data["counts"],
        "train_class_counts": split_counts["train"],
        "test_class_counts": split_counts["test"],
        "split_class_counts": split_counts,
        "missing_classes": data["missing_classes"],
        "unreadable_images": [str(path) for path in unreadable],
        "class_distribution_graph": str(graph_path),
    }
    logger.info("Dataset audit complete: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FER2013 dataset structure and quality.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--graph_dir", type=Path, default=GRAPHS_DIR)
    args = parser.parse_args()
    try:
        result = audit_dataset(args.data_dir, args.graph_dir)
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError, OSError) as error:
        logger.error("Dataset audit failed: %s", error)
        raise SystemExit(f"Dataset audit failed: {error}")


if __name__ == "__main__":
    main()
