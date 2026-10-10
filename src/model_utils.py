"""Shared model, metrics, and logging utilities."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from config import BASE_DIR, MODELS_DIR, LOGS_DIR, EMOTION_CLASSES


def configure_logging(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_class_names(path: Path = MODELS_DIR / "class_names.json") -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing class names file: {path}")
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != len(EMOTION_CLASSES):
        raise ValueError("class_names.json must contain exactly seven emotion classes.")
    return names


def normalize_prediction(probabilities: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Normalize model probabilities into a stable, documented output format."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size != len(class_names):
        raise ValueError("Model output count does not match the class-name count.")
    total = float(np.sum(values))
    values = values / total if total > 0 else np.full(len(class_names), 1.0 / len(class_names))
    index = int(np.argmax(values))
    return {
        "emotion": class_names[index],
        "probability": float(values[index]),
        "all_class_probabilities": {name: float(value) for name, value in zip(class_names, values)},
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> dict[str, Any]:
    """Compute standard classification metrics and confusion matrix."""
    labels = labels or EMOTION_CLASSES
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "class_names": labels,
    }


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics.get("confusion_matrix", [])).to_csv(path.with_name(path.stem + "_confusion_matrix.csv"), index=False, header=False)
    save_json(path.with_name(path.stem + "_metrics.json"), metrics)
