"""Emotion inference for detected faces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from config import MODELS_DIR, MOBILENET_INPUT_SIZE
from src.model_utils import configure_logging, load_class_names, normalize_prediction

logger = configure_logging("emotion_detector")


class EmotionDetector:
    """Loads and uses the saved MobileNetV2 emotion model."""

    def __init__(self, model_path: str | Path | None = None, class_names_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or MODELS_DIR / "mobilenetv2_emotion.keras")
        self.class_names_path = Path(class_names_path or MODELS_DIR / "class_names.json")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing trained emotion model: {self.model_path}. Train MobileNetV2 first."
            )
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Missing class names file: {self.class_names_path}.")
        self.class_names = load_class_names(self.class_names_path)
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        logger.info("Emotion model loaded from %s", self.model_path)

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized / 255.0, axis=0)

    def predict(self, face_crop: np.ndarray) -> dict[str, Any]:
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Face crop is empty.")
        prediction = self.model.predict(self.preprocess(face_crop), verbose=0)
        probabilities = np.asarray(prediction[0])
        return normalize_prediction(probabilities, self.class_names)

    def predict_from_path(self, image_path: str | Path) -> dict[str, Any]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")
        return self.predict(image)
