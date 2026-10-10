
"""Experimental, observable video-behavior proxy metrics.

These metrics are not validated measures of psychological confidence,
eye contact, competence, or internal emotional state.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class ConfidenceAnalyzer:
    """Calculate experimental image and frame-consistency proxies."""

    DEFAULT_WEIGHTS = {
        "face_image_quality_proxy": 0.25,
        "head_stability_proxy": 0.35,
        "emotion_stability_proxy": 0.40,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        diagnostics: bool = True,
    ) -> None:
        supplied = dict(
            self.DEFAULT_WEIGHTS if weights is None else weights
        )

        if set(supplied) != set(self.DEFAULT_WEIGHTS):
            raise ValueError(
                "Weights must contain exactly these keys: "
                + ", ".join(self.DEFAULT_WEIGHTS)
            )

        if any(
            not np.isfinite(value) or value < 0
            for value in supplied.values()
        ):
            raise ValueError(
                "Weights must be finite, non-negative numbers."
            )

        total = sum(supplied.values())
        if total <= 0:
            raise ValueError("At least one weight must be positive.")

        self.weights = {
            key: value / total
            for key, value in supplied.items()
        }
        self.diagnostics = diagnostics

    @staticmethod
    def _bounded(value: float) -> float:
        """Clamp a finite value to the range 0..1."""
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert a supported image crop to grayscale."""
        if image.ndim == 2:
            return image

        if image.ndim != 3:
            raise ValueError("Unsupported image dimensions.")

        channels = image.shape[2]

        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        raise ValueError("Unsupported number of image channels.")

    @staticmethod
    def _normalize_bbox(
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        """Clip a bounding box to image boundaries."""
        x, y, width, height = map(int, bbox)

        if width <= 0 or height <= 0:
            return None

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2 - x1, y2 - y1)

    def frame_features(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        emotion: str,
        emotion_probability: float,
        previous_emotion: str | None = None,
        previous_face: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        """Extract frame-level image and consistency proxies."""

        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim not in (2, 3)
        ):
            raise ValueError("frame must be a non-empty image array.")

        image_height, image_width = frame.shape[:2]

        current_bbox = self._normalize_bbox(
            face_bbox, image_width, image_height
        )

        quality_proxy = 0.0
        sharpness = 0.0
        crop_width = 0
        crop_height = 0

        if current_bbox is not None:
            x, y, width, height = current_bbox
            face_crop = frame[y:y + height, x:x + width]

            if face_crop.size > 0:
                gray = self._to_gray(face_crop)
                crop_height, crop_width = gray.shape[:2]

                if crop_width >= 3 and crop_height >= 3:
                    # Laplacian variance is a rough sharpness heuristic.
                    sharpness = float(
                        cv2.Laplacian(gray, cv2.CV_64F).var()
                    )

                    # Smooth, bounded transformation instead of dividing
                    # by 100 and calling the result a quality percentage.
                    # The scale of 20 is a heuristic, not a calibration.
                    sharpness = max(sharpness, 0.0)
                    quality_proxy = self._bounded(
                        sharpness / (sharpness + 20.0)
                    )

        if self.diagnostics:
            print(
                "[Face quality diagnostic] "
                f"Crop: {crop_width}x{crop_height}, "
                f"Sharpness variance: {sharpness:.2f}, "
                f"Sharpness proxy: {quality_proxy * 100:.2f}/100"
            )

        # Compare normalized face-box movement between sampled frames.
        if current_bbox is None:
            head_stability = 0.0
        elif previous_face is None:
            # First observation is a baseline, not evidence of stability.
            head_stability = 1.0
        else:
            previous_bbox = self._normalize_bbox(
                previous_face, image_width, image_height
            )

            if previous_bbox is None:
                head_stability = 0.0
            else:
                px, py, pw, ph = previous_bbox
                cx, cy, cw, ch = current_bbox

                frame_diagonal = max(
                    float(np.hypot(image_width, image_height)), 1.0
                )

                old_center = (px + pw / 2.0, py + ph / 2.0)
                new_center = (cx + cw / 2.0, cy + ch / 2.0)

                center_change = float(
                    np.hypot(
                        new_center[0] - old_center[0],
                        new_center[1] - old_center[1],
                    )
                ) / frame_diagonal

                old_area = max(pw * ph, 1)
                new_area = max(cw * ch, 1)

                size_change = abs(
                    float(np.log(new_area / old_area))
                )

                movement = center_change + 0.10 * size_change
                head_stability = self._bounded(1.0 - 4.0 * movement)

        # This measures repeated predicted labels, not real emotional
        # stability. The first frame has no previous label to compare.
        emotion_stability = float(
            previous_emotion is not None
            and previous_emotion == emotion
        )

        try:
            probability = self._bounded(float(emotion_probability))
        except (TypeError, ValueError):
            probability = 0.0

        return {
            "face_image_quality_proxy": quality_proxy,
            "head_stability_proxy": head_stability,
            "emotion_stability_proxy": emotion_stability,
            "emotion": str(emotion),
            "emotion_probability": probability,
            "face_bbox": current_bbox,
            "face_crop_width": crop_width,
            "face_crop_height": crop_height,
            "sharpness_variance": sharpness,
        }

    def analyze(
        self,
        frame_features: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate frame-level proxies into an experimental summary."""

        warning = (
            "Experimental image and behavior proxies only. These are not "
            "validated measures of psychological confidence, eye contact, "
            "competence, or internal emotional state. Scores depend on "
            "camera conditions, sampling, and heuristic choices."
        )

        if not frame_features:
            return {
                "face_image_quality_proxy": 0.0,
                "head_stability_proxy": 0.0,
                "emotion_stability_proxy": 0.0,
                "frames_analyzed": 0,
                "behavioral_proxy_score": 0.0,
                "score_status": "insufficient_data",
                "warning": warning,
            }

        def mean_metric(name: str) -> float:
            values = []
            for feature in frame_features:
                try:
                    value = float(feature.get(name, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                values.append(self._bounded(value))
            return float(np.mean(values))

        metrics = {
            "face_image_quality_proxy": round(
                mean_metric("face_image_quality_proxy"), 4
            ),
            "head_stability_proxy": round(
                mean_metric("head_stability_proxy"), 4
            ),
            "emotion_stability_proxy": round(
                mean_metric("emotion_stability_proxy"), 4
            ),
            "frames_analyzed": len(frame_features),
        }

        score = sum(
            self.weights[name] * metrics[name]
            for name in self.weights
        )

        metrics["behavioral_proxy_score"] = round(
            self._bounded(score) * 100.0, 2
        )
        metrics["score_status"] = "experimental_proxy"
        metrics["warning"] = warning

        return metrics
