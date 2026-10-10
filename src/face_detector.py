
"""Face detection using OpenCV's Haar Cascade classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.model_utils import configure_logging

logger = configure_logging("face_detector")


@dataclass
class FaceDetection:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as (x1, y1, x2, y2)."""
        return (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )


class FaceDetector:
    """Detect faces in video frames using an OpenCV Haar Cascade."""

    def __init__(self, cascade_path: str | None = None) -> None:
        path = (
            cascade_path
            if cascade_path
            else str(
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
        )

        self.cascade = cv2.CascadeClassifier(path)

        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade classifier from: {path}"
            )

        logger.info("Face detector initialized using cascade: %s", path)

    def detect(
        self,
        frame: np.ndarray,
        min_size: tuple[int, int] | None = None,
    ) -> list[FaceDetection]:
        """Return detected face boxes for a frame."""

        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim not in (2, 3)
        ):
            return []

        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            logger.warning("Unsupported frame format: %s", frame.shape)
            return []

        # Slightly reduce the strictness to help detect smaller faces.
        # Detection may produce more false positives, so verify results.
        if min_size is None:
            min_dimension = min(gray.shape[:2])
            minimum = max(30, min(80, min_dimension // 8))
            effective_min_size = (minimum, minimum)
        else:
            effective_min_size = min_size

        # Equalization can help in some lighting conditions.
        enhanced = cv2.equalizeHist(gray)

        detections = self.cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=effective_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        frame_height, frame_width = gray.shape[:2]
        results: list[FaceDetection] = []

        for x, y, width, height in detections:
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)

            if width <= 0 or height <= 0:
                continue

            # Reject boxes outside the frame.
            if (
                x < 0
                or y < 0
                or x + width > frame_width
                or y + height > frame_height
            ):
                continue

            results.append(
                FaceDetection(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    # Haar Cascade does not provide a calibrated
                    # per-face confidence score here.
                    confidence=0.0,
                )
            )

        return results

    def select_primary(
        self,
        faces: list[FaceDetection],
        frame_shape: tuple[int, ...],
    ) -> FaceDetection | None:
        """Select a centrally located face, preferring larger faces."""

        if not faces:
            return None

        height, width = frame_shape[:2]
        frame_area = max(height * width, 1)

        def score(face: FaceDetection) -> float:
            center_x = face.x + face.width / 2.0
            center_y = face.y + face.height / 2.0

            distance = (
                abs(center_x - width / 2.0) / max(width, 1)
                + abs(center_y - height / 2.0) / max(height, 1)
            )

            relative_area = (face.width * face.height) / frame_area

            # Lower score is better. Center proximity is the main factor.
            return distance - 0.25 * relative_area

        return min(faces, key=score)

    def crop(
        self,
        frame: np.ndarray,
        face: FaceDetection,
    ) -> np.ndarray:
        """Crop the detected face with modest padding."""

        if frame is None or frame.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        frame_height, frame_width = frame.shape[:2]

        # Use face dimensions, not frame dimensions, to set padding.
        padding = max(
            4,
            int(round(max(face.width, face.height) * 0.08)),
        )

        x0 = max(0, face.x - padding)
        y0 = max(0, face.y - padding)
        x1 = min(frame_width, face.x + face.width + padding)
        y1 = min(frame_height, face.y + face.height + padding)

        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        return frame[y0:y1, x0:x1].copy()
