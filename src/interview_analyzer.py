
"""Interview video emotion and experimental behavior-proxy analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from config import MODELS_DIR
from src.confidence_analyzer import ConfidenceAnalyzer
from src.emotion_detector import EmotionDetector
from src.face_detector import FaceDetector
from src.model_utils import configure_logging

logger = configure_logging("interview_analyzer")

ProgressCallback = Callable[[int, int, int], None]


class InterviewAnalyzer:
    """Analyze sampled frames from an interview video."""

    def __init__(
        self,
        video_path: str | Path,
        model_path: str | Path | None = None,
    ) -> None:
        self.video_path = Path(video_path)

        selected_model = (
            Path(model_path)
            if model_path
            else MODELS_DIR / "mobilenetv2_emotion.keras"
        )

        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}"
            )

        if not selected_model.is_file():
            raise FileNotFoundError(
                f"Emotion model not found: {selected_model}"
            )

        self.detector = EmotionDetector(selected_model)
        self.face_detector = FaceDetector()
        self.confidence_analyzer = ConfidenceAnalyzer()

    @staticmethod
    def _select_primary_face(
        faces: Any,
        frame_shape: tuple[int, ...],
    ) -> Any | None:
        """Select the largest detected face."""

        if faces is None:
            return None

        candidates = list(faces)
        if not candidates:
            return None

        height, width = frame_shape[:2]

        def face_score(face: Any) -> float:
            area = max(face.width, 0) * max(face.height, 0)

            center_x = face.x + face.width / 2.0
            center_y = face.y + face.height / 2.0

            distance = (
                abs(center_x - width / 2.0) / max(width, 1)
                + abs(center_y - height / 2.0) / max(height, 1)
            )

            # Prefer large faces, with a small preference for the center.
            return area * (1.0 - min(distance, 0.9) * 0.25)

        return max(candidates, key=face_score)

    def analyze(
        self,
        max_frames: int | None = None,
        frame_interval: int = 10,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Analyze every Nth frame and return a JSON-compatible report."""

        if frame_interval < 1:
            raise ValueError("frame_interval must be at least 1.")

        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1 or None.")

        capture = cv2.VideoCapture(str(self.video_path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {self.video_path}")

        fps_value = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps_value) if fps_value and fps_value > 0 else 30.0

        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_total = (
            min(reported_frames, max_frames)
            if reported_frames > 0 and max_frames is not None
            else reported_frames
        )

        if max_frames is not None and reported_frames <= 0:
            effective_total = max_frames

        duration = (
            reported_frames / fps if reported_frames > 0 else 0.0
        )

        frame_features: list[dict[str, Any]] = []
        emotion_counts: Counter[str] = Counter()
        probabilities: list[float] = []
        timeline: list[dict[str, Any]] = []

        frame_number = 0
        sampled_frames = 0
        analyzed_frames = 0
        frames_without_face = 0

        previous_emotion: str | None = None

        # ConfidenceAnalyzer expects (x, y, width, height).
        previous_face: tuple[int, int, int, int] | None = None

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_number += 1

                if (
                    max_frames is not None
                    and frame_number > max_frames
                ):
                    frame_number -= 1
                    break

                # Sample frame 1, then 1 + interval, 1 + 2*interval, ...
                if (frame_number - 1) % frame_interval != 0:
                    if progress_callback and (
                        frame_number % 10 == 0
                        or frame_number == effective_total
                    ):
                        progress_callback(
                            frame_number,
                            effective_total,
                            analyzed_frames,
                        )
                    continue

                sampled_frames += 1

                faces = self.face_detector.detect(frame)
                face = self._select_primary_face(faces, frame.shape)

                if face is None:
                    frames_without_face += 1
                else:
                    crop = self.face_detector.crop(frame, face)

                    if crop is None or crop.size == 0:
                        frames_without_face += 1
                    else:
                        prediction = self.detector.predict(crop)

                        emotion = str(prediction["emotion"])
                        probability = float(prediction["probability"])

                        # IMPORTANT:
                        # FaceDetection.bbox is (x1, y1, x2, y2).
                        # ConfidenceAnalyzer expects (x, y, width, height).
                        current_face = (
                            int(face.x),
                            int(face.y),
                            int(face.width),
                            int(face.height),
                        )

                        features = (
                            self.confidence_analyzer.frame_features(
                                frame=frame,
                                face_bbox=current_face,
                                emotion=emotion,
                                emotion_probability=probability,
                                previous_emotion=previous_emotion,
                                previous_face=previous_face,
                            )
                        )

                        features["video_frame_number"] = frame_number
                        frame_features.append(features)

                        emotion_counts[emotion] += 1
                        probabilities.append(probability)

                        timeline.append({
                            "frame": frame_number,
                            "time_seconds": round(
                                (frame_number - 1) / fps, 3
                            ),
                            "emotion": emotion,
                            "probability": round(probability, 4),
                        })

                        analyzed_frames += 1
                        previous_emotion = emotion

                        if features["face_bbox"] is not None:
                            previous_face = features["face_bbox"]

                if progress_callback:
                    progress_callback(
                        frame_number,
                        effective_total,
                        analyzed_frames,
                    )

        finally:
            capture.release()

        if not frame_features:
            raise ValueError(
                "No faces were successfully analyzed in the sampled "
                "video frames. Try a video with a clearly visible face, "
                "good lighting, and a front-facing camera angle."
            )

        proxy_metrics = self.confidence_analyzer.analyze(frame_features)

        total_emotions = sum(emotion_counts.values())

        emotion_distribution = {
            emotion: round(count / total_emotions * 100.0, 2)
            for emotion, count in emotion_counts.items()
        }

        result = {
            "video": str(self.video_path),
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 2),
            "total_frames": int(frame_number),
            "frame_interval": int(frame_interval),
            "sampled_frames": int(sampled_frames),
            "analyzed_frames": int(analyzed_frames),
            "face_detected_frames": int(analyzed_frames),
            "frames_without_face": int(frames_without_face),
            "face_image_quality_proxy": round(
                proxy_metrics["face_image_quality_proxy"] * 100.0,
                2,
            ),
            "head_stability_proxy": round(
                proxy_metrics["head_stability_proxy"] * 100.0,
                2,
            ),
            "emotion_stability_proxy": round(
                proxy_metrics["emotion_stability_proxy"] * 100.0,
                2,
            ),
            "dominant_emotion": emotion_counts.most_common(1)[0][0],
            "emotion_distribution": emotion_distribution,
            "emotion_timeline": timeline,
            "average_emotion_probability": round(
                float(np.mean(probabilities)), 4
            ),
            "behavioral_proxy_score": proxy_metrics[
                "behavioral_proxy_score"
            ],
            "behavioral_proxy_details": proxy_metrics,
            "overall_interpretation": (
                "Experimental observable-video proxy only; not a "
                "clinical or psychological confidence measurement."
            ),
            "warning": (
                "Facial expressions and predicted emotion labels do not "
                "necessarily reveal internal emotional states. These "
                "results should not be used alone for hiring decisions."
            ),
        }

        logger.info(
            "Video analysis completed: %d sampled, %d analyzed, "
            "%d without a detected face.",
            sampled_frames,
            analyzed_frames,
            frames_without_face,
        )

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze emotion and observable behavior in a video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional path to a trained emotion model",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Analyze every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of video frames to read",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the JSON report",
    )

    args = parser.parse_args()

    analyzer = InterviewAnalyzer(
        video_path=args.video,
        model_path=args.model,
    )

    report = analyzer.analyze(
        max_frames=args.max_frames,
        frame_interval=args.frame_interval,
    )

    report_json = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json, encoding="utf-8")
        print(f"Report saved to: {output_path}")

    print(report_json)


if __name__ == "__main__":
    main()
