"""Shared configuration for the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "fer2013"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
CONFUSION_DIR = OUTPUTS_DIR / "confusion_matrices"
REPORTS_DIR = OUTPUTS_DIR / "interview_reports"
PROCESSED_VIDEOS_DIR = OUTPUTS_DIR / "processed_videos"
LOGS_DIR = BASE_DIR / "logs"
INTERVIEW_VIDEOS_DIR = BASE_DIR / "interview_videos"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = 48
MOBILENET_INPUT_SIZE = 224
RANDOM_SEED = 42

for directory in [
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, GRAPHS_DIR, CONFUSION_DIR,
    REPORTS_DIR, PROCESSED_VIDEOS_DIR, LOGS_DIR, INTERVIEW_VIDEOS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
