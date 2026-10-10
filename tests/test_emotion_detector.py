import numpy as np

from src.model_utils import normalize_prediction


def test_emotion_prediction_output_format():
    names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    probabilities = np.array([0.1, 0.05, 0.05, 0.55, 0.1, 0.05, 0.1])
    result = normalize_prediction(probabilities, names)
    assert result["emotion"] == "happy"
    assert result["probability"] == 0.55
    assert set(result["all_class_probabilities"]) == set(names)
    assert abs(sum(result["all_class_probabilities"].values()) - 1.0) < 1e-8
