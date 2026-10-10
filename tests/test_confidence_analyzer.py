from src.confidence_analyzer import ConfidenceAnalyzer


def test_confidence_score_is_bounded():
    analyzer = ConfidenceAnalyzer()
    frames = [
        {"face_visibility": 1.0, "eye_contact": 1.0, "head_stability": 1.0,
         "expression_stability": 1.0, "emotion_stability": 1.0, "positive_neutral_ratio": 1.0},
        {"face_visibility": 0.0, "eye_contact": 0.0, "head_stability": 0.0,
         "expression_stability": 0.0, "emotion_stability": 0.0, "positive_neutral_ratio": 0.0},
    ]
    result = analyzer.analyze(frames)
    assert 0.0 <= result["behavioral_confidence_score"] <= 100.0
    assert result["frames_analyzed"] == 2
