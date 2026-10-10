import numpy as np

from src.preprocessing import normalize_image, preprocess_grayscale, preprocess_rgb


def test_preprocessing_returns_expected_shapes_and_range():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    grayscale = preprocess_grayscale(image)
    rgb = preprocess_rgb(image)
    assert grayscale.shape == (48, 48)
    assert rgb.shape == (48, 48, 3)
    assert np.all((grayscale >= 0.0) & (grayscale <= 1.0))
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))
    assert normalize_image(np.array([0, 255], dtype=np.uint8)).shape == (2,)


def test_preprocessing_handles_invalid_dimensions():
    try:
        preprocess_grayscale(np.zeros((0, 0, 3), dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid image dimensions must raise ValueError")
