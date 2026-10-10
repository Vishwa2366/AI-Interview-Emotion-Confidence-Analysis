"""Image preprocessing and TensorFlow dataset utilities."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from config import IMAGE_SIZE, RANDOM_SEED

try:
    import tensorflow as tf
except ImportError:
    tf = None

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if tf is not None:
    tf.random.set_seed(RANDOM_SEED)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image in BGR order and apply a defensive error boundary."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """Resize an image while preserving its aspect ratio by the requested dimensions."""
    height, width = image.shape[:2]
    target_height, target_width = size
    scale = min(target_height / height, target_width / width) if height and width else 1.0
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Normalize an image to the 0-1 range."""
    normalized = image.astype(dtype) / 255.0
    return np.clip(normalized, 0.0, 1.0)


def preprocess_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale, resize, and normalize it."""
    if image.size == 0 or image.ndim < 3:
        raise ValueError("Image must contain at least one pixel and three dimensions.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = resize_image(grayscale, (IMAGE_SIZE, IMAGE_SIZE))
    return normalize_image(resized)


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image to RGB, resize, and normalize it."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = resize_image(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    return normalize_image(resized)


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply restrained random horizontal flip, rotation, zoom, and translation."""
    height, width = image.shape[:2]
    transform = tf.random.uniform_initializer(minval=0.0, maxval=1.0)
    flip = bool(tf.random.uniform(shape=(), minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5)
    if flip:
        image = tf.image.flip_left_right(image)
    rotation = tf.random.uniform(shape=(), minval=-0.08, maxval=0.08, dtype=tf.float32)
    zoom = tf.random.uniform(shape=(), minval=0.92, maxval=1.08, dtype=tf.float32)
    translation_x = tf.random.uniform(shape=(), minval=-0.04, maxval=0.04, dtype=tf.float32)
    translation_y = tf.random.uniform(shape=(), minval=-0.04, maxval=0.04, dtype=tf.float32)
    image = tf.cast(image, tf.float32)
    image = tf.image.stateless_random_apply(
        lambda value: tf.image.rotate(value, rotation),
        image,
        seed=[int(tf.random.uniform(shape=(), maxval=2**31, dtype=tf.int32)), int(tf.random.uniform(shape=(), maxval=2**31, dtype=tf.int32))],
        rate=1.0,
    )
    image = tf.image.stateless_random_zoom(
        image,
        zoom,
        translation_x,
        translation_y,
    )
    return tf.cast(image, tf.float32)


def create_tf_dataset(
    images: Iterable[str | Path],
    labels: Iterable[int],
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    """Create a reproducible TensorFlow dataset from image paths and labels."""
    image_paths = list(images)
    label_values = list(labels)
    if len(image_paths) != len(label_values):
        raise ValueError("Image and label counts must match.")
    if not image_paths:
        raise ValueError("No images were supplied for dataset creation.")

    paths = tf.constant(image_paths, dtype=tf.string)
    labels_tensor = tf.constant(label_values, dtype=tf.int32)

    def decode(path: tf.Tensor) -> tf.Tensor:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_png(image_bytes, channels=3) if path.numpy().endswith((b".png", b".jpg", b".jpeg")) else tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        if augment:
            image = tf.image.random_flip_left_right(image)
        return image

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_tensor))
    dataset = dataset.map(lambda path, label: (decode(path), label), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 10000))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def image_paths_for_class(root: Path, class_name: str) -> list[Path]:
    """Return readable image paths from a class directory."""
    directory = root / class_name
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
