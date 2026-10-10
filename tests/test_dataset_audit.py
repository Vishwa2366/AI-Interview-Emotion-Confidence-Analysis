import pytest

from config import EMOTION_CLASSES
from src.dataset_audit import audit_dataset, validate_dataset


def create_dataset(root):
    for split in ("train", "test"):
        for class_name in EMOTION_CLASSES:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").write_bytes(b"test")


def test_validate_dataset_counts_supported_images_by_split(tmp_path):
    create_dataset(tmp_path)
    class_dir = tmp_path / "train" / "angry"
    for extension in (".jpeg", ".png", ".bmp"):
        (class_dir / f"extra{extension}").write_bytes(b"test")
    (class_dir / "ignored.txt").write_text("not an image")

    data = validate_dataset(tmp_path)

    assert data["counts"]["angry"] == 4
    assert data["split_counts"]["train"]["angry"] == 4
    assert data["split_counts"]["test"]["angry"] == 1
    assert len(data["paths"]) == 17


def test_validate_dataset_reports_missing_split_class_folders(tmp_path):
    (tmp_path / "train" / "angry").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"train/disgust.*test/angry"):
        validate_dataset(tmp_path)


def test_validate_dataset_reports_empty_classes(tmp_path):
    create_dataset(tmp_path)
    (tmp_path / "test" / "sad" / "sample.jpg").unlink()

    with pytest.raises(ValueError, match="test/sad"):
        validate_dataset(tmp_path)


def test_audit_report_and_plot_distinguish_splits(tmp_path):
    data_dir = tmp_path / "dataset"
    graph_dir = tmp_path / "graphs"
    create_dataset(data_dir)

    report = audit_dataset(data_dir, graph_dir)

    assert report["total_images"] == 14
    assert report["train_class_counts"]["angry"] == 1
    assert report["test_class_counts"]["angry"] == 1
    assert report["split_class_counts"]["test"]["surprise"] == 1
    assert (graph_dir / "class_distribution.png").is_file()
