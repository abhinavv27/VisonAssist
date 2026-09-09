"""Fine-tune a VisionAssist YOLO model on a checked labeled dataset.

Expected dataset layout:
    dataset/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      dataset.yaml

Each label file uses standard YOLO format:
    class_id x_center y_center width height

This script intentionally refuses to train when images or labels are missing.
A small set of guessed or synthetic labels is not enough to improve a safety
model reliably.
"""

import argparse
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def validate_split(root: Path, split: str) -> int:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images = image_files(images_dir)
    missing = [path for path in images if not (labels_dir / f"{path.stem}.txt").exists()]
    if not images:
        raise ValueError(f"No images found in {images_dir}")
    if missing:
        examples = ", ".join(path.name for path in missing[:3])
        raise ValueError(f"Missing labels for {len(missing)} {split} images: {examples}")
    return len(images)


def validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    root = data_yaml.parent
    train_count = validate_split(root, "train")
    val_count = validate_split(root, "val")
    print(f"Validated dataset: {train_count} train images, {val_count} validation images")


def train(
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    device: str | None,
    name: str,
) -> None:
    validate_dataset(data_yaml)
    from ultralytics import YOLO

    detector = YOLO(model)
    options = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "project": str(data_yaml.parent / "runs"),
        "name": name,
        "patience": 20,
        "pretrained": True,
        "plots": True,
        "verbose": True,
    }
    if device:
        options["device"] = device
    detector.train(**options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for VisionAssist")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="Starting weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="CPU, CUDA index, or leave unset")
    parser.add_argument("--name", default="visionassist-finetune")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model, args.epochs, args.imgsz, args.device, args.name)
