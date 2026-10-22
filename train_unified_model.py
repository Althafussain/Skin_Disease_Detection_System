from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from unified_skin_classifier import BASE_DIR, DEFAULT_MODEL_PATH, DEVICE, build_model


DATASET_DIRS = {
    "Bacterial_Skin_Disease": BASE_DIR / "Bacterial_Skin_Disease" / "Bacterial-Dataset",
    "Viral": BASE_DIR / "Viral" / "Viral-Dataset",
    "Parasitic": BASE_DIR / "Parastic_Skin_Disease" / "Parastic-Dataset",
    "Fungal": BASE_DIR / "fungal" / "Fungal-Dataset",
}
IMAGE_SIZE = 224


class MildImageAugmentation:
    def __call__(self, image):
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        return image


def _transforms(training: bool):
    pipeline = []
    if training:
        pipeline.append(MildImageAugmentation())
    pipeline.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(pipeline)


def _load_split(split: str, training: bool, class_to_index: Dict[str, int]):
    datasets_by_group = []
    class_counts: Dict[str, int] = {}
    for group, dataset_root in DATASET_DIRS.items():
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing {split} dataset directory: {split_root}")

        image_dataset = datasets.ImageFolder(
            str(split_root),
            transform=_transforms(training),
        )
        if not image_dataset.samples:
            raise ValueError(f"No labeled images found under {split_root}")

        local_to_global = {
            local_index: class_to_index[class_name]
            for class_name, local_index in image_dataset.class_to_idx.items()
        }
        image_dataset.target_transform = lambda target, mapping=local_to_global: mapping[target]
        datasets_by_group.append(image_dataset)
        sample_counts = Counter(image_dataset.targets)
        for class_name, local_index in image_dataset.class_to_idx.items():
            class_counts[class_name] = class_counts.get(class_name, 0) + sample_counts[local_index]

    return ConcatDataset(datasets_by_group), class_counts


def _evaluate(model, loader, classes: List[str]) -> Tuple[float, Dict[str, Dict[str, int]]]:
    model.eval()
    correct = 0
    total = 0
    per_class = {name: {"correct": 0, "total": 0} for name in classes}
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            matches = predictions.eq(targets)
            correct += int(matches.sum().item())
            total += targets.numel()
            for target, matched in zip(targets.cpu().tolist(), matches.cpu().tolist()):
                label = classes[target]
                per_class[label]["total"] += 1
                per_class[label]["correct"] += int(matched)
    accuracy = correct / total if total else 0.0
    return accuracy, per_class


def train(
    model_path: Path = DEFAULT_MODEL_PATH,
    epochs: int = 12,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    workers: int = 0,
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_roots = [root / "Train" for root in DATASET_DIRS.values()]
    classes = sorted(
        {
            class_dir.name
            for root in train_roots
            if root.is_dir()
            for class_dir in root.iterdir()
            if class_dir.is_dir()
        }
    )
    if len(classes) < 2:
        raise ValueError("At least two disease classes are required to train the unified model.")
    class_to_index = {name: index for index, name in enumerate(classes)}

    train_dataset, train_counts = _load_split("Train", True, class_to_index)
    test_dataset, test_counts = _load_split("Test", False, class_to_index)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=DEVICE.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=DEVICE.type == "cuda",
    )

    model = build_model(num_classes=len(classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total += targets.numel()
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / max(1, total):.4f}, "
            f"train_accuracy={correct / max(1, total):.4f}"
        )

    test_accuracy, per_class = _evaluate(model, test_loader, classes)
    per_group_counts: Dict[str, Dict[str, int]] = {}
    for class_name, counts in per_class.items():
        prefix = class_name.split("-", maxsplit=1)[0].strip().lower()
        group = {
            "ba": "bacterial",
            "fu": "fungal",
            "pa": "parasitic",
            "vi": "viral",
        }.get(prefix, "other")
        group_counts = per_group_counts.setdefault(group, {"correct": 0, "total": 0})
        group_counts["correct"] += counts["correct"]
        group_counts["total"] += counts["total"]

    metrics = {
        "test_accuracy": test_accuracy,
        "test_accuracy_by_group": {
            group: counts["correct"] / counts["total"] if counts["total"] else 0.0
            for group, counts in per_group_counts.items()
        },
        "per_class": {
            name: {
                **counts,
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in per_class.items()
        },
        "train_image_counts": train_counts,
        "test_image_counts": test_counts,
        "epochs": epochs,
        "training_source": "ImageNet pretrained MobileNetV2, fine-tuned on project train splits",
    }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": classes,
            "class_to_idx": class_to_index,
            "num_classes": len(classes),
            "metrics": metrics,
            "model_architecture": "mobilenet_v2",
            "image_size": IMAGE_SIZE,
        },
        destination,
    )
    print(f"Saved unified model: {destination.resolve()}")
    print(f"Held-out test accuracy: {test_accuracy:.2%}")
    for group, accuracy in metrics["test_accuracy_by_group"].items():
        print(f"  {group.capitalize()} classes: {accuracy:.2%}")
    for name, result in metrics["per_class"].items():
        print(f"  {name}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the unified 8-class skin disease model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    train(
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
