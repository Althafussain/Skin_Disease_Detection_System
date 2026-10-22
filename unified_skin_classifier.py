from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "unified_skin_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
_MODEL_CACHE: Dict[str, tuple] = {}


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _group_accuracies(metrics: Dict[str, Any]) -> Dict[str, float]:
    saved = metrics.get("test_accuracy_by_group")
    if isinstance(saved, dict):
        return saved

    groups: Dict[str, Dict[str, int]] = {}
    for class_name, counts in metrics.get("per_class", {}).items():
        prefix = class_name.split("-", maxsplit=1)[0].strip().lower()
        group = {
            "ba": "bacterial",
            "fu": "fungal",
            "pa": "parasitic",
            "vi": "viral",
        }.get(prefix, "other")
        totals = groups.setdefault(group, {"correct": 0, "total": 0})
        totals["correct"] += int(counts.get("correct", 0))
        totals["total"] += int(counts.get("total", 0))

    return {
        group: counts["correct"] / counts["total"]
        for group, counts in groups.items()
        if counts["total"]
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Unified skin classifier not found: {checkpoint_path}. "
            "Train it with `python train_unified_model.py`."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    classes = checkpoint.get("classes")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(classes, list) or len(classes) < 2 or not isinstance(state_dict, dict):
        raise ValueError(f"Invalid unified classifier checkpoint: {checkpoint_path}")

    model = build_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, classes, checkpoint.get("metrics", {})


def get_model(model_path: str | Path = DEFAULT_MODEL_PATH, force_reload: bool = False):
    resolved_path = Path(model_path).resolve()
    cache_key = str(resolved_path)
    if force_reload or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(resolved_path)
    return _MODEL_CACHE[cache_key]


def get_label_catalog(model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict[str, list]:
    _, classes, _ = get_model(model_path)
    return {"unified": list(classes)}


@torch.no_grad()
def predict_from_bytes(
    image_bytes: bytes,
    topk: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    model, classes, metrics = get_model(model_path)
    with Image.open(io.BytesIO(image_bytes)) as image:
        tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    count = min(max(1, topk), len(classes))
    scores, indexes = torch.topk(probabilities, k=count)
    results = [
        {"prediction": classes[int(index)], "confidence": float(score)}
        for score, index in zip(scores.cpu().tolist(), indexes.cpu().tolist())
    ]
    return {
        "prediction": results[0]["prediction"],
        "confidence": results[0]["confidence"],
        "topk": results,
        "method": "unified_8_class_classifier",
        "model_name": "Unified skin disease classifier",
        "evaluation_accuracy": metrics.get("test_accuracy"),
        "evaluation_accuracy_by_group": _group_accuracies(metrics),
    }


class UnifiedSkinPredictor:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)

    def get_label_catalog(self, force_reload: bool = False) -> Dict[str, list]:
        if force_reload:
            get_model(self.model_path, force_reload=True)
        return get_label_catalog(self.model_path)

    def predict_from_bytes(self, image_bytes: bytes, topk: int = 5) -> Dict[str, Any]:
        return predict_from_bytes(image_bytes, topk=topk, model_path=self.model_path)
