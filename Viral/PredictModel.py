import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from .model_builder import build_viral_model
except ImportError:
    from model_builder import build_viral_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "Viral-Dataset" / "Train"
DEFAULT_MODEL_PATH = BASE_DIR / "saved_viral_models" / "viral_skin_model.pth"

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_MODEL_CACHE = {}


def _resolve_default_model_path():
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    legacy_candidates = sorted((BASE_DIR / "saved_viral_models").glob("viral_skin_model_*epochs.pth"))
    if legacy_candidates:
        return legacy_candidates[-1]
    return DEFAULT_MODEL_PATH


def _classes_from_train_dir(train_dir):
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not classes:
        raise RuntimeError(f"No class folders found in: {train_dir}")
    return classes


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("model_state_dict"), dict):
        return checkpoint_obj["model_state_dict"], checkpoint_obj
    if isinstance(checkpoint_obj, dict) and checkpoint_obj and all(
        torch.is_tensor(v) for v in checkpoint_obj.values()
    ):
        return checkpoint_obj, {}
    raise ValueError("Unsupported checkpoint format. Expected checkpoint dict or raw state_dict.")


def _resolve_classes(metadata, train_dir):
    classes = metadata.get("classes")
    if isinstance(classes, list) and classes:
        return classes

    class_to_idx = metadata.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
        return [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    return _classes_from_train_dir(train_dir)


def _strict_load(model, state_dict, checkpoint_path):
    model_state = model.state_dict()
    mismatched = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    if mismatched:
        first_key, got_shape, expected_shape = mismatched[0]
        raise ValueError(
            f"Checkpoint shape mismatch for '{first_key}' in {checkpoint_path}: "
            f"got {got_shape}, expected {expected_shape}"
        )

    missing_keys = sorted(set(model_state.keys()) - set(state_dict.keys()))
    unexpected_keys = sorted(set(state_dict.keys()) - set(model_state.keys()))
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"Checkpoint keys mismatch for {checkpoint_path}. "
            f"missing_keys={missing_keys[:5]} unexpected_keys={unexpected_keys[:5]}"
        )

    model.load_state_dict(state_dict, strict=True)


def load_model(num_classes=None, model_path=None):
    checkpoint_path = Path(model_path) if model_path is not None else _resolve_default_model_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict, metadata = _extract_state_dict(checkpoint_obj)
    classes = _resolve_classes(metadata, TRAIN_DIR)

    inferred_num_classes = metadata.get("num_classes") or len(classes)
    target_num_classes = num_classes if num_classes is not None else inferred_num_classes
    if target_num_classes != len(classes):
        raise ValueError(
            f"Class count mismatch for {checkpoint_path}: "
            f"model num_classes={target_num_classes}, classes={len(classes)}"
        )

    model = build_viral_model(num_classes=target_num_classes, train_from_scratch=True).to(DEVICE)
    _strict_load(model, state_dict, checkpoint_path)
    model.eval()
    return model, classes, metadata


def get_model(model_path=None, force_reload=False):
    resolved_path = Path(model_path) if model_path is not None else _resolve_default_model_path()
    cache_key = str(resolved_path.resolve())
    if force_reload or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(model_path=resolved_path)
    return _MODEL_CACHE[cache_key]


def preprocess_pil(image):
    image = image.convert("RGB")
    return _TRANSFORM(image).unsqueeze(0).to(DEVICE)


def preprocess_image(image_path):
    return preprocess_pil(Image.open(image_path))


@torch.no_grad()
def predict_from_pil(image, model_path=None, topk=3):
    model, classes, _ = get_model(model_path=model_path)
    x = preprocess_pil(image)
    outputs = model(x)
    probabilities = torch.softmax(outputs, dim=1).squeeze(0)

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    label = classes[int(predicted_idx)]

    k = min(topk, len(classes))
    top_probs, top_idxs = torch.topk(probabilities, k)
    topk_predictions = [
        {"prediction": classes[int(i)], "confidence": float(p)}
        for p, i in zip(top_probs.tolist(), top_idxs.tolist())
    ]

    return {
        "prediction": label,
        "confidence": float(confidence.item()),
        "topk": topk_predictions,
    }


@torch.no_grad()
def predict_from_bytes(image_bytes, model_path=None, topk=3):
    image = Image.open(io.BytesIO(image_bytes))
    return predict_from_pil(image, model_path=model_path, topk=topk)


@torch.no_grad()
def predict_from_path(image_path, model_path=None, topk=3):
    image = Image.open(image_path)
    return predict_from_pil(image, model_path=model_path, topk=topk)


def predict(model, image_path, classes=None):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100.0

    local_classes = classes if classes is not None else _classes_from_train_dir(TRAIN_DIR)
    disease = local_classes[predicted_idx]
    return f"Viral ({disease}) - {confidence:.2f}% confidence"


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run viral skin disease prediction from the terminal.")
    parser.add_argument("image", nargs="?", help="Path to the image file to predict.")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to print.")
    parser.add_argument("--model-path", default=str(_resolve_default_model_path()), help="Path to the model checkpoint.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    image_path = Path(args.image) if args.image else BASE_DIR / "t1.jpeg"
    if not image_path.exists():
        print(
            f"Input image not found: {image_path}\n"
            f"Run with an image path, for example:\n"
            f"python3 Viral/PredictModel.py /absolute/or/relative/path/to/image.jpg"
        )
    else:
        result = predict_from_path(
            str(image_path),
            model_path=args.model_path,
            topk=max(1, args.topk),
        )
        print(json.dumps(result, indent=2))
