import argparse
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from .model_builder import build_fungal_model
except ImportError:
    from model_builder import build_fungal_model


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = BASE_DIR / "trained_model.pth"
TRAIN_DIR = BASE_DIR / "Fungal-Dataset" / "Train"

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_MODEL_CACHE = {}


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


def load_model(model_path=MODEL_PATH, num_classes=None):
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

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

    model = build_fungal_model(num_classes=target_num_classes, train_from_scratch=True).to(DEVICE)
    _strict_load(model, state_dict, checkpoint_path)
    model.eval()
    return model, classes, metadata


def get_model(model_path=MODEL_PATH, force_reload=False):
    resolved = Path(model_path).resolve()
    cache_key = str(resolved)
    if force_reload or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(model_path=resolved)
    return _MODEL_CACHE[cache_key]


def preprocess_pil(pil_img):
    pil_img = pil_img.convert("RGB")
    x = _TRANSFORM(pil_img).unsqueeze(0)
    return x.to(DEVICE)


@torch.no_grad()
def predict_from_pil(pil_img, topk=3, model_path=MODEL_PATH):
    model, classes, _ = get_model(model_path=model_path)
    x = preprocess_pil(pil_img)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    conf = float(conf.item())
    label = classes[int(idx)]

    k = min(topk, len(classes))
    top_probs, top_idxs = torch.topk(probs, k=k)
    topk_list = [
        {"prediction": classes[int(i)], "confidence": float(p)}
        for p, i in zip(top_probs.tolist(), top_idxs.tolist())
    ]

    return {
        "prediction": label,
        "confidence": conf,
        "topk": topk_list,
    }


@torch.no_grad()
def predict_from_bytes(image_bytes, topk=3, model_path=MODEL_PATH):
    pil = Image.open(io.BytesIO(image_bytes))
    return predict_from_pil(pil, topk=topk, model_path=model_path)


@torch.no_grad()
def predict_from_path(image_path, topk=3, model_path=MODEL_PATH):
    pil = Image.open(image_path)
    return predict_from_pil(pil, topk=topk, model_path=model_path)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run fungal skin disease prediction from the terminal.")
    parser.add_argument("image", nargs="?", help="Path to the image file to predict.")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to print.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to the model checkpoint.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    image_path = Path(args.image) if args.image else BASE_DIR / "test.jpg"
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}. "
            f"Pass an image path, for example: python PredictModel.py {BASE_DIR / 'test.jpg'}"
        )

    result = predict_from_path(
        str(image_path),
        topk=max(1, args.topk),
        model_path=args.model_path,
    )
    print(json.dumps(result, indent=2))
