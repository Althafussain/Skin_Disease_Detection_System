from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from Bacterial_Skin_Disease.PredictModel import (
    get_model as get_bacterial_model,
    predict_from_bytes as predict_bacterial_from_bytes,
)
from Parastic_Skin_Disease.Predict import (
    get_model as get_parasitic_model,
    predict_from_bytes as predict_parasitic_from_bytes,
)
from Viral.PredictModel import (
    get_model as get_viral_model,
    predict_from_bytes as predict_viral_from_bytes,
)
from fedavg_utils import fedavg_label_probabilities
from fungal.PredictModel import (
    get_model as get_fungal_model,
    predict_from_bytes as predict_fungal_from_bytes,
)


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATHS: Dict[str, Path] = {
    "viral": BASE_DIR / "Viral" / "saved_viral_models" / "viral_skin_model.pth",
    "parasitic": BASE_DIR / "Parastic_Skin_Disease" / "checkpoints" / "parasitic_local.pth",
    "fungal": BASE_DIR / "fungal" / "trained_model.pth",
    "bacterial": BASE_DIR / "Bacterial_Skin_Disease" / "bacterial_model.pth",
}

MODEL_FILE_HINTS = {
    "viral": ("viral_skin_model.pth",),
    "parasitic": ("parasitic_local.pth",),
    "fungal": ("trained_model.pth",),
    "bacterial": ("bacterial_model.pth",),
}

CLIENT_PREDICTORS = {
    "viral": predict_viral_from_bytes,
    "parasitic": predict_parasitic_from_bytes,
    "fungal": predict_fungal_from_bytes,
    "bacterial": predict_bacterial_from_bytes,
}

CLIENT_MODEL_GETTERS = {
    "viral": get_viral_model,
    "parasitic": get_parasitic_model,
    "fungal": get_fungal_model,
    "bacterial": get_bacterial_model,
}


def resolve_model_path(client_name: str, model_path: Optional[str | Path]) -> Path:
    if client_name not in DEFAULT_MODEL_PATHS:
        raise ValueError(f"Unknown client name: {client_name}")

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATHS[client_name]
    if path.is_file():
        return path.resolve()

    if path.is_dir():
        for filename in MODEL_FILE_HINTS.get(client_name, ()):
            candidate = path / filename
            if candidate.exists():
                return candidate.resolve()

        fallback_candidates = sorted(path.glob("*.pth"))
        if fallback_candidates:
            return fallback_candidates[-1].resolve()
        raise FileNotFoundError(f"No .pth model file found in directory: {path}")

    raise FileNotFoundError(f"Model path not found: {path}")


def _probability_map_from_prediction(prediction: Mapping[str, Any]) -> Dict[str, float]:
    probabilities: Dict[str, float] = {}

    topk = prediction.get("topk", [])
    if isinstance(topk, list):
        for row in topk:
            if not isinstance(row, dict):
                continue
            label = row.get("prediction")
            confidence = row.get("confidence")
            if label is None or confidence is None:
                continue
            probabilities[str(label)] = max(float(confidence), probabilities.get(str(label), 0.0))

    label = prediction.get("prediction")
    confidence = prediction.get("confidence")
    if label is not None and confidence is not None and str(label) != "Skin not found":
        probabilities.setdefault(str(label), float(confidence))

    return probabilities


def _to_topk(prob_map: Mapping[str, float], k: int) -> list:
    ordered = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for label, confidence in ordered[: max(1, k)]:
        result.append({"prediction": label, "confidence": float(confidence)})
    return result


class MultiModelFedAvgPredictor:
    def __init__(
        self,
        model_paths: Optional[Mapping[str, str | Path]] = None,
        client_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.model_paths = {
            client: resolve_model_path(client, None if model_paths is None else model_paths.get(client))
            for client in CLIENT_PREDICTORS
        }
        self.client_weights = dict(client_weights) if client_weights is not None else None
        self._labels_cache: Optional[Dict[str, list]] = None

    def get_label_catalog(self, force_reload: bool = False) -> Dict[str, list]:
        if self._labels_cache is not None and not force_reload:
            return self._labels_cache

        labels_by_client = {}
        for client_name, get_model in CLIENT_MODEL_GETTERS.items():
            _, classes, _ = get_model(model_path=self.model_paths[client_name], force_reload=force_reload)
            labels_by_client[client_name] = list(classes)
        self._labels_cache = labels_by_client
        return labels_by_client

    def predict_from_bytes(self, image_bytes: bytes, topk: int = 5) -> Dict[str, Any]:
        per_model_predictions: Dict[str, Dict[str, Any]] = {}
        per_model_probabilities: Dict[str, Dict[str, float]] = {}

        for client_name, predictor in CLIENT_PREDICTORS.items():
            prediction = predictor(
                image_bytes,
                model_path=self.model_paths[client_name],
                topk=50,  # 50 is safely above current class counts and returns full local distribution.
            )
            per_model_predictions[client_name] = prediction
            per_model_probabilities[client_name] = _probability_map_from_prediction(prediction)

        fedavg_probs = fedavg_label_probabilities(
            per_model_probabilities,
            client_weights=self.client_weights,
            normalize_output=True,
        )
        fedavg_topk = _to_topk(fedavg_probs, topk)

        best_label = fedavg_topk[0]["prediction"] if fedavg_topk else None
        best_confidence = fedavg_topk[0]["confidence"] if fedavg_topk else 0.0

        return {
            "prediction": best_label,
            "confidence": float(best_confidence),
            "topk": fedavg_topk,
            "method": "fedavg_label_probabilities",
            "client_weights": self.client_weights
            if self.client_weights is not None
            else {name: 1.0 for name in CLIENT_PREDICTORS},
            "labels_by_model": self.get_label_catalog(),
            "model_paths": {name: str(path) for name, path in self.model_paths.items()},
            "model_predictions": per_model_predictions,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run integrated 4-model prediction with FedAvg aggregation.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k labels to return from FedAvg output.")
    parser.add_argument("--viral-model", default=None, help="Viral model path or directory.")
    parser.add_argument("--parasitic-model", default=None, help="Parasitic model path or directory.")
    parser.add_argument("--fungal-model", default=None, help="Fungal model path or directory.")
    parser.add_argument("--bacterial-model", default=None, help="Bacterial model path or directory.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    predictor = MultiModelFedAvgPredictor(
        model_paths={
            "viral": args.viral_model,
            "parasitic": args.parasitic_model,
            "fungal": args.fungal_model,
            "bacterial": args.bacterial_model,
        }
    )
    result = predictor.predict_from_bytes(image_path.read_bytes(), topk=max(1, args.topk))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
