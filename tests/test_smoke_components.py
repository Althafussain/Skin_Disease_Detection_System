import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from Bacterial_Skin_Disease.PredictModel import (
    load_model as load_bacterial_model,
    predict_from_bytes as predict_bacterial_from_bytes,
)
from Bacterial_Skin_Disease.model_builder import build_bacterial_model
from Parastic_Skin_Disease.Predict import (
    load_model as load_parastic_model,
    predict_from_bytes as predict_parastic_from_bytes,
)
from Parastic_Skin_Disease.model_builder import build_parasitic_model
from Viral.PredictModel import load_model as load_viral_model, predict_from_bytes as predict_viral_from_bytes
from Viral.model_builder import build_viral_model
from fungal.PredictModel import load_model as load_fungal_model, predict_from_bytes as predict_fungal_from_bytes
from fungal.model_builder import build_fungal_model
import Bacterial_Skin_Disease.api as bacterial_api
import Parastic_Skin_Disease.api as parastic_api
import Viral.api as viral_api
import fungal.main as fungal_api


def _make_image_bytes():
    image = Image.new("RGB", (224, 224), color=(12, 128, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _save_checkpoint(model, path, classes):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": len(classes),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "epochs": 1,
            "final_loss": 0.1,
            "final_accuracy": 99.9,
            "train_from_scratch": True,
            "model_architecture": "smoke_test",
        },
        path,
    )


class ComponentSmokeTests(unittest.TestCase):
    def test_viral_build_forward_checkpoint_and_prediction_schema(self):
        classes = ["Healthy", "None", "VI-chickenpox", "VI-shingles"]
        model = build_viral_model(num_classes=len(classes), train_from_scratch=True)
        out = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, len(classes)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "viral_smoke.pth"
            _save_checkpoint(model, checkpoint_path, classes)
            _, loaded_classes, _ = load_viral_model(model_path=checkpoint_path)
            self.assertEqual(loaded_classes, classes)

            result = predict_viral_from_bytes(_make_image_bytes(), model_path=checkpoint_path, topk=3)
            self.assertTrue({"prediction", "confidence", "topk"} <= set(result.keys()))

    def test_bacterial_build_forward_checkpoint_and_prediction_schema(self):
        classes = ["BA- cellulitis", "BA-Ery", "BA-impetigo", "None", "Normal"]
        model = build_bacterial_model(num_classes=len(classes), train_from_scratch=True)
        out = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, len(classes)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "bacterial_smoke.pth"
            _save_checkpoint(model, checkpoint_path, classes)
            _, loaded_classes, _ = load_bacterial_model(model_path=checkpoint_path)
            self.assertEqual(loaded_classes, classes)

            result = predict_bacterial_from_bytes(_make_image_bytes(), model_path=checkpoint_path, topk=3)
            self.assertTrue({"prediction", "confidence", "topk", "threshold", "skin_found"} <= set(result.keys()))

    def test_parastic_build_forward_checkpoint_and_prediction_schema(self):
        classes = ["None", "NormalSkin", "PA-cutaneous-larva-migrans"]
        model = build_parasitic_model(num_classes=len(classes), train_from_scratch=True)
        out = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, len(classes)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "parastic_smoke.pth"
            _save_checkpoint(model, checkpoint_path, classes)
            _, loaded_classes, _ = load_parastic_model(model_path=checkpoint_path)
            self.assertEqual(loaded_classes, classes)

            result = predict_parastic_from_bytes(_make_image_bytes(), model_path=checkpoint_path, topk=3)
            self.assertTrue({"prediction", "confidence", "topk"} <= set(result.keys()))

    def test_single_class_parasitic_checkpoint_is_not_reported_as_a_prediction(self):
        classes = ["PA-cutaneous-larva-migrans"]
        model = build_parasitic_model(num_classes=len(classes), train_from_scratch=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "parasitic_single_class.pth"
            _save_checkpoint(model, checkpoint_path, classes)

            result = predict_parastic_from_bytes(_make_image_bytes(), model_path=checkpoint_path)

        self.assertIsNone(result["prediction"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["topk"], [])
        self.assertIn("fewer than two output classes", result["warning"])

    def test_fungal_build_forward_checkpoint_and_prediction_schema(self):
        classes = ["FU-athlete-foot", "FU-nail-fungus", "FU-ringworm", "None"]
        model = build_fungal_model(num_classes=len(classes), train_from_scratch=True)
        out = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, len(classes)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "fungal_smoke.pth"
            _save_checkpoint(model, checkpoint_path, classes)
            _, loaded_classes, _ = load_fungal_model(model_path=checkpoint_path)
            self.assertEqual(loaded_classes, classes)

            result = predict_fungal_from_bytes(_make_image_bytes(), model_path=checkpoint_path, topk=3)
            self.assertTrue({"prediction", "confidence", "topk"} <= set(result.keys()))

    def test_api_predict_routes_return_json_schema(self):
        image_bytes = _make_image_bytes()
        files = {"file": ("smoke.jpg", image_bytes, "image/jpeg")}

        with patch.object(
            viral_api,
            "predict_from_bytes",
            return_value={"prediction": "Healthy", "confidence": 0.9, "topk": []},
        ):
            client = TestClient(viral_api.app)
            response = client.post("/predict", files=files)
            self.assertEqual(response.status_code, 200)
            self.assertIn("prediction", response.json())

        with patch.object(
            bacterial_api,
            "predict_from_bytes",
            return_value={"prediction": "Normal", "confidence": 0.8, "topk": [], "threshold": 0.6, "skin_found": True},
        ):
            client = TestClient(bacterial_api.app)
            response = client.post("/predict", files=files)
            self.assertEqual(response.status_code, 200)
            self.assertTrue({"prediction", "confidence"} <= set(response.json().keys()))

        with patch.object(
            parastic_api,
            "predict_from_bytes",
            return_value={"prediction": "None", "confidence": 0.7, "topk": []},
        ):
            client = TestClient(parastic_api.app)
            response = client.post("/predict", files=files)
            self.assertEqual(response.status_code, 200)
            self.assertIn("prediction", response.json())

        with patch.object(
            fungal_api,
            "predict_from_bytes",
            return_value={"prediction": "None", "confidence": 0.6, "topk": []},
        ):
            client = TestClient(fungal_api.app)
            response = client.post("/predict", files=files)
            self.assertEqual(response.status_code, 200)
            self.assertIn("prediction", response.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
