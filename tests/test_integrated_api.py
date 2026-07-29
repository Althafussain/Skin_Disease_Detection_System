import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import integrated_api


client = TestClient(integrated_api.app)


def _make_image_bytes(fmt="JPEG", size=(64, 64), color=(10, 120, 200)):
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class RootAndLabelsTests(unittest.TestCase):
    def test_root_status(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("aggregation_method", body)

    def test_labels_endpoint(self):
        fake_labels = {"viral": ["Healthy", "None"], "bacterial": ["Normal", "None"]}
        with patch.object(integrated_api.predictor, "get_label_catalog", return_value=fake_labels):
            response = client.get("/labels")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labels_by_model"], fake_labels)


class SymptomCatalogAndSuggestTests(unittest.TestCase):
    def test_symptom_catalog_returns_list(self):
        response = client.get("/symptoms/catalog")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()["symptoms"], list)

    def test_suggest_symptoms_success(self):
        payload = {"topk": [{"prediction": "Scabies", "confidence": 0.7}], "limit": 3}
        response = client.post("/symptoms/suggest", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("show_suggestions", body)
        self.assertIn("suggestions", body)

    def test_suggest_symptoms_invalid_limit_rejected(self):
        # Pydantic Field(ge=1, le=20) on `limit` must reject out-of-range values.
        response = client.post("/symptoms/suggest", json={"topk": [], "limit": 0})
        self.assertEqual(response.status_code, 422)


class FinalPredictTests(unittest.TestCase):
    def test_final_predict_requires_symptoms_or_keywords(self):
        response = client.post(
            "/symptoms/final-predict",
            json={"selected_symptoms": [], "custom_keywords": [], "image_topk": []},
        )
        self.assertEqual(response.status_code, 400)

    def test_final_predict_with_keywords_returns_diagnosis(self):
        response = client.post(
            "/symptoms/final-predict",
            json={
                "selected_symptoms": [],
                "custom_keywords": ["scabies", "itching"],
                "image_topk": [],
                "limit": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("final_prediction", body)
        self.assertIn("medical_notice", body)


class PredictEndpointTests(unittest.TestCase):
    """Covers the production /predict route: success path, input validation,
    and error handling for invalid/oversized/corrupted uploads."""

    def test_predict_success_with_mocked_model(self):
        fake_result = {
            "prediction": "Healthy",
            "confidence": 0.91,
            "topk": [{"prediction": "Healthy", "confidence": 0.91}],
        }
        with patch.object(integrated_api.predictor, "predict_from_bytes", return_value=fake_result):
            files = {"file": ("sample.jpg", _make_image_bytes(), "image/jpeg")}
            response = client.post("/predict?topk=5", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "Healthy")

    def test_predict_rejects_missing_file(self):
        response = client.post("/predict")
        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_unsupported_content_type(self):
        files = {"file": ("notes.txt", b"just some text", "text/plain")}
        response = client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_predict_rejects_empty_file(self):
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_predict_rejects_oversized_file(self):
        oversized = b"\x00" * (integrated_api.MAX_UPLOAD_BYTES + 1)
        files = {"file": ("big.jpg", oversized, "image/jpeg")}
        response = client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("size limit", response.json()["detail"].lower())

    def test_predict_rejects_corrupted_image_bytes(self):
        # Content-type header claims an image, but the bytes have no valid
        # image header at all, so Pillow must raise before any model runs.
        garbage = b"this is definitely not an image file" * 5
        files = {"file": ("corrupt.jpg", garbage, "image/jpeg")}
        response = client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a valid", response.json()["detail"].lower())

    def test_predict_maps_missing_model_file_to_500(self):
        with patch.object(
            integrated_api.predictor,
            "predict_from_bytes",
            side_effect=FileNotFoundError("model checkpoint missing"),
        ):
            files = {"file": ("sample.jpg", _make_image_bytes(), "image/jpeg")}
            response = client.post("/predict", files=files)
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
