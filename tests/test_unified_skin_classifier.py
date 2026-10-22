import io
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from unified_skin_classifier import UnifiedSkinPredictor


class _FixedOutputModel:
    def __call__(self, image):
        return torch.tensor([[0.0, 1.0, 3.0]])


class UnifiedSkinPredictorTests(unittest.TestCase):
    def test_prediction_uses_one_shared_disease_label_space(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), color=(180, 90, 70)).save(buffer, format="JPEG")
        classes = ["BA-cellulitis", "FU-ringworm", "PA-cutaneous-larva-migrans"]
        model = UnifiedSkinPredictor()

        with patch(
            "unified_skin_classifier.get_model",
            return_value=(
                _FixedOutputModel(),
                classes,
                {
                    "test_accuracy": 0.9,
                    "test_accuracy_by_group": {"parasitic": 0.8},
                },
            ),
        ):
            result = model.predict_from_bytes(buffer.getvalue(), topk=3)

        self.assertEqual(result["prediction"], "PA-cutaneous-larva-migrans")
        self.assertEqual(
            [row["prediction"] for row in result["topk"]],
            ["PA-cutaneous-larva-migrans", "FU-ringworm", "BA-cellulitis"],
        )
        self.assertEqual(result["method"], "unified_8_class_classifier")
        self.assertEqual(result["evaluation_accuracy"], 0.9)
        self.assertEqual(result["evaluation_accuracy_by_group"], {"parasitic": 0.8})


if __name__ == "__main__":
    unittest.main(verbosity=2)
