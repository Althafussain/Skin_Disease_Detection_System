import unittest
from unittest.mock import patch

from multi_model_predictor import CLIENT_PREDICTORS, MultiModelSpecialistPredictor


def _prediction(rows):
    return {
        "prediction": rows[0]["prediction"],
        "confidence": rows[0]["confidence"],
        "topk": rows,
    }


class MultiModelPredictorTests(unittest.TestCase):
    def test_one_class_model_is_excluded_and_results_can_vary_by_image(self):
        outputs = {
            "viral": {
                b"viral": _prediction([
                    {"prediction": "VI-shingles", "confidence": 0.9},
                    {"prediction": "VI-chickenpox", "confidence": 0.1},
                ]),
                b"fungal": _prediction([
                    {"prediction": "VI-shingles", "confidence": 0.1},
                    {"prediction": "VI-chickenpox", "confidence": 0.9},
                ]),
            },
            "parasitic": {
                b"viral": _prediction([{"prediction": "PA-only-class", "confidence": 1.0}]),
                b"fungal": _prediction([{"prediction": "PA-only-class", "confidence": 1.0}]),
            },
            "fungal": {
                b"viral": _prediction([
                    {"prediction": "FU-ringworm", "confidence": 0.6},
                    {"prediction": "FU-nail-fungus", "confidence": 0.4},
                ]),
                b"fungal": _prediction([
                    {"prediction": "FU-ringworm", "confidence": 0.95},
                    {"prediction": "FU-nail-fungus", "confidence": 0.05},
                ]),
            },
            "bacterial": {
                b"viral": _prediction([
                    {"prediction": "BA-cellulitis", "confidence": 0.55},
                    {"prediction": "BA-impetigo", "confidence": 0.45},
                ]),
                b"fungal": _prediction([
                    {"prediction": "BA-cellulitis", "confidence": 0.55},
                    {"prediction": "BA-impetigo", "confidence": 0.45},
                ]),
            },
        }
        callbacks = {
            client: lambda image_bytes, model_path, topk, name=client: outputs[name][image_bytes]
            for client in CLIENT_PREDICTORS
        }
        predictor = MultiModelSpecialistPredictor()

        with patch.dict(CLIENT_PREDICTORS, callbacks), patch.object(
            predictor,
            "get_label_catalog",
            return_value={
                "viral": ["VI-shingles", "VI-chickenpox"],
                "parasitic": ["PA-only-class"],
                "fungal": ["FU-ringworm", "FU-nail-fungus"],
                "bacterial": ["BA-cellulitis", "BA-impetigo"],
            },
        ):
            viral_result = predictor.predict_from_bytes(b"viral", topk=20)
            fungal_result = predictor.predict_from_bytes(b"fungal", topk=20)

        self.assertEqual(viral_result["prediction"], "VI-shingles")
        self.assertEqual(fungal_result["prediction"], "FU-ringworm")
        self.assertNotIn("PA-only-class", [row["prediction"] for row in viral_result["topk"]])
        self.assertEqual(viral_result["model_predictions"]["parasitic"]["prediction"], None)
        self.assertIn("parasitic", viral_result["model_warnings"])
        self.assertEqual(viral_result["active_models"], ["viral", "fungal", "bacterial"])
        self.assertAlmostEqual(sum(row["confidence"] for row in viral_result["topk"]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
