import unittest

import torch
import torch.nn as nn

from fedavg_utils import fedavg_compatible_state_dicts, fedavg_label_probabilities


class FedAvgUtilsTests(unittest.TestCase):
    def test_label_probabilities_equal_weights(self):
        result = fedavg_label_probabilities(
            {
                "viral": {"Healthy": 0.7, "None": 0.3},
                "fungal": {"FU-ringworm": 0.6, "None": 0.4},
            },
            client_weights=None,
            normalize_output=True,
        )
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertGreater(result["None"], 0.34)

    def test_label_probabilities_custom_weights(self):
        result = fedavg_label_probabilities(
            {
                "viral": {"VI-shingles": 0.9, "None": 0.1},
                "parasitic": {"Scabies": 0.6, "None": 0.4},
            },
            client_weights={"viral": 3.0, "parasitic": 1.0},
            normalize_output=True,
        )
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertEqual(next(iter(result.keys())), "VI-shingles")


class FedAvgWeightAveragingTests(unittest.TestCase):
    """Covers the server-side aggregation step of the federated learning
    workflow: averaging locally-trained client weights into a global model."""

    def test_averages_shared_shape_tensors_across_all_clients(self):
        client_state_dicts = {
            "client_a": {"backbone.weight": torch.tensor([2.0, 4.0])},
            "client_b": {"backbone.weight": torch.tensor([4.0, 8.0])},
        }
        averaged, stats = fedavg_compatible_state_dicts(client_state_dicts)
        self.assertTrue(torch.allclose(averaged["backbone.weight"], torch.tensor([3.0, 6.0])))
        self.assertEqual(stats["num_clients"], 2)
        self.assertEqual(stats["averaged_keys"], 1)

    def test_skips_incompatible_classifier_head_shapes(self):
        client_state_dicts = {
            "bacterial": {
                "backbone.weight": torch.tensor([1.0, 1.0]),
                "classifier.weight": torch.zeros(5, 10),  # 5-class head
            },
            "viral": {
                "backbone.weight": torch.tensor([3.0, 3.0]),
                "classifier.weight": torch.zeros(4, 10),  # 4-class head, incompatible
            },
        }
        averaged, stats = fedavg_compatible_state_dicts(client_state_dicts, require_all_clients=True)
        self.assertIn("backbone.weight", averaged)
        self.assertNotIn("classifier.weight", averaged)
        self.assertIn("classifier.weight", stats["skipped_reasons"])
        self.assertTrue(torch.allclose(averaged["backbone.weight"], torch.tensor([2.0, 2.0])))

    def test_respects_custom_client_weights(self):
        client_state_dicts = {
            "a": {"w": torch.tensor([0.0])},
            "b": {"w": torch.tensor([10.0])},
        }
        averaged, _ = fedavg_compatible_state_dicts(
            client_state_dicts, client_weights={"a": 3.0, "b": 1.0}
        )
        self.assertTrue(torch.allclose(averaged["w"], torch.tensor([2.5])))

    def test_raises_on_empty_client_state_dicts(self):
        with self.assertRaises(ValueError):
            fedavg_compatible_state_dicts({})


class FederatedLearningWorkflowTests(unittest.TestCase):
    """End-to-end simulation of the FL round used by this project: several
    clients hold local models with a shared backbone but different classifier
    head sizes (matching the 4 real disease components), the server
    aggregates compatible weights, and the resulting global state dict is
    redistributed into a fresh model for a working forward pass."""

    class _TinyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, num_classes)

        def forward(self, x):
            return self.classifier(self.backbone(x))

    def test_aggregated_global_weights_load_and_run_on_a_fresh_model(self):
        client_models = {
            "bacterial": self._TinyModel(num_classes=5),
            "viral": self._TinyModel(num_classes=4),
            "fungal": self._TinyModel(num_classes=4),
        }
        client_state_dicts = {name: model.state_dict() for name, model in client_models.items()}

        averaged, stats = fedavg_compatible_state_dicts(client_state_dicts, require_all_clients=True)

        # Shared backbone (identical shape across all clients) is aggregated.
        self.assertIn("backbone.weight", averaged)
        self.assertIn("backbone.bias", averaged)
        # Classifier heads differ in shape (5 vs 4 output classes) and must be skipped.
        self.assertNotIn("classifier.weight", averaged)
        self.assertEqual(stats["num_clients"], 3)

        # Redistribution step: the aggregated global weights must load into a
        # fresh client model and still produce a valid forward pass.
        global_model = self._TinyModel(num_classes=4)
        global_model.load_state_dict(averaged, strict=False)
        output = global_model(torch.randn(2, 4))
        self.assertEqual(tuple(output.shape), (2, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
