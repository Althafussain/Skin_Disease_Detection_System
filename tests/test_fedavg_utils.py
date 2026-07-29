import unittest

from fedavg_utils import fedavg_label_probabilities


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
