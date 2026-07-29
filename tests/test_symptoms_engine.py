import tempfile
import unittest
from pathlib import Path

from symptoms_engine import SymptomRefinementEngine


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _build_synthetic_symptoms_dir(tmp_dir: str) -> Path:
    """Builds a small, deterministic set of symptom CSVs (synthetic test
    fixtures, not the real project dataset) so these tests stay stable and
    fast regardless of the real Symptoms/ data."""
    symptoms_dir = Path(tmp_dir)

    _write_csv(
        symptoms_dir / "BacterialDataset.csv",
        "fever,redness_severity,warmth_level,diagnosis,severity_level,treatment_recommendation",
        [
            "1,1,1,BA-Ery,Severe,Seek urgent care.",
            "0,0,0,Normal,Mild,No treatment needed.",
        ],
    )
    _write_csv(
        symptoms_dir / "fungal_skin_dataset.csv",
        "scaling,peeling,itching,diagnosis,severity_level,treatment_recommendation",
        [
            "1,1,1,Tinea,Moderate,Use antifungal cream.",
            "0,0,0,None,Mild,No treatment needed.",
        ],
    )
    _write_csv(
        symptoms_dir / "skin.csv",
        "redness,itching,spreading_area,diagnosis,severity_level,treatment_recommendation",
        [
            "1,1,1,Scabies,Moderate,See a doctor.",
            "1,1,0,Scabies,Moderate,See a doctor.",
            "0,0,0,NormalSkin,Mild,No treatment needed.",
        ],
    )
    _write_csv(
        symptoms_dir / "viral_infection_dataset.csv",
        "fever,rash_extent,itching,diagnosis,severity_level,treatment_recommendation",
        [
            "1,1,0,Chickenpox,Moderate,Rest and fluids.",
            "0,0,0,Healthy,Mild,No treatment needed.",
        ],
    )
    return symptoms_dir


class SymptomEngineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        symptoms_dir = _build_synthetic_symptoms_dir(self._tmp.name)
        self.engine = SymptomRefinementEngine(symptoms_dir)

    def tearDown(self):
        self._tmp.cleanup()


class ShouldShowSuggestionsTests(SymptomEngineTestCase):
    def test_suppressed_on_high_confidence_neutral(self):
        rows = [{"prediction": "NormalSkin", "confidence": 0.9}]
        self.assertFalse(self.engine.should_show_suggestions(rows))

    def test_shown_on_low_confidence_neutral(self):
        rows = [{"prediction": "NormalSkin", "confidence": 0.5}]
        self.assertTrue(self.engine.should_show_suggestions(rows))

    def test_shown_on_disease_prediction(self):
        rows = [{"prediction": "Scabies", "confidence": 0.8}]
        self.assertTrue(self.engine.should_show_suggestions(rows))

    def test_hidden_on_empty_topk(self):
        self.assertFalse(self.engine.should_show_suggestions([]))


class SuggestSymptomsTests(SymptomEngineTestCase):
    def test_returns_ranked_list_within_limit(self):
        rows = [{"prediction": "Scabies", "confidence": 0.8}]
        suggestions = self.engine.suggest_symptoms(rows, limit=2)
        self.assertLessEqual(len(suggestions), 2)
        for item in suggestions:
            self.assertIn("name", item)
            self.assertIn("label", item)
            self.assertIn("score", item)

    def test_empty_when_suppressed_by_neutral_result(self):
        rows = [{"prediction": "Normal", "confidence": 0.95}]
        self.assertEqual(self.engine.suggest_symptoms(rows), [])


class ResolveSymptomsTests(SymptomEngineTestCase):
    def test_resolves_known_alias_to_a_registered_symptom(self):
        resolved = self.engine.resolve_symptoms(["itchy"])
        self.assertTrue(resolved)
        self.assertTrue({"itching", "itching_level"} & set(resolved))

    def test_unknown_term_resolves_to_empty(self):
        self.assertEqual(self.engine.resolve_symptoms(["zzz_not_a_symptom_zzz"]), [])


class FinalPredictTests(SymptomEngineTestCase):
    def test_raises_without_any_symptoms_or_keywords(self):
        with self.assertRaises(ValueError):
            self.engine.final_predict(selected_symptoms=[], image_topk=[])

    def test_returns_diagnosis_with_normalized_confidence(self):
        result = self.engine.final_predict(
            selected_symptoms=["itching", "redness"],
            image_topk=[{"prediction": "Scabies", "confidence": 0.6}],
            limit=5,
        )
        self.assertIn("final_prediction", result)
        self.assertIn("top_diagnoses", result)
        self.assertIn("medical_notice", result)
        total_confidence = sum(row["confidence"] for row in result["top_diagnoses"])
        self.assertAlmostEqual(total_confidence, 1.0, places=2)

    def test_keyword_only_input_still_produces_a_prediction(self):
        result = self.engine.final_predict(
            selected_symptoms=[],
            image_topk=[],
            custom_keywords=["scabies", "burrow"],
        )
        self.assertEqual(result["final_prediction"]["diagnosis"], "Scabies")


class CanonicalDiagnosisTests(SymptomEngineTestCase):
    def test_maps_known_raw_labels_to_canonical_names(self):
        self.assertEqual(self.engine._canonical_diagnosis("healthy"), "Normal")
        self.assertEqual(self.engine._canonical_diagnosis("BA-Ery"), "BA-Ery")
        self.assertEqual(self.engine._canonical_diagnosis("ringworm"), "Ringworm")
        self.assertEqual(self.engine._canonical_diagnosis("scabies"), "Scabies")


class SymptomCatalogTests(SymptomEngineTestCase):
    def test_catalog_includes_symptoms_from_loaded_csvs(self):
        names = {item["name"] for item in self.engine.symptom_catalog()}
        self.assertTrue({"redness", "itching"} & names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
