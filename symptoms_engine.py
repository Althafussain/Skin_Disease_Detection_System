from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


META_COLUMNS = {"diagnosis", "severity_level", "treatment_recommendation"}
TEXT_DATASET_FILES = ("skin_text_classifier_filtered.csv", "Skin_text_classifier 3.csv")

NEUTRAL_LABELS = {
    "normal",
    "normal skin",
    "normalskin",
    "healthy",
    "none",
    "skin not found",
}

MANUAL_DIAGNOSIS_SYMPTOM_BOOSTS: Dict[str, Dict[str, float]] = {
    "Scabies": {
        "itching": 1.0,
        "itching_level": 1.0,
        "linear_tracks": 0.95,
        "redness": 0.8,
        "bumps": 0.85,
        "secondary_infection": 0.55,
        "spreading_area": 0.55,
    },
    "Tinea": {
        "scaling": 1.0,
        "peeling": 0.95,
        "cracking": 0.8,
        "itching": 0.9,
        "itching_level": 0.9,
        "redness": 0.75,
        "spreading_area": 0.75,
        "odor": 0.35,
    },
    "BA-Ery": {
        "fever": 0.8,
        "redness_severity": 1.0,
        "warmth_level": 0.95,
        "swelling_severity": 0.88,
        "pain_level": 0.85,
        "spreading_area": 0.72,
    },
}

MANUAL_DIAGNOSIS_DEFAULTS = {
    "Scabies": {
        "severity_level": "Moderate",
        "treatment_recommendation": "Seek clinical confirmation; treat close contacts and wash bedding/clothes.",
    },
    "Tinea": {
        "severity_level": "Moderate",
        "treatment_recommendation": "Use topical antifungal therapy and keep affected skin dry and clean.",
    },
    "BA-Ery": {
        "severity_level": "Moderate",
        "treatment_recommendation": "Prompt clinical review is recommended for possible bacterial skin infection.",
    },
}

KEYWORD_SYMPTOM_ALIASES = {
    "itch": ("itching", "itching_level"),
    "itchy": ("itching", "itching_level"),
    "night itch": ("itching", "itching_level"),
    "red": ("redness", "redness_severity"),
    "rash": ("redness", "spreading_area"),
    "blister": ("blisters", "blisters_count"),
    "crust": ("crust", "honey_crust_presence"),
    "honey crust": ("honey_crust_presence", "crust"),
    "warm": ("warmth_level",),
    "swelling": ("swelling_severity",),
    "pain": ("pain", "pain_level"),
    "fever": ("fever",),
    "fatigue": ("fatigue",),
    "scaly": ("scaling",),
    "scaling": ("scaling",),
    "peel": ("peeling",),
    "peeling": ("peeling",),
    "crack": ("cracking",),
    "cracked": ("cracking",),
    "burrow": ("linear_tracks",),
    "track": ("linear_tracks",),
    "ring": ("scaling", "spreading_area"),
    "toes": ("peeling", "cracking"),
    "toe": ("peeling", "cracking"),
    "spread": ("spreading_area",),
    "bump": ("bumps",),
    "odor": ("odor",),
    "pus": ("blisters_count", "secondary_infection"),
}

MANUAL_KEYWORD_DIAGNOSIS_HINTS = {
    "scabies": {"Scabies": 1.0},
    "mite": {"Scabies": 0.8},
    "burrow": {"Scabies": 0.9},
    "night itch": {"Scabies": 0.85},
    "tinea": {"Tinea": 1.0},
    "ringworm": {"Tinea": 0.9},
    "athlete foot": {"Tinea": 0.9},
    "toes": {"Tinea": 0.5},
    "fungal": {"Tinea": 0.45},
    "ba ery": {"BA-Ery": 1.0},
    "erysipelas": {"BA-Ery": 0.95},
    "warmth": {"BA-Ery": 0.7},
    "spreading redness": {"BA-Ery": 0.9},
    "cellulitis": {"BA-Ery": 0.65},
}

TEXT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "have",
    "has",
    "from",
    "they",
    "them",
    "been",
    "just",
    "very",
    "into",
    "your",
    "their",
    "what",
    "when",
    "where",
    "over",
    "after",
    "before",
    "also",
    "could",
    "would",
    "should",
    "about",
    "more",
    "some",
    "like",
    "really",
    "these",
    "those",
    "around",
    "across",
    "through",
    "while",
    "because",
    "there",
    "been",
    "than",
}

KNOWN_DIAGNOSES = {
    "normal": "Normal",
    "healthy": "Normal",
    "none": "Normal",
    "normal skin": "Normal",
    "normalskin": "Normal",
    "skin not found": "Normal",
    "impetigo": "Impetigo",
    "ba impetigo": "Impetigo",
    "cellulitis": "Cellulitis",
    "ba cellulitis": "Cellulitis",
    "ery": "BA-Ery",
    "erysipelas": "BA-Ery",
    "ba ery": "BA-Ery",
    "ba erysipelas": "BA-Ery",
    "athlete foot": "Athlete Foot",
    "nail fungus": "Nail Fungus",
    "ringworm": "Ringworm",
    "tinea": "Tinea",
    "chickenpox": "Chickenpox",
    "shingles": "Shingles",
    "cutaneous larva migrans": "Cutaneous Larva Migrans",
    "scabies": "Scabies",
}

DATASET_PATHS = {
    "bacterial": "BacterialDataset.csv",
    "fungal": "fungal_skin_dataset.csv",
    "parasitic": "skin.csv",
    "viral": "viral_infection_dataset.csv",
}


def _normalize_text(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", compact)


def _to_snake_case(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def _humanize_symptom(symptom: str) -> str:
    return symptom.replace("_", " ").strip().title()


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _split_terms(values: Iterable[str]) -> List[str]:
    terms: List[str] = []
    for value in values:
        if not value:
            continue
        for segment in re.split(r"[,\n;/]+", str(value)):
            normalized = _normalize_text(segment)
            if normalized:
                terms.append(normalized)
    return terms


@dataclass
class DiagnosisProfile:
    diagnosis: str
    count: int = 0
    symptom_sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    symptom_positive: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    severity_counter: Counter = field(default_factory=Counter)
    treatment_counter: Counter = field(default_factory=Counter)

    def prevalence(self, symptom: str) -> float:
        if self.count == 0:
            return 0.0
        return self.symptom_positive.get(symptom, 0) / self.count

    def mean(self, symptom: str) -> float:
        if self.count == 0:
            return 0.0
        return self.symptom_sums.get(symptom, 0.0) / self.count

    def top_severity(self) -> str:
        if not self.severity_counter:
            return "Unknown"
        return self.severity_counter.most_common(1)[0][0]

    def top_treatment(self) -> str:
        if not self.treatment_counter:
            return "Consult a qualified clinician for next steps."
        return self.treatment_counter.most_common(1)[0][0]


class SymptomRefinementEngine:
    def __init__(self, symptoms_dir: Path) -> None:
        self.symptoms_dir = symptoms_dir
        self.profiles: Dict[str, DiagnosisProfile] = {}
        self.symptom_max: Dict[str, float] = defaultdict(float)
        self.global_positive: Dict[str, int] = defaultdict(int)
        self.global_rows: int = 0
        self.symptom_lookup: Dict[str, str] = {}
        self.keyword_diag_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._load_all()

    def is_neutral_label(self, label: str) -> bool:
        return _normalize_text(label) in NEUTRAL_LABELS

    def should_show_suggestions(
        self,
        image_topk: Iterable[Mapping[str, Any]],
        neutral_confidence_threshold: float = 0.8,
    ) -> bool:
        rows = list(image_topk)
        if not rows:
            return False
        for row in rows[:5]:
            if not self.is_neutral_label(str(row.get("prediction", ""))):
                continue
            confidence = self._normalize_confidence(_safe_float(row.get("confidence", 0.0)))
            if confidence >= neutral_confidence_threshold:
                return False
        return True

    def _canonical_diagnosis(self, diagnosis: str) -> str:
        normalized = _normalize_text(diagnosis)
        if normalized in KNOWN_DIAGNOSES:
            return KNOWN_DIAGNOSES[normalized]

        for prefix in ("ba", "fu", "vi", "pa"):
            if normalized.startswith(f"{prefix} "):
                trimmed = normalized[len(prefix) + 1 :]
                if trimmed in KNOWN_DIAGNOSES:
                    return KNOWN_DIAGNOSES[trimmed]
                normalized = trimmed
                break

        if "shingles" in normalized or "herpes zoster" in normalized:
            return "Shingles"
        if "chickenpox" in normalized:
            return "Chickenpox"
        if "impetigo" in normalized:
            return "Impetigo"
        if "ery" in normalized or "erysipelas" in normalized:
            return "BA-Ery"
        if "cellulitis" in normalized:
            return "Cellulitis"
        if "ringworm" in normalized:
            return "Ringworm"
        if "athlete" in normalized and "foot" in normalized:
            return "Athlete Foot"
        if "tinea" in normalized:
            return "Tinea"
        if "scabies" in normalized:
            return "Scabies"
        if "nail" in normalized and "fung" in normalized:
            return "Nail Fungus"
        if "cutaneous" in normalized and "larva" in normalized:
            return "Cutaneous Larva Migrans"
        if self.is_neutral_label(normalized):
            return "Normal"
        return diagnosis.replace("_", " ").replace("-", " ").strip().title()

    def _register_symptom_alias(self, alias: str, symptom_name: str) -> None:
        normalized_alias = _normalize_text(alias)
        if not normalized_alias:
            return
        self.symptom_lookup[normalized_alias] = symptom_name
        self.symptom_lookup[_to_snake_case(normalized_alias)] = symptom_name

    def _register_symptom_name(self, symptom_name: str) -> None:
        self._register_symptom_alias(symptom_name, symptom_name)
        self._register_symptom_alias(_humanize_symptom(symptom_name), symptom_name)

    def _register_builtin_aliases(self) -> None:
        for keyword, mapped_symptoms in KEYWORD_SYMPTOM_ALIASES.items():
            for symptom in mapped_symptoms:
                self._register_symptom_alias(keyword, symptom)

    def _load_structured_symptom_datasets(self) -> None:
        for filename in DATASET_PATHS.values():
            dataset_path = self.symptoms_dir / filename
            if not dataset_path.exists():
                continue

            with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue

                raw_to_snake = {
                    raw: _to_snake_case(raw) for raw in reader.fieldnames if raw is not None
                }
                symptom_cols = [
                    snake
                    for snake in raw_to_snake.values()
                    if snake and snake not in META_COLUMNS
                ]

                for symptom in symptom_cols:
                    self._register_symptom_name(symptom)

                for row in reader:
                    diagnosis = self._canonical_diagnosis(str(row.get("diagnosis", "")))
                    if not diagnosis:
                        continue

                    profile = self.profiles.setdefault(diagnosis, DiagnosisProfile(diagnosis=diagnosis))
                    profile.count += 1
                    self.global_rows += 1

                    severity = str(row.get("severity_level", "")).strip()
                    treatment = str(row.get("treatment_recommendation", "")).strip()
                    if severity:
                        profile.severity_counter[severity] += 1
                    if treatment:
                        profile.treatment_counter[treatment] += 1

                    for raw_col, snake_col in raw_to_snake.items():
                        if not snake_col or snake_col in META_COLUMNS:
                            continue
                        value = _safe_float(row.get(raw_col))
                        profile.symptom_sums[snake_col] += value
                        self.symptom_max[snake_col] = max(self.symptom_max[snake_col], value)
                        if value > 0:
                            profile.symptom_positive[snake_col] += 1
                            self.global_positive[snake_col] += 1

    def _load_text_classifier_keywords(self) -> None:
        token_counts_by_diag: Dict[str, Counter] = defaultdict(Counter)

        text_path = None
        for filename in TEXT_DATASET_FILES:
            candidate = self.symptoms_dir / filename
            if candidate.exists():
                text_path = candidate
                break
        if text_path is None:
            return

        with text_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return

            disease_col = next((c for c in reader.fieldnames if _normalize_text(c) == "disease name"), "Disease name")
            text_col = next((c for c in reader.fieldnames if _normalize_text(c) == "text"), "Text")

            for row in reader:
                disease_name = str(row.get(disease_col, "")).strip()
                narrative = str(row.get(text_col, "")).strip()
                if not disease_name or not narrative:
                    continue

                diagnosis = self._canonical_diagnosis(disease_name)
                if diagnosis == "Normal":
                    continue

                text = _normalize_text(narrative)
                words = [w for w in re.findall(r"[a-z]{3,}", text) if w not in TEXT_STOPWORDS]
                if not words:
                    continue
                token_counts_by_diag[diagnosis].update(words)
                token_counts_by_diag[diagnosis].update(f"{w1} {w2}" for w1, w2 in zip(words, words[1:]))

        token_diag: Dict[str, Counter] = defaultdict(Counter)
        for diagnosis, counts in token_counts_by_diag.items():
            for token, count in counts.items():
                if count < 2:
                    continue
                token_diag[token][diagnosis] += count

        for token, diag_counter in token_diag.items():
            total = sum(diag_counter.values())
            if total <= 0:
                continue
            self.keyword_diag_scores[token] = {
                diagnosis: value / total
                for diagnosis, value in diag_counter.items()
            }

    def _load_all(self) -> None:
        self._load_structured_symptom_datasets()
        self._register_builtin_aliases()
        self._load_text_classifier_keywords()

        # Ensure manual boost symptoms are available in the catalog and lookup even if absent from CSVs.
        for boost_map in MANUAL_DIAGNOSIS_SYMPTOM_BOOSTS.values():
            for symptom, score in boost_map.items():
                self._register_symptom_name(symptom)
                self.symptom_max[symptom] = max(self.symptom_max.get(symptom, 0.0), score)

    def _normalize_confidence(self, confidence: float) -> float:
        if confidence > 1.0:
            return confidence / 100.0
        return max(0.0, min(confidence, 1.0))

    def _prior_from_topk(
        self,
        image_topk: Iterable[Mapping[str, Any]],
        limit: int | None = None,
        drop_neutral: bool = False,
    ) -> Dict[str, float]:
        rows = list(image_topk)
        if limit is not None and limit > 0:
            rows = rows[:limit]

        priors: Dict[str, float] = defaultdict(float)
        for row in rows:
            label = str(row.get("prediction", "")).strip()
            if not label:
                continue
            diagnosis = self._canonical_diagnosis(label)
            if drop_neutral and diagnosis == "Normal":
                continue
            confidence = self._normalize_confidence(_safe_float(row.get("confidence", 0.0)))
            if confidence <= 0:
                continue
            priors[diagnosis] += confidence

        total = sum(priors.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in priors.items()}

    def _diagnosis_prior_from_keywords(self, custom_keywords: Iterable[str]) -> Dict[str, float]:
        terms = _split_terms(custom_keywords)
        if not terms:
            return {}

        accum: Dict[str, float] = defaultdict(float)
        for term in terms:
            for hint, diag_scores in MANUAL_KEYWORD_DIAGNOSIS_HINTS.items():
                if hint in term:
                    for diagnosis, weight in diag_scores.items():
                        accum[diagnosis] += weight

            if term in self.keyword_diag_scores:
                for diagnosis, weight in self.keyword_diag_scores[term].items():
                    accum[diagnosis] += weight
                continue

            tokens = [tok for tok in term.split() if tok]
            for token in tokens:
                if token in self.keyword_diag_scores:
                    for diagnosis, weight in self.keyword_diag_scores[token].items():
                        accum[diagnosis] += 0.7 * weight

        total = sum(accum.values())
        if total <= 0:
            return {}
        return {diagnosis: value / total for diagnosis, value in accum.items()}

    def resolve_symptoms(self, symptoms: Iterable[str]) -> List[str]:
        resolved: List[str] = []
        terms = _split_terms(symptoms)
        for term in terms:
            snake = _to_snake_case(term)
            key = (
                self.symptom_lookup.get(term)
                or self.symptom_lookup.get(snake)
                or self.symptom_lookup.get(_normalize_text(term))
            )
            if key and key not in resolved:
                resolved.append(key)
                continue

            # Phrase-level alias fallback.
            for alias, mapped in KEYWORD_SYMPTOM_ALIASES.items():
                if alias in term:
                    for symptom in mapped:
                        if symptom not in resolved:
                            resolved.append(symptom)
        return resolved

    def symptom_catalog(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        all_symptoms = sorted({key for key in self.symptom_max.keys() if key not in META_COLUMNS})
        if not all_symptoms:
            return items

        denominator = self.global_rows if self.global_rows > 0 else 1
        for symptom in all_symptoms:
            prevalence = self.global_positive.get(symptom, 0) / denominator
            items.append(
                {
                    "name": symptom,
                    "label": _humanize_symptom(symptom),
                    "global_prevalence": round(float(prevalence), 4),
                }
            )
        return items

    def _symptom_score_for_diagnosis(self, diagnosis: str, symptom: str) -> float:
        score = 0.0
        profile = self.profiles.get(diagnosis)
        if profile is not None:
            max_val = self.symptom_max.get(symptom, 1.0) or 1.0
            prevalence = profile.prevalence(symptom)
            intensity = profile.mean(symptom) / max_val
            score = 0.7 * prevalence + 0.3 * intensity
        score = max(score, MANUAL_DIAGNOSIS_SYMPTOM_BOOSTS.get(diagnosis, {}).get(symptom, 0.0))
        return min(max(score, 0.0), 1.0)

    def suggest_symptoms(
        self,
        image_topk: Iterable[Mapping[str, Any]],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if not self.should_show_suggestions(image_topk):
            return []

        priors = self._prior_from_topk(image_topk, limit=3, drop_neutral=True)
        scores: Dict[str, float] = defaultdict(float)

        if priors:
            for diagnosis, prior in priors.items():
                profile = self.profiles.get(diagnosis)
                if profile is not None:
                    for symptom in profile.symptom_sums.keys():
                        scores[symptom] += prior * self._symptom_score_for_diagnosis(diagnosis, symptom)
                for symptom, boost in MANUAL_DIAGNOSIS_SYMPTOM_BOOSTS.get(diagnosis, {}).items():
                    scores[symptom] += prior * boost
        else:
            # Fallback: globally frequent symptoms.
            denominator = self.global_rows if self.global_rows > 0 else 1
            for symptom, count in self.global_positive.items():
                scores[symptom] = count / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        suggestions = []
        for symptom, score in ranked[:limit]:
            suggestions.append(
                {
                    "name": symptom,
                    "label": _humanize_symptom(symptom),
                    "score": round(float(score), 4),
                }
            )
        return suggestions

    def _default_severity(self, diagnosis: str) -> str:
        if diagnosis in MANUAL_DIAGNOSIS_DEFAULTS:
            return MANUAL_DIAGNOSIS_DEFAULTS[diagnosis]["severity_level"]
        profile = self.profiles.get(diagnosis)
        if profile is None:
            return "Unknown"
        return profile.top_severity()

    def _default_treatment(self, diagnosis: str) -> str:
        if diagnosis in MANUAL_DIAGNOSIS_DEFAULTS:
            return MANUAL_DIAGNOSIS_DEFAULTS[diagnosis]["treatment_recommendation"]
        profile = self.profiles.get(diagnosis)
        if profile is None:
            return "Consult a qualified clinician for next steps."
        return profile.top_treatment()

    def final_predict(
        self,
        selected_symptoms: Iterable[str],
        image_topk: Iterable[Mapping[str, Any]],
        custom_keywords: Iterable[str] | None = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        keyword_terms = list(custom_keywords or [])
        resolved_selected = self.resolve_symptoms(selected_symptoms)
        resolved_from_keywords = self.resolve_symptoms(keyword_terms)
        resolved = list(dict.fromkeys(resolved_selected + resolved_from_keywords))
        keyword_diag_priors = self._diagnosis_prior_from_keywords(keyword_terms)

        if not resolved and not keyword_diag_priors:
            raise ValueError("Select symptoms or enter symptom keywords before final prediction")

        priors = self._prior_from_topk(image_topk, limit=5, drop_neutral=False)
        diagnosis_candidates = set(self.profiles.keys())
        diagnosis_candidates.update(priors.keys())
        diagnosis_candidates.update(keyword_diag_priors.keys())
        diagnosis_candidates.update(MANUAL_DIAGNOSIS_SYMPTOM_BOOSTS.keys())

        diagnosis_rows: List[Dict[str, Any]] = []
        for diagnosis in diagnosis_candidates:
            symptom_scores: List[float] = []
            matched_symptoms: List[str] = []
            for symptom in resolved:
                score = self._symptom_score_for_diagnosis(diagnosis, symptom)
                if score > 0:
                    symptom_scores.append(score)
                if score >= 0.25 and symptom not in matched_symptoms:
                    matched_symptoms.append(symptom)

            symptom_match = (sum(symptom_scores) / len(symptom_scores)) if symptom_scores else 0.0
            image_prior = priors.get(diagnosis, 0.0)
            keyword_prior = keyword_diag_priors.get(diagnosis, 0.0)

            if resolved:
                combined = 0.65 * symptom_match + 0.2 * image_prior + 0.15 * keyword_prior
            else:
                combined = 0.55 * image_prior + 0.45 * keyword_prior

            if diagnosis == "Normal" and (resolved or keyword_diag_priors):
                combined *= 0.7

            if combined <= 0:
                continue

            diagnosis_rows.append(
                {
                    "diagnosis": diagnosis,
                    "symptom_match": round(float(symptom_match), 4),
                    "image_prior": round(float(image_prior), 4),
                    "keyword_prior": round(float(keyword_prior), 4),
                    "combined_score": float(combined),
                    "severity_level": self._default_severity(diagnosis),
                    "treatment_recommendation": self._default_treatment(diagnosis),
                    "matched_symptoms": [
                        {"name": symptom, "label": _humanize_symptom(symptom)}
                        for symptom in matched_symptoms
                    ],
                }
            )

        if not diagnosis_rows:
            raise ValueError("Unable to compute final prediction from the provided symptoms/keywords")

        diagnosis_rows.sort(key=lambda row: row["combined_score"], reverse=True)
        top_rows = diagnosis_rows[: max(1, limit)]

        total = sum(row["combined_score"] for row in top_rows) or 1.0
        for row in top_rows:
            row["confidence"] = round(row["combined_score"] / total, 4)

        best = dict(top_rows[0])
        return {
            "selected_symptoms": [
                {"name": symptom, "label": _humanize_symptom(symptom)}
                for symptom in resolved_selected
            ],
            "keyword_symptoms": [
                {"name": symptom, "label": _humanize_symptom(symptom)}
                for symptom in resolved_from_keywords
            ],
            "custom_keywords": [term for term in _split_terms(keyword_terms)],
            "final_prediction": best,
            "top_diagnoses": top_rows,
            "medical_notice": "This is decision-support output, not medical advice.",
        }
