#!/usr/bin/env python3
"""
Prepare text training CSV for SymptomRefinementEngine without changing app logic.

The engine reads:
  Symptoms/skin_text_classifier_filtered.csv
with columns:
  Disease name,Text
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = [
    Path("symptomsTexts/skin_disease_text_train_dataset_9000.csv"),
]
DEFAULT_OUTPUT = Path("Symptoms/skin_text_classifier_filtered.csv")


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...], fallback: str) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return fallback


def _clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_label(label: str) -> str:
    label = str(label or "").strip()
    return re.sub(r"\s+", " ", label)


def build_rows(input_paths: list[Path]) -> tuple[list[dict[str, str]], Counter]:
    rows: list[dict[str, str]] = []
    class_counts: Counter = Counter()
    seen: set[tuple[str, str]] = set()

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue

            disease_col = _pick_column(
                list(reader.fieldnames),
                ("Disease name", "label", "disease", "diagnosis"),
                "Disease name",
            )
            text_col = _pick_column(
                list(reader.fieldnames),
                ("Text", "text", "sentence", "description"),
                "Text",
            )

            for raw in reader:
                disease = _normalize_label(raw.get(disease_col, ""))
                text = _clean_text(raw.get(text_col, ""))
                if not disease or not text:
                    continue
                if len(text) < 8:
                    continue

                dedupe_key = (disease.lower(), text.lower())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                rows.append({"Disease name": disease, "Text": text})
                class_counts[disease] += 1

    return rows, class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build keyword training CSV for symptoms engine.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUT],
        help="One or more source CSVs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path in engine format.",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input]
    output_path = Path(args.output)

    rows, class_counts = build_rows(input_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Disease name", "Text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {output_path}")
    print(f"Unique classes: {len(class_counts)}")
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
