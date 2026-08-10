#!/usr/bin/env python3
"""
Organize new datasets from DS folder into Train/Test splits and retrain all models.
Targets 80% accuracy.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

DS_ROOT = Path("DS")
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Disease categories mapping
DISEASE_CONFIGS = {
    "Bacterial": {
        "source": DS_ROOT / "Bacterial",
        "target_base": Path("Bacterial_Skin_Disease") / "Bacterial-Dataset",
        "classes": ["BA- cellulitis", "BA-impetigo"],
    },
    "Fungal": {
        "source": DS_ROOT / "Fungal",
        "target_base": Path("fungal") / "Fungal-Dataset",
        "classes": ["FU-athlete-foot", "FU-nail-fungus", "FU-ringworm"],
    },
    "Parasitic": {
        "source": DS_ROOT / "Paracitic",
        "target_base": Path("Parastic_Skin_Disease") / "Parastic-Dataset",
        "classes": ["PA-cutaneous-larva-migrans"],
    },
    "Viral": {
        "source": DS_ROOT / "Viral",
        "target_base": Path("Viral") / "Viral-Dataset",
        "classes": ["VI-chickenpox", "VI-shingles"],
    },
}


def organize_dataset(disease_name, config):
    """Organize dataset into Train/Test splits."""
    source_root = config["source"]
    target_base = config["target_base"]
    classes = config["classes"]

    train_dir = target_base / "Train"
    test_dir = target_base / "Test"

    # Clean existing
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_test = 0

    for class_name in classes:
        # Find class directory (case-insensitive)
        class_dir = None
        for d in source_root.iterdir():
            if d.is_dir() and d.name.lower() == class_name.lower():
                class_dir = d
                break

        if not class_dir:
            print(f"  Warning: Class directory not found for {class_name} in {source_root}")
            continue

        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = [f for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

        if not images:
            print(f"  Warning: No images found for {class_name}")
            continue

        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create class directories
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in test_images:
            shutil.copy2(img, test_dir / class_name / img.name)

        print(f"  {class_name}: {len(train_images)} train, {len(test_images)} test")
        total_train += len(train_images)
        total_test += len(test_images)

    print(f"  Total: {total_train} train, {total_test} test images")
    return total_train > 0 and total_test > 0


def main():
    print("=" * 60)
    print("Organizing datasets from DS folder...")
    print("=" * 60)

    all_success = True
    for disease_name, config in DISEASE_CONFIGS.items():
        print(f"\n{disease_name}:")
        success = organize_dataset(disease_name, config)
        if not success:
            print(f"  ERROR: Failed to organize {disease_name}")
            all_success = False

    if not all_success:
        print("\nSome datasets failed to organize. Check warnings above.")
        return

    print("\n" + "=" * 60)
    print("Dataset organization complete!")
    print("=" * 60)

    # Now retrain models
    print("\nStarting model retraining...")
    print("=" * 60)

    import subprocess
    import sys

    training_scripts = [
        ("Bacterial", "Bacterial_Skin_Disease/client_train.py"),
        ("Fungal", "fungal/client_train.py"),
        ("Viral", "Viral/client_train.py"),
        ("Parasitic", "Parastic_Skin_Disease/client_train.py"),
    ]

    base_dir = Path.cwd()
    for disease_name, script_rel_path in training_scripts:
        print(f"\n{'='*60}")
        print(f"Training {disease_name} model...")
        print(f"{'='*60}")

        script_path = base_dir / script_rel_path
        script_dir = script_path.parent
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.returncode != 0:
                print(f"WARNING: {disease_name} training returned code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"ERROR: {disease_name} training timed out")
        except Exception as e:
            print(f"ERROR: {disease_name} training failed: {e}")

    print("\n" + "=" * 60)
    print("All training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()