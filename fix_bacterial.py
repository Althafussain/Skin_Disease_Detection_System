import shutil
from pathlib import Path
import random

random.seed(42)

source_root = Path("DS/Bacterial")
target_base = Path("Bacterial_Skin_Disease") / "Bacterial-Dataset"
classes = ["BA- cellulitis", "BA-impetigo"]

train_dir = Path("Bacterial_Skin_Disease") / "Bacterial-Dataset" / "Train"
test_dir = Path("Bacterial_Skin_Disease") / "Bacterial-Dataset" / "Test"

if train_dir.exists():
    shutil.rmtree(train_dir)
if test_dir.exists():
    shutil.rmtree(test_dir)

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

for class_name in classes:
    class_dir = source_root / class_name
    if not class_dir.exists():
        print(f"Class directory not found: {class_dir}")
        continue

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = [f for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    (test_dir / class_name).mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.copy2(img, train_dir / class_name / img.name)
    for img in test_images:
        shutil.copy2(img, test_dir / class_name / img.name)

    print(f"  {class_name}: {len(train_images)} train, {len(test_images)} test")