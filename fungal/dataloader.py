from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_fungal_data(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "Fungal-Dataset"
    train_path = dataset_dir / "Train"
    test_path = dataset_dir / "Test"

    if not train_path.exists():
        raise FileNotFoundError(f"Train folder not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test folder not found: {test_path}")

    train_ds = datasets.ImageFolder(str(train_path), transform)
    test_ds = datasets.ImageFolder(str(test_path), transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        len(train_ds.classes),
    )
