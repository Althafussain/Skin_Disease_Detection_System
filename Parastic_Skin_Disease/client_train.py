from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .dataloader import load_parasitic_data
    from .model_builder import build_parasitic_model
    from .utils import get_params, set_params, set_seed
except ImportError:
    from dataloader import load_parasitic_data
    from model_builder import build_parasitic_model
    from utils import get_params, set_params, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 3e-4
LOCAL_EPOCHS = 10
BATCH_SIZE = 16
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "parasitic_local.pth"
MODEL_ARCHITECTURE = "mobilenet_v2"


def client_update(
    global_params=None,
    train_from_scratch=True,
    local_epochs=LOCAL_EPOCHS,
    batch_size=BATCH_SIZE,
    checkpoint_path=None,
    seed=42,
):
    set_seed(seed)

    try:
        train_loader, _, num_classes, local_size = load_parasitic_data(batch_size=batch_size)
    except FileNotFoundError as exc:
        print(f"[Parasitic] Dataset not found: {exc}")
        return None, 0

    if len(train_loader) == 0:
        print("[Parasitic] Train loader is empty. Skipping client update.")
        return None, 0

    classes = list(getattr(train_loader.dataset, "classes", []))
    class_to_idx = dict(getattr(train_loader.dataset, "class_to_idx", {}))

    print(f"[Parasitic] Loaded dataset with {num_classes} classes (local size: {local_size}).")
    model = build_parasitic_model(num_classes=num_classes, train_from_scratch=train_from_scratch).to(DEVICE)

    if global_params is not None:
        try:
            set_params(model, global_params, DEVICE)
        except Exception as exc:
            print(f"[Parasitic] Error setting global params: {exc}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    model.train()
    final_loss = 0.0
    final_accuracy = 0.0
    for epoch in range(local_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            _, predicted = torch.max(out, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        final_loss = running_loss / (len(train_loader.dataset) if hasattr(train_loader, "dataset") else local_size)
        final_accuracy = (correct / total) * 100.0 if total else 0.0
        print(f"[Parasitic] Epoch {epoch + 1}/{local_epochs} loss: {final_loss:.4f} acc: {final_accuracy:.2f}%")

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_FILE
    resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "epochs": local_epochs,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "train_from_scratch": train_from_scratch,
            "model_architecture": MODEL_ARCHITECTURE,
            "loss": final_loss,
            "accuracy": final_accuracy,
        },
        resolved_checkpoint,
    )

    print(f"[Parasitic] Model saved at {resolved_checkpoint}")
    print("[Parasitic] Client update completed.\n")

    return get_params(model), local_size


if __name__ == "__main__":
    print(f"[Parasitic] Starting training on device: {DEVICE}")
    client_update(train_from_scratch=True)
