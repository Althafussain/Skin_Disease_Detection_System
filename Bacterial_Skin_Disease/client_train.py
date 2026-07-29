from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .dataloader import load_bacterial_data
    from .model_builder import build_bacterial_model
    from .utils import get_params, set_params, set_seed
except ImportError:
    from dataloader import load_bacterial_data
    from model_builder import build_bacterial_model
    from utils import get_params, set_params, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 5e-4
LOCAL_EPOCHS = 5
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "bacterial_model.pth"
MODEL_ARCHITECTURE = "mobilenet_v2"


def client_update(
    global_params=None,
    train_from_scratch=True,
    local_epochs=LOCAL_EPOCHS,
    checkpoint_path=None,
    seed=42,
):
    set_seed(seed)

    try:
        train_loader, _, num_classes = load_bacterial_data()
    except FileNotFoundError as exc:
        print(f"[Bacterial] Dataset not found: {exc}")
        return None, None

    if len(train_loader) == 0:
        print("[Bacterial] Train loader is empty. Skipping client update.")
        return None, None

    classes = list(getattr(train_loader.dataset, "classes", []))
    class_to_idx = dict(getattr(train_loader.dataset, "class_to_idx", {}))

    print(f"[Bacterial] Loaded dataset with {num_classes} classes.")
    model = build_bacterial_model(num_classes=num_classes, train_from_scratch=train_from_scratch).to(DEVICE)

    if global_params is not None:
        try:
            set_params(model, global_params, DEVICE)
        except Exception as exc:
            print(f"[Bacterial] Error setting global params: {exc}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    final_loss = 0.0
    final_accuracy = 0.0
    for epoch in range(local_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        final_loss = total_loss / len(train_loader)
        final_accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        print(
            f"[Bacterial] Epoch {epoch + 1}/{local_epochs} "
            f"loss: {final_loss:.4f} acc: {final_accuracy:.2f}%"
        )

    resolved_checkpoint_path = (
        Path(checkpoint_path).resolve() if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
    )
    resolved_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
        resolved_checkpoint_path,
    )
    print(f"[Bacterial] Model saved to {resolved_checkpoint_path}")

    return get_params(model), model


if __name__ == "__main__":
    print("[Bacterial] Starting client training...")
    params, model = client_update(train_from_scratch=True)
    if params is not None:
        print("[Bacterial] Training finished. Parameters collected.")
    else:
        print("[Bacterial] Training failed or dataset missing.")
