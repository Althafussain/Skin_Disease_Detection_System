from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .dataloader import load_fungal_data
    from .model_builder import build_fungal_model
    from .utils import get_params, set_params, set_seed
except ImportError:
    from dataloader import load_fungal_data
    from model_builder import build_fungal_model
    from utils import get_params, set_params, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 5e-4
LOCAL_EPOCHS = 25
BASE_DIR = Path(__file__).resolve().parent
MODEL_SAVE_PATH = BASE_DIR / "trained_model.pth"
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
        train_loader, _, num_classes = load_fungal_data()
    except FileNotFoundError as exc:
        print(f"[Fungal] Dataset not found: {exc}")
        return None, None

    if len(train_loader) == 0:
        print("[Fungal] Train loader is empty. Skipping client update.")
        return None, None

    classes = list(getattr(train_loader.dataset, "classes", []))
    class_to_idx = dict(getattr(train_loader.dataset, "class_to_idx", {}))

    print(f"[Fungal] Loaded dataset with {num_classes} classes.")
    print(f"[Fungal] Device: {DEVICE}")

    model = build_fungal_model(num_classes=num_classes, train_from_scratch=train_from_scratch).to(DEVICE)

    if global_params is not None:
        try:
            set_params(model, global_params, DEVICE)
            print("[Fungal] Loaded global parameters into model.")
        except Exception as exc:
            print(f"[Fungal] Error setting global params: {exc}")

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
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        final_loss = total_loss / len(train_loader)
        final_accuracy = (correct / total * 100.0) if total > 0 else 0.0
        print(
            f"[Fungal] Epoch {epoch + 1}/{local_epochs} | "
            f"loss: {final_loss:.4f} | train acc: {final_accuracy:.2f}%"
        )

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else MODEL_SAVE_PATH
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
    print(f"[Fungal] Saved trained model to: {resolved_checkpoint}")
    print("[Fungal] Client training completed.\n")

    return get_params(model), model


if __name__ == "__main__":
    print("[Fungal] Starting client training...")
    params, model = client_update(global_params=None, train_from_scratch=True)
    if params is None:
        print("[Fungal] Training failed (dataset missing or empty).")
    else:
        print("[Fungal] Training finished. Parameters collected.")
