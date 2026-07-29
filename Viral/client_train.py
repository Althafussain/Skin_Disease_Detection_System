import ssl
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ssl._create_default_https_context = ssl._create_unverified_context

try:
    from .dataloader import load_viral_data
    from .model_builder import build_viral_model
    from .utils import get_params, set_params, set_seed
except ImportError:
    from dataloader import load_viral_data
    from model_builder import build_viral_model
    from utils import get_params, set_params, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 5e-4
LOCAL_EPOCHS = 20
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = BASE_DIR / "saved_viral_models"
DEFAULT_MODEL_PATH = DEFAULT_SAVE_DIR / "viral_skin_model.pth"
DEFAULT_PARAMS_PATH = DEFAULT_SAVE_DIR / "viral_skin_params.pth"
MODEL_ARCHITECTURE = "mobilenet_v2"


def client_update(
    global_params=None,
    save_model=False,
    model_save_path=None,
    params_save_path=None,
    save_dir=None,
    train_from_scratch=True,
    local_epochs=LOCAL_EPOCHS,
    seed=42,
):
    set_seed(seed)
    empty_stats = {"loss_history": [], "final_loss": None, "final_accuracy": None, "num_samples": 0}

    try:
        train_loader, _, num_classes, local_size = load_viral_data()
    except FileNotFoundError as exc:
        print(f"[Viral_Skin] Dataset not found: {exc}")
        return None, empty_stats
    except Exception as exc:
        print(f"[Viral_Skin] Error loading dataset: {exc}")
        return None, empty_stats

    if len(train_loader) == 0:
        print("[Viral_Skin] Train loader is empty. Skipping training.")
        return None, empty_stats

    classes = list(getattr(train_loader.dataset, "classes", []))
    class_to_idx = dict(getattr(train_loader.dataset, "class_to_idx", {}))

    print(f"[Viral_Skin] Loaded dataset with {num_classes} classes.")
    model = build_viral_model(num_classes=num_classes, train_from_scratch=train_from_scratch).to(DEVICE)

    if global_params is not None:
        try:
            set_params(model, global_params, DEVICE)
            print("[Viral_Skin] Global parameters loaded.")
        except Exception as exc:
            print(f"[Viral_Skin] Failed to set global params: {exc}")
            return None, empty_stats

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    training_stats = {
        "loss_history": [],
        "final_loss": None,
        "final_accuracy": None,
        "num_samples": local_size,
    }

    model.train()
    print(f"[Viral_Skin] Training for {local_epochs} epochs...")

    final_loss = 0.0
    final_accuracy = 0.0
    for epoch in range(local_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        final_loss = total_loss / len(train_loader)
        final_accuracy = 100.0 * correct / total if total else 0.0
        training_stats["loss_history"].append(final_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == local_epochs - 1:
            print(
                f"[Viral_Skin] Epoch {epoch + 1}/{local_epochs} "
                f"Loss: {final_loss:.4f} | Accuracy: {final_accuracy:.2f}%"
            )

    training_stats["final_loss"] = final_loss
    training_stats["final_accuracy"] = final_accuracy
    print("[Viral_Skin] Training completed.")

    updated_params = get_params(model)

    if save_model:
        resolved_save_dir = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
        resolved_save_dir.mkdir(parents=True, exist_ok=True)

        resolved_model_path = Path(model_save_path) if model_save_path is not None else DEFAULT_MODEL_PATH
        resolved_model_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
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
            "training_stats": training_stats,
            "loss": final_loss,
            "accuracy": final_accuracy,
        }
        torch.save(checkpoint, resolved_model_path)
        print(f"[Viral_Skin] Model saved -> {resolved_model_path}")

        resolved_params_path = Path(params_save_path) if params_save_path is not None else DEFAULT_PARAMS_PATH
        resolved_params_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": updated_params,
                "num_classes": num_classes,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "training_stats": training_stats,
            },
            resolved_params_path,
        )
        print(f"[Viral_Skin] Params saved -> {resolved_params_path}")

    return updated_params, training_stats


if __name__ == "__main__":
    print("[Viral_Skin] Starting client training...")
    params, stats = client_update(save_model=True, train_from_scratch=True)
    if params is not None:
        print("[Viral_Skin] Training finished successfully.")
        print(f"[Viral_Skin] Final loss: {stats['final_loss']:.4f}")
        print(f"[Viral_Skin] Final accuracy: {stats['final_accuracy']:.2f}%")
    else:
        print("[Viral_Skin] Training failed.")
