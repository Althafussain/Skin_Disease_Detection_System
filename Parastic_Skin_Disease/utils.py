import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_params(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_params(model, params, device):
    if params is None:
        return

    model_dict = model.state_dict()
    compatible_params = {k: v.to(device) for k, v in params.items() if k in model_dict}
    model_dict.update(compatible_params)
    model.load_state_dict(model_dict, strict=False)


def save_checkpoint(model, out_path="student3_checkpoint.pth"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)


def compute_class_weights(dataset):
    counts = Counter([y for _, y in dataset.samples])
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(len(counts))]
    return torch.tensor(weights, dtype=torch.float)
