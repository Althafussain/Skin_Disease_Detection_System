import random

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
    load_dict = {k: params[k].to(device) for k in params if k in model_dict}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict, strict=False)
