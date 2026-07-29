from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _normalize_weights(
    keys: Sequence[str],
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    if not keys:
        raise ValueError("No clients provided for FedAvg.")

    if weights is None:
        return {key: 1.0 for key in keys}

    normalized: Dict[str, float] = {}
    for key in keys:
        value = float(weights.get(key, 1.0))
        if value < 0:
            raise ValueError(f"Negative client weight is not allowed: {key} -> {value}")
        normalized[key] = value

    if not any(weight > 0 for weight in normalized.values()):
        raise ValueError("At least one client must have a positive weight.")
    return normalized


def fedavg_label_probabilities(
    client_label_probs: Mapping[str, Mapping[str, float]],
    client_weights: Optional[Mapping[str, float]] = None,
    normalize_output: bool = True,
) -> Dict[str, float]:
    """Average per-label probabilities from multiple client models."""
    if not client_label_probs:
        raise ValueError("client_label_probs is empty.")

    clients = list(client_label_probs.keys())
    weights = _normalize_weights(clients, client_weights)

    label_scores: MutableMapping[str, float] = defaultdict(float)
    total_weight = 0.0

    for client_name, probs in client_label_probs.items():
        client_weight = weights[client_name]
        if client_weight <= 0:
            continue
        total_weight += client_weight
        for label, prob in probs.items():
            label_scores[str(label)] += client_weight * float(prob)

    if total_weight == 0:
        raise ValueError("Total positive client weight is zero.")

    averaged = {label: score / total_weight for label, score in label_scores.items()}
    if normalize_output:
        norm = sum(averaged.values())
        if norm > 0:
            averaged = {label: score / norm for label, score in averaged.items()}

    return dict(sorted(averaged.items(), key=lambda kv: kv[1], reverse=True))


def fedavg_compatible_state_dicts(
    client_state_dicts: Mapping[str, Mapping[str, "torch.Tensor"]],
    client_weights: Optional[Mapping[str, float]] = None,
    require_all_clients: bool = True,
    min_clients_per_key: int = 2,
) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, object]]:
    """
    Average only compatible tensor keys across clients.

    This is useful when classifier heads have different shapes (different label spaces).
    Shared backbone keys are still averaged with FedAvg.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for parameter FedAvg.")
    if not client_state_dicts:
        raise ValueError("client_state_dicts is empty.")

    clients = list(client_state_dicts.keys())
    weights = _normalize_weights(clients, client_weights)
    all_keys = sorted({k for sd in client_state_dicts.values() for k in sd.keys()})

    averaged: Dict[str, torch.Tensor] = {}
    skipped = {}
    used_keys = []

    for key in all_keys:
        tensors = []
        for client_name in clients:
            state_dict = client_state_dicts[client_name]
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            if not torch.is_tensor(tensor):
                continue
            if not tensor.is_floating_point():
                continue
            tensors.append((client_name, tensor.detach().cpu()))

        if not tensors:
            skipped[key] = "no_floating_tensor"
            continue

        shape_groups: MutableMapping[Tuple[int, ...], list] = defaultdict(list)
        for item in tensors:
            shape_groups[tuple(item[1].shape)].append(item)
        selected_group = max(shape_groups.values(), key=len)

        required_count = len(clients) if require_all_clients else max(1, min_clients_per_key)
        if len(selected_group) < required_count:
            skipped[key] = "incompatible_or_missing"
            continue

        selected_weight = 0.0
        acc = None
        for client_name, tensor in selected_group:
            w = weights[client_name]
            if w <= 0:
                continue
            tensor_f = tensor.float()
            acc = tensor_f * w if acc is None else acc + (tensor_f * w)
            selected_weight += w

        if acc is None or selected_weight == 0:
            skipped[key] = "zero_weight"
            continue

        avg_tensor = (acc / selected_weight).to(dtype=selected_group[0][1].dtype)
        averaged[key] = avg_tensor
        used_keys.append(key)

    stats = {
        "num_clients": len(clients),
        "total_candidate_keys": len(all_keys),
        "averaged_keys": len(used_keys),
        "skipped_keys": len(skipped),
        "skipped_reasons": skipped,
    }
    return averaged, stats
