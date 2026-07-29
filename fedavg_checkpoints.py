from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import torch

from fedavg_utils import fedavg_compatible_state_dicts
from multi_model_predictor import DEFAULT_MODEL_PATHS, resolve_model_path


def _load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("model_state_dict"), dict):
        return checkpoint_obj["model_state_dict"]
    if isinstance(checkpoint_obj, dict) and checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
        return checkpoint_obj
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def fedavg_checkpoints(
    model_paths: Mapping[str, str | Path],
    output_path: str | Path,
    client_weights: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    resolved_paths = {
        client_name: resolve_model_path(client_name, model_paths.get(client_name))
        for client_name in DEFAULT_MODEL_PATHS
    }
    state_dicts = {client_name: _load_state_dict(path) for client_name, path in resolved_paths.items()}
    averaged_state_dict, stats = fedavg_compatible_state_dicts(
        state_dicts,
        client_weights=client_weights,
        require_all_clients=True,
    )

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_state_dict": averaged_state_dict,
            "stats": stats,
            "source_models": {name: str(path) for name, path in resolved_paths.items()},
            "client_weights": dict(client_weights) if client_weights is not None else None,
            "note": "Classifier-head keys with incompatible shapes are skipped.",
        },
        out_path,
    )
    return {"output_path": str(out_path), "stats": stats}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a FedAvg global checkpoint from viral/parasitic/fungal/bacterial model checkpoints."
    )
    parser.add_argument("--output", required=True, help="Path for aggregated FedAvg checkpoint.")
    parser.add_argument("--viral-model", default=None, help="Viral checkpoint path or directory.")
    parser.add_argument("--parasitic-model", default=None, help="Parasitic checkpoint path or directory.")
    parser.add_argument("--fungal-model", default=None, help="Fungal checkpoint path or directory.")
    parser.add_argument("--bacterial-model", default=None, help="Bacterial checkpoint path or directory.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = fedavg_checkpoints(
        model_paths={
            "viral": args.viral_model,
            "parasitic": args.parasitic_model,
            "fungal": args.fungal_model,
            "bacterial": args.bacterial_model,
        },
        output_path=args.output,
    )
    print("FedAvg checkpoint created.")
    print(f"Output: {result['output_path']}")
    print(f"Stats: {result['stats']}")


if __name__ == "__main__":
    main()
