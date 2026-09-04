"""Model and lens loading through the exact path validated in verify_environment.py."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from .paths import LENS_FILES, MODEL_DIR


def load_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers(str(MODEL_DIR), device=device, dtype=dtype)
    model.eval()
    return model


def load_lenses(kinds: Sequence[str] = ("J", "R")) -> Dict[str, object]:
    from transformer_lens.tools.analysis import JacobianLens

    return {kind: JacobianLens.load(str(LENS_FILES[kind])) for kind in kinds}


def model_summary(model) -> Dict[str, object]:
    return {
        "model_name": getattr(model.cfg, "model_name", None),
        "n_layers": int(model.cfg.n_layers),
        "d_model": int(model.cfg.d_model),
        "d_vocab": int(model.W_U.shape[1]),
        "unembed_dtype": str(model.W_U.dtype),
        "device": str(model.W_U.device),
    }


def lens_summary(lens) -> Dict[str, object]:
    return {
        "d_model": int(lens.d_model),
        "n_prompts": int(lens.n_prompts),
        "source_layers": list(lens.source_layers),
        "metadata": dict(lens.metadata),
    }
