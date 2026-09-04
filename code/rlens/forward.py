"""Thin forward helpers: final-position log-probs with optional intervention hooks."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


@torch.inference_mode()
def final_logprobs(model, tokens: torch.Tensor, fwd_hooks: Optional[List[Tuple[str, object]]] = None) -> torch.Tensor:
    """Log-softmax at the final position, fp32, shape ``[d_vocab]``."""
    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(tokens, return_type="logits")
    else:
        logits = model(tokens, return_type="logits")
    return torch.log_softmax(logits[0, -1].float(), dim=-1)


def kl_divergence(logp_ref: torch.Tensor, logp_other: torch.Tensor) -> float:
    """KL(ref || other) between two log-prob vectors."""
    return float((logp_ref.exp() * (logp_ref - logp_other)).sum().item())
