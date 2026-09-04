"""Rank readout: at every (layer, position) the rank of tracked tokens under a lens.

Reuses the library's private ``_unembed`` so numerics match ``JacobianLens.readout``
exactly (final norm + unembed + architecture logit transform).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name, _unembed

from .tokens import ranks_rowwise


@torch.inference_mode()
def rank_readout(
    model,
    lens: Optional[object],
    tokens: torch.Tensor,
    token_ids: Sequence[int],
    layers: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ranks, top1)`` with shapes ``[n_layers+1, pos, n]`` and ``[n_layers+1, pos]``.

    Row ``i`` corresponds to ``layers[i]`` read through ``lens`` (or the identity /
    logit lens when ``lens`` is None); the last row is the model's own output.
    """
    final_layer = model.cfg.n_layers - 1
    names = {layer: _resid_post_hook_name(layer) for layer in layers}
    wanted = set(names.values())
    logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted)
    ids = tuple(int(i) for i in token_ids)
    rank_rows = []
    top_rows = []
    for layer in layers:
        act = cache[names[layer]][0]  # [pos, d_model]
        transported = lens.transport(act, layer) if lens is not None else act.float()
        layer_logits = _unembed(model, transported)  # [pos, vocab] fp32
        rank_rows.append(ranks_rowwise(layer_logits, ids).cpu())
        top_rows.append(layer_logits.argmax(dim=-1).cpu())
    model_logits = logits[0].float()
    rank_rows.append(ranks_rowwise(model_logits, ids).cpu())
    top_rows.append(model_logits.argmax(dim=-1).cpu())
    del cache
    return torch.stack(rank_rows), torch.stack(top_rows)
