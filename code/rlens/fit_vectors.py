"""Fit only the lens vectors we intervene with, instead of the whole transport matrix.

The reference estimator builds ``J`` row by row: for output dimension ``d`` it
plants the one-hot cotangent ``e_d`` at every valid target position, takes one
backward pass, and averages the gradient at the source layer over valid source
positions. That is ``J[d, :]``, and a full fit costs ``d_model`` backward passes
per prompt -- 4096 here, which is hours for a 9B model.

Every intervention in this project uses only ``v_t = J^T W_U[:, t]`` for the
handful of concept tokens in the item set. By linearity of the backward pass,

    J^T u = sum_d u_d J[d, :]

is obtained by planting the cotangent ``u`` itself instead of a one-hot: one
backward pass per direction. With ~90 concept tokens that is ~90 passes per
prompt rather than 4096, and the result is numerically the same quantity, not an
approximation. ``tests/test_fit_vectors.py`` checks that against the reference
``JacobianLens.fit`` on a shared prompt.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from transformer_lens.tools.analysis.jacobian_lens import (
    DEFAULT_SKIP_FIRST_POSITIONS,
    _frozen_parameters,
    _resid_post_hook_name,
    _validate_residual_activation,
)


def lens_vectors_for_prompt(
    model,
    tokens: torch.Tensor,
    cotangents: torch.Tensor,
    source_layers: Sequence[int],
    *,
    dim_batch: int = 16,
    skip_first_positions: int = DEFAULT_SKIP_FIRST_POSITIONS,
) -> Dict[int, torch.Tensor]:
    """``{layer: [n, d_model]}`` where row i is ``J_layer^T cotangents[i]`` for this prompt.

    Mirrors ``_jacobian_for_prompt``: cotangents are planted at every valid target
    position, and gradients are averaged over valid source positions.
    """
    d_model = model.cfg.d_model
    target_layer = model.cfg.n_layers - 1
    seq_len = tokens.shape[1]
    valid = list(range(skip_first_positions, seq_len - 1))
    if not valid:
        raise ValueError(f"prompt of length {seq_len} has no valid source position")
    n = cotangents.shape[0]
    batch = min(dim_batch, n)
    replicated = tokens.expand(batch, -1)
    captured: Dict[str, torch.Tensor] = {}
    root = _resid_post_hook_name(min(source_layers))
    hook_layers = sorted(set(source_layers) | {target_layer})

    def capture(activation, hook):
        _validate_residual_activation(activation, d_model=d_model, hook_name=hook.name)
        if hook.name == root and not activation.requires_grad:
            activation.requires_grad_(True)
        captured[hook.name] = activation
        return activation

    hooks = [(_resid_post_hook_name(layer), capture) for layer in hook_layers]
    with torch.enable_grad(), model.hooks(fwd_hooks=hooks):
        model(replicated, return_type=None)
    target = captured[_resid_post_hook_name(target_layer)]
    sources = [captured[_resid_post_hook_name(layer)] for layer in source_layers]
    device = target.device
    positions = torch.tensor(valid, device=device)
    out = {layer: torch.zeros(n, d_model, dtype=torch.float32) for layer in source_layers}
    cotangent = torch.zeros_like(target)
    n_passes = -(-n // batch)
    for k in range(n_passes):
        start = k * batch
        take = min(batch, n - start)
        cotangent.zero_()
        block = cotangents[start : start + take].to(device=device, dtype=cotangent.dtype)
        cotangent[:take, positions, :] = block[:, None, :]
        grads = torch.autograd.grad(
            outputs=target, inputs=sources, grad_outputs=cotangent, retain_graph=k < n_passes - 1
        )
        for layer, grad in zip(source_layers, grads):
            rows = grad[:take, positions.to(grad.device), :].float().mean(dim=1)
            out[layer][start : start + take, :] = rows.cpu()
        del grads
    return out


def fit_lens_vectors(
    model,
    prompts: Sequence[str],
    token_ids: Sequence[int],
    source_layers: Sequence[int],
    *,
    dim_batch: int = 16,
    max_seq_len: int = 128,
    skip_first_positions: int = DEFAULT_SKIP_FIRST_POSITIONS,
    progress_every: int = 10,
) -> Dict[int, torch.Tensor]:
    """Average ``J^T W_U[:, t]`` over prompts, for every requested token."""
    cotangents = model.W_U[:, list(token_ids)].float().T.contiguous()  # [n, d_model]
    totals: Dict[int, torch.Tensor] = {}
    done = 0
    with _frozen_parameters(model):
        for i, prompt in enumerate(prompts):
            tokens = model.to_tokens(prompt)[:, :max_seq_len]
            if tokens.shape[1] <= skip_first_positions + 1:
                continue
            part = lens_vectors_for_prompt(
                model, tokens, cotangents, source_layers,
                dim_batch=dim_batch, skip_first_positions=skip_first_positions,
            )
            for layer, value in part.items():
                totals[layer] = value if layer not in totals else totals[layer] + value
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"    [fit_vectors] {done}/{len(prompts)} passages", flush=True)
    if done == 0:
        raise ValueError("no prompt was long enough to contribute")
    return {layer: value / done for layer, value in totals.items()}, done


class FittedVectorLens:
    """Minimal stand-in exposing ``lens_vectors`` for code that expects a JacobianLens."""

    def __init__(self, vectors: Dict[int, torch.Tensor], token_ids: Sequence[int], n_prompts: int, metadata=None):
        self.vectors = {int(k): v for k, v in vectors.items()}
        self.index = {int(t): i for i, t in enumerate(token_ids)}
        self.n_prompts = int(n_prompts)
        self.metadata = dict(metadata or {})

    @property
    def source_layers(self) -> List[int]:
        return sorted(self.vectors)

    def lens_vectors(self, model, tokens, layer: int) -> torch.Tensor:
        ids = [tokens] if isinstance(tokens, int) else list(tokens)
        rows = [self.index[int(t)] for t in ids]
        return self.vectors[int(layer)][rows].to(model.W_U.device)

    def save(self, path: str) -> None:
        torch.save({"vectors": self.vectors, "token_ids": list(self.index), "n_prompts": self.n_prompts,
                    "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str) -> "FittedVectorLens":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(payload["vectors"], payload["token_ids"], payload["n_prompts"], payload.get("metadata"))
