"""Own implementations of the paper's interventions with per-layer energy bookkeeping.

The formulas are identical to TransformerLens' ``JacobianLens.swap_hooks`` /
``ablation_hooks`` (verified by tests/test_interventions.py); the extra value is a
single code path for raw / unit / random / foil bases and a record of the actual
``||Δh||`` produced at every hooked layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name


@dataclass
class EnergyStats:
    """Per-layer sums over the intervened positions of ||Δh||², ||h||², and max ||Δh||/||h||."""

    sum_dh2: Dict[int, float] = field(default_factory=dict)
    sum_h2: Dict[int, float] = field(default_factory=dict)
    max_rel: Dict[int, float] = field(default_factory=dict)

    def record(self, layer: int, h: torch.Tensor, delta: torch.Tensor) -> None:
        h_norm = h.float().norm(dim=-1)
        d_norm = delta.float().norm(dim=-1)
        self.sum_dh2[layer] = self.sum_dh2.get(layer, 0.0) + float((d_norm**2).sum().item())
        self.sum_h2[layer] = self.sum_h2.get(layer, 0.0) + float((h_norm**2).sum().item())
        rel = (d_norm / h_norm.clamp_min(1e-6)).max().item()
        self.max_rel[layer] = max(self.max_rel.get(layer, 0.0), float(rel))

    def as_rows(self) -> List[Dict[str, float]]:
        return [
            {"layer": layer, "sum_dh2": self.sum_dh2[layer], "sum_h2": self.sum_h2[layer], "max_rel": self.max_rel[layer]}
            for layer in sorted(self.sum_dh2)
        ]

    @property
    def total_dh2(self) -> float:
        return float(sum(self.sum_dh2.values()))


def unit_basis(basis: torch.Tensor) -> torch.Tensor:
    """Normalize each of the two basis vectors (rows) to unit length."""
    return basis / basis.norm(dim=-1, keepdim=True)


def gram_matched_random_basis(basis: torch.Tensor, seed: int) -> torch.Tensor:
    """Random 2-D basis (rows) in the same ambient space with exactly the same Gram matrix."""
    gram = basis @ basis.T  # [2, 2]
    chol = torch.linalg.cholesky(gram)  # gram = chol @ chol.T
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gauss = torch.randn(basis.shape[1], 2, generator=gen).to(basis.device, basis.dtype)
    q, _ = torch.linalg.qr(gauss)  # [d, 2] orthonormal columns
    return (q @ chol.T).T  # rows: [2, d]; Gram = chol @ q^T q @ chol^T = gram


def swap_hooks_with_stats(
    model,
    basis: torch.Tensor,
    layers: Sequence[int],
    *,
    alpha: float,
    positions: Optional[Sequence[int]] = None,
    stats: Optional[EnergyStats] = None,
) -> List[Tuple[str, object]]:
    """Coordinate swap ``h <- h + alpha * V (sigma(c) - c)``, ``c = V^+ h``, for a 2-row basis."""
    if basis.shape[0] != 2:
        raise ValueError(f"basis must have two rows, got {tuple(basis.shape)}")
    cosine = abs(float(torch.nn.functional.cosine_similarity(basis[0:1], basis[1:2]).item()))
    if cosine >= 0.999:
        raise ValueError(f"basis vectors are numerically parallel (|cos|={cosine:.6f})")
    matrix = basis.T.float()  # [d, 2]
    pinv = torch.linalg.pinv(matrix)  # [2, d]
    hooks = []
    for layer in layers:

        def transform(selected, matrix=matrix, pinv=pinv, layer=layer):
            local_matrix = matrix.to(selected.device)
            local_pinv = pinv.to(selected.device)
            h = selected.float()
            coords = h @ local_pinv.T
            delta = alpha * ((coords[..., [1, 0]] - coords) @ local_matrix.T)
            if stats is not None:
                stats.record(layer, h, delta)
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, positions, model.cfg.d_model)))
    return hooks


def ablation_hooks_with_stats(
    model,
    vectors: torch.Tensor,
    layers: Sequence[int],
    *,
    positions: Optional[Sequence[int]] = None,
    stats: Optional[EnergyStats] = None,
) -> List[Tuple[str, object]]:
    """Project the unit directions of ``vectors`` (rows) out of the residual stream, sequentially."""
    units = unit_basis(vectors.float())
    hooks = []
    for layer in layers:

        def transform(selected, units=units, layer=layer):
            local_units = units.to(selected.device)
            h = selected.float()
            result = h
            for unit in local_units:
                coeff = result @ unit
                result = result - coeff.unsqueeze(-1) * unit
            if stats is not None:
                stats.record(layer, h, result - h)
            return result

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, positions, model.cfg.d_model)))
    return hooks


def logit_lens_vectors(model, token_ids: Sequence[int]) -> torch.Tensor:
    """Identity-transport (logit lens) directions: the unembedding columns, rows ``[n, d]``."""
    return model.W_U[:, list(token_ids)].float().T


def coordinate_map_hooks(
    model,
    basis: torch.Tensor,
    layers: Sequence[int],
    *,
    mode: str,
    alpha: float = 1.0,
    positions: Optional[Sequence[int]] = None,
    stats: Optional[EnergyStats] = None,
    orthogonalize_to: Optional[torch.Tensor] = None,
    rescale_after_orthogonalize: bool = False,
) -> List[Tuple[str, object]]:
    """The swap and its exact additive halves, in one code path.

    With ``c = V^+ h`` for ``V = [v_s, v_t]``:

    * ``swap``    -> ``h += alpha (c_t - c_s)(v_s - v_t)``   (coords ``(c_s, c_t) -> (c_t, c_s)``)
    * ``remove``  -> ``h += alpha (c_t - c_s) v_s``          (coords -> ``(c_t, c_t)``: source pulled down)
    * ``install`` -> ``h += alpha (c_s - c_t) v_t``          (coords -> ``(c_s, c_s)``: target raised)

    ``swap = remove + install`` exactly, so the two halves decompose the paper's
    intervention into "erase the bridge entity" and "write the replacement".

    ``orthogonalize_to`` removes the component of the update along a given direction
    before it is applied. Passing the layer's answer-contrast direction
    ``J^T (W_U[:, swap_answer] - W_U[:, answer])`` deletes the update's first-order
    *direct* push on the two answer logits, leaving only whatever effect is mediated
    by the model's own computation. ``rescale_after_orthogonalize`` restores the
    original update norm so the comparison is energy-matched rather than weaker.
    """
    if basis.shape[0] != 2:
        raise ValueError(f"basis must have two rows, got {tuple(basis.shape)}")
    if mode not in ("swap", "remove", "install"):
        raise ValueError(f"unknown mode {mode!r}")
    matrix = basis.T.float()  # [d, 2]
    pinv = torch.linalg.pinv(matrix)  # [2, d]
    hooks = []
    for layer in layers:

        def transform(selected, matrix=matrix, pinv=pinv, layer=layer):
            local_matrix = matrix.to(selected.device)
            local_pinv = pinv.to(selected.device)
            h = selected.float()
            coords = h @ local_pinv.T  # [..., 2]
            gap = (coords[..., 1] - coords[..., 0]).unsqueeze(-1)  # c_t - c_s
            v_s = local_matrix[:, 0]
            v_t = local_matrix[:, 1]
            if mode == "swap":
                delta = alpha * gap * (v_s - v_t)
            elif mode == "remove":
                delta = alpha * gap * v_s
            else:
                delta = -alpha * gap * v_t
            if orthogonalize_to is not None:
                unit = orthogonalize_to.to(selected.device).float()
                unit = unit / unit.norm()
                before = delta.norm(dim=-1, keepdim=True)
                delta = delta - (delta @ unit).unsqueeze(-1) * unit
                if rescale_after_orthogonalize:
                    delta = delta * (before / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9))
            if stats is not None:
                stats.record(layer, h, delta)
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, positions, model.cfg.d_model)))
    return hooks


def clamp_hooks(
    model,
    basis: torch.Tensor,
    layers: Sequence[int],
    clean_coords: Dict[int, torch.Tensor],
    *,
    exchange: bool = True,
    positions: Optional[Sequence[int]] = None,
    stats: Optional[EnergyStats] = None,
    orthogonalize_to: Optional[torch.Tensor] = None,
    rescale_after_orthogonalize: bool = False,
) -> List[Tuple[str, object]]:
    """Hold the two lens coordinates at their exchanged clean-run values.

    The published protocol describes the intervention as *clamping* a lens
    coordinate to another token's. Clamping is idempotent: applied at thirteen
    consecutive layers it keeps the state swapped. The coordinate exchange that
    the reference hooks implement is an involution instead -- it exchanges
    whatever coordinates are currently present -- so applying it layer after
    layer alternates between the swapped and the original state. This hook
    implements the clamp: at every layer the coordinates are forced to the clean
    run's exchanged values ``(c_t, c_s)``, regardless of what the previous layer
    left behind.

    Args:
        clean_coords: ``{layer: [1, pos, 2]}`` coordinates captured from the
            unmodified run at the same layers and positions.
        exchange: With the default ``True`` the two coordinates are exchanged
            before clamping, which is the swap. Pass ``False`` when
            ``clean_coords`` already holds the final target values -- otherwise
            they are exchanged a second time and the intervention cancels itself.
    """
    if basis.shape[0] != 2:
        raise ValueError(f"basis must have two rows, got {tuple(basis.shape)}")
    matrix = basis.T.float()
    pinv = torch.linalg.pinv(matrix)
    hooks = []
    for layer in layers:
        target = (clean_coords[layer][..., [1, 0]] if exchange else clean_coords[layer]).float()

        def transform(selected, matrix=matrix, pinv=pinv, target=target, layer=layer):
            local_matrix = matrix.to(selected.device)
            local_pinv = pinv.to(selected.device)
            local_target = target.to(selected.device)
            h = selected.float()
            coords = h @ local_pinv.T
            if local_target.shape[-2] != coords.shape[-2]:
                raise ValueError(
                    f"clamp targets cover {local_target.shape[-2]} positions but the hook saw "
                    f"{coords.shape[-2]}"
                )
            delta = (local_target - coords) @ local_matrix.T
            if orthogonalize_to is not None:
                unit = orthogonalize_to.to(selected.device).float()
                unit = unit / unit.norm()
                before = delta.norm(dim=-1, keepdim=True)
                delta = delta - (delta @ unit).unsqueeze(-1) * unit
                if rescale_after_orthogonalize:
                    delta = delta * (before / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9))
            if stats is not None:
                stats.record(layer, h, delta)
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, positions, model.cfg.d_model)))
    return hooks


def swapped_fraction(got: torch.Tensor, clean: torch.Tensor, *, min_gap_frac: float = 1.0) -> float:
    """How far the source coordinate has travelled toward the target, in gap units.

    0 means the coordinates are where the clean run left them, 1 means fully
    exchanged. Positions whose two clean coordinates nearly coincide have no
    meaningful gap to measure against, so they are dropped rather than divided by:
    an earlier version clamped the denominator with ``gap.clamp(min=1e-6)``, which
    silently flips the sign of every negative gap and produced ratios in the
    millions. Kept positions are those whose |gap| is at least ``min_gap_frac``
    times the median |gap|.
    """
    gap = clean[..., 1] - clean[..., 0]
    magnitude = gap.abs()
    keep = magnitude >= min_gap_frac * magnitude.median()
    if not bool(keep.any()):
        return float("nan")
    return float(((got[..., 0] - clean[..., 0])[keep] / gap[keep]).median().item())
