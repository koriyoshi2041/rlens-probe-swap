"""Stage: lens artifact compatibility and geometry (norms, J-R agreement, conditioning)."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch

from ..data import SwapItem
from ..model import lens_summary
from .common import resolve_item


def top_singular_value(matrix: torch.Tensor, iters: int = 30) -> float:
    vec = torch.randn(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    vec = vec / vec.norm()
    for _ in range(iters):
        vec = matrix.T @ (matrix @ vec)
        vec = vec / vec.norm()
    return float((matrix @ vec).norm().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


@torch.inference_mode()
def run(model, lenses: Dict[str, object], items: List[SwapItem], out_dir) -> Dict[str, object]:
    for kind, lens in lenses.items():
        lens.validate_model(model)
        print(f"[artifacts] {kind}-lens validate_model OK: {lens_summary(lens)}")
    tokenizer = model.tokenizer
    resolved = [resolve_item(tokenizer, item, "as_is") for item in items]
    feasible = [r for r in resolved if r.concept_feasible]
    unique_ids = sorted({r.src.first_id for r in feasible} | {r.tgt.first_id for r in feasible})
    id_pos = {tid: i for i, tid in enumerate(unique_ids)}
    layers = list(lenses["J"].source_layers)
    device = model.W_U.device

    per_layer: List[Dict[str, object]] = []
    cos_st = {kind: np.full((len(items), len(layers)), np.nan, dtype=np.float32) for kind in lenses}
    for li, layer in enumerate(layers):
        vecs = {kind: lens.lens_vectors(model, unique_ids, layer) for kind, lens in lenses.items()}
        norms = {kind: v.norm(dim=-1) for kind, v in vecs.items()}
        cos_jr = _cos(vecs["J"], vecs["R"])
        for r in feasible:
            s, t = id_pos[r.src.first_id], id_pos[r.tgt.first_id]
            for kind in lenses:
                cos_st[kind][r.item.index, li] = float(_cos(vecs[kind][s : s + 1], vecs[kind][t : t + 1]).item())
        mats = {kind: lens.jacobians[layer].to(device=device, dtype=torch.float32) for kind, lens in lenses.items()}
        rel_diff = float(((mats["J"] - mats["R"]).norm() / mats["J"].norm()).item())
        row = {
            "layer": layer,
            "median_norm_J": float(norms["J"].median()),
            "median_norm_R": float(norms["R"].median()),
            "median_norm_ratio_R_over_J": float((norms["R"] / norms["J"]).median()),
            "median_cos_J_R_same_token": float(cos_jr.median()),
            "min_cos_J_R_same_token": float(cos_jr.min()),
            "median_abs_cos_src_tgt_J": float(np.nanmedian(np.abs(cos_st["J"][:, li]))),
            "median_abs_cos_src_tgt_R": float(np.nanmedian(np.abs(cos_st["R"][:, li]))),
            "max_abs_cos_src_tgt_J": float(np.nanmax(np.abs(cos_st["J"][:, li]))),
            "max_abs_cos_src_tgt_R": float(np.nanmax(np.abs(cos_st["R"][:, li]))),
            "fro_J": float(mats["J"].norm()),
            "fro_R": float(mats["R"].norm()),
            "spectral_J": top_singular_value(mats["J"]),
            "spectral_R": top_singular_value(mats["R"]),
            "rel_fro_diff_J_minus_R": rel_diff,
            "cos_flat_J_R": float(_cos(mats["J"].flatten()[None], mats["R"].flatten()[None]).item()),
        }
        per_layer.append(row)
        print(
            f"[artifacts] L{layer:02d} |vJ|={row['median_norm_J']:.3g} |vR|={row['median_norm_R']:.3g} "
            f"R/J={row['median_norm_ratio_R_over_J']:.2f} cos(vJ,vR)={row['median_cos_J_R_same_token']:.3f} "
            f"|cos(s,t)| J={row['median_abs_cos_src_tgt_J']:.3f} R={row['median_abs_cos_src_tgt_R']:.3f} "
            f"maxJ={row['max_abs_cos_src_tgt_J']:.3f} maxR={row['max_abs_cos_src_tgt_R']:.3f} "
            f"fro J={row['fro_J']:.3g} R={row['fro_R']:.3g} spec J={row['spectral_J']:.3g} R={row['spectral_R']:.3g} "
            f"relΔ={rel_diff:.3f}"
        )
        for lens in lenses.values():
            lens.clear_device_cache()
    result = {
        "lenses": {kind: lens_summary(lens) for kind, lens in lenses.items()},
        "n_feasible_items": len(feasible),
        "n_unique_concept_tokens": len(unique_ids),
        "per_layer": per_layer,
    }
    (out_dir / "artifacts.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    np.savez(out_dir / "artifacts_cos_src_tgt.npz", layers=np.array(layers), **{f"cos_st_{k}": v for k, v in cos_st.items()})
    return result
