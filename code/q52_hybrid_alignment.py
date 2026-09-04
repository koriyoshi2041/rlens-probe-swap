#!/usr/bin/env python
"""Block 47: why does "J removes, R installs" beat both lenses? Ground truth from minimal pairs.

Block 43/43b: in L8-20 a clamp built from J's source vector and R's target vector flips
0.47 vs 0.41 (J) / 0.39 (R), robust to unit-normalisation; the reverse mix is worse. The 14
minimal-contrast pairs (same template, the cue token names the other entity) give the true
counterfactual activation difference D_l = h_cf - h at every position. Per lens and layer:

  * coefficient of D on the unit source vector (should be negative: the source is removed)
    and on the unit target vector (positive: the target is installed), at the cue position
    and summed over positions -> which lens's v_s / v_t is better aligned with what the
    model itself removes / installs;
  * the four hybrid clamps set to the COUNTERFACTUAL run's coordinates (block 10's
    subspace patch, but with mixed bases) -> does R's target direction help even when the
    target coordinate is the true one rather than the exchange heuristic.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block47_hybrid_alignment")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    pairs = json.loads((RESULTS_DIR / "block01" / "counterfactual_pairs.json").read_text(encoding="utf-8"))
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    align = (out_dir / "alignment.jsonl").open("w", encoding="utf-8")
    causal = (out_dir / "hybrid_counterfactual.jsonl").open("w", encoding="utf-8")
    for n, (i, j) in enumerate(pairs):
        ri, rj = resolved[i], resolved[j]
        ti = model.to_tokens(ri.prompt, prepend_bos=False)
        tj = model.to_tokens(rj.prompt, prepend_bos=False)
        if ti.shape != tj.shape:
            continue
        a, s = ri.answer.first_id, ri.swap_answer.first_id
        ids = [ri.src.first_id, ri.tgt.first_id]
        _, ci = model.run_with_cache(ti, names_filter=lambda x: x in wanted)
        _, cj = model.run_with_cache(tj, names_filter=lambda x: x in wanted)
        clean = final_logprobs(model, ti)
        base = float(clean[s] - clean[a])
        diff_tokens = [p for p in range(ti.shape[1]) if ti[0, p] != tj[0, p]]
        cue = diff_tokens[-1] if diff_tokens else ti.shape[1] - 1
        vec = {k: {l: lenses[k].lens_vectors(model, ids, l).float() for l in BAND} for k in ("J", "R")}
        for l in BAND:
            D = (cj[names[l]][0] - ci[names[l]][0]).float()  # [pos, d]
            row = {"index": i, "partner": j, "name": ri.item.name, "layer": l, "cue_pos": cue, "n_pos": int(ti.shape[1]),
                   "D_norm_cue": float(D[cue].norm()), "D_norm_total": float(D.norm())}
            for k in ("J", "R"):
                vs, vt = vec[k][l][0], vec[k][l][1]
                us, ut = vs / vs.norm(), vt / vt.norm()
                row[f"{k}_coef_src_cue"] = float(D[cue] @ us)
                row[f"{k}_coef_tgt_cue"] = float(D[cue] @ ut)
                row[f"{k}_cos_src_cue"] = float((D[cue] @ us) / D[cue].norm().clamp_min(1e-9))
                row[f"{k}_cos_tgt_cue"] = float((D[cue] @ ut) / D[cue].norm().clamp_min(1e-9))
                # coordinates of the true counterfactual in this lens's plane vs the exchange heuristic
                pinv = torch.linalg.pinv(vec[k][l].T)
                c_i = ci[names[l]][0, cue].float() @ pinv.T
                c_j = cj[names[l]][0, cue].float() @ pinv.T
                row[f"{k}_coord_src_clean"] = float(c_i[0]); row[f"{k}_coord_tgt_clean"] = float(c_i[1])
                row[f"{k}_coord_src_cf"] = float(c_j[0]); row[f"{k}_coord_tgt_cf"] = float(c_j[1])
                # fraction of ||D_cue||^2 inside this lens's plane
                q, _ = torch.linalg.qr(vec[k][l].T)
                row[f"{k}_frac_D_in_plane"] = float(((D[cue] @ q) ** 2).sum() / (D[cue] @ D[cue]).clamp_min(1e-9))
            align.write(json.dumps(row) + "\n")
        combos = {"J_s+J_t": ("J", "J"), "R_s+R_t": ("R", "R"), "R_s+J_t": ("R", "J"), "J_s+R_t": ("J", "R")}
        for arm, (ks, kt) in combos.items():
            for target_kind in ("exchange", "counterfactual"):
                hooks = []
                for l in BAND:
                    basis = torch.stack([vec[ks][l][0], vec[kt][l][1]])
                    pinv = torch.linalg.pinv(basis.T)
                    if target_kind == "exchange":
                        coords = ci[names[l]].float() @ pinv.T
                        hooks += clamp_hooks(model, basis, [l], {l: coords})
                    else:
                        coords_cf = cj[names[l]].float() @ pinv.T
                        hooks += clamp_hooks(model, basis, [l], {l: coords_cf}, exchange=False)
                logp = final_logprobs(model, ti, hooks)
                top1 = int(logp.argmax().item())
                causal.write(json.dumps({
                    "index": i, "partner": j, "name": ri.item.name, "arm": arm, "target": target_kind,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, logp),
                }) + "\n")
        print(f"[q52] pair {n + 1}/{len(pairs)} {ri.item.name}", flush=True)
    align.close(); causal.close()
    print("[block47] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
