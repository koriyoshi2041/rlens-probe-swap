#!/usr/bin/env python
"""Block 27: second-model replication (Qwen3.5-4B, matched J/R lenses, same 90 items).

Why. Every method-level claim so far rests on one model. The three claims that a
reviewer would most want to see on a second model are cheap to re-run and do not
depend on any 9B-specific choice:

  1. involution (published swap formula, alpha=1) vs idempotent clamp on the band;
  2. the band-width parity signature (odd widths work, even widths cancel) for the
     involution but not for the clamp;
  3. the fraction of the clamp effect that survives projecting out the first-order
     answer-contrast direction (the "direct push" control).

Everything is decided by the same pre-registered rules as on 9B: hybrid prompt
mode per item; eligibility = clean-correct + single-token concepts + single,
distinct answer tokens; band = the longest contiguous run of layers where BOTH
lenses read the intermediate into the top-10 for >= 60% of items. The last lens
layer (identity) is never used.

Paths are taken from environment variables so the 9B package stays untouched:
  MATS_4B_MODEL   (default $MATS_ROOT/assets/Qwen3.5-4B)
  MATS_4B_LENSES  (default .../assets/workspace-lenses/qwen3.5-4b)
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks, logit_lens_vectors
from rlens.model import lens_summary, model_summary
from rlens.paths import results_dir
from rlens.stages import clean, readout_scan
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

from rlens.paths import MATS_ROOT as ROOT  # set MATS_ROOT; see rlens/paths.py
MODEL_DIR = pathlib.Path(os.environ.get("MATS_4B_MODEL", ROOT / "assets" / "Qwen3.5-4B"))
LENS_DIR = pathlib.Path(os.environ.get("MATS_4B_LENSES", ROOT / "assets" / "workspace-lenses" / "qwen3.5-4b"))
HIT_RULE = 0.60


def load_4b():
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import JacobianLens

    model = TransformerBridge.boot_transformers(str(MODEL_DIR), device="cuda", dtype=torch.bfloat16)
    model.eval()
    lenses = {k: JacobianLens.load(str(LENS_DIR / f"{k.lower()}-lens" / "lens.pt")) for k in ("J", "R")}
    for k, lens in lenses.items():
        lens.validate_model(model)
    return model, lenses


def pick_band(summary):
    """Longest contiguous run of layers where both J and R hit10_any >= HIT_RULE; excludes the last (identity) layer."""
    layers = summary["layers"][:-1]  # last row is the model's own output
    j = summary["per_lens"]["J"]["intermediate"]["hit10_any_pos"]
    r = summary["per_lens"]["R"]["intermediate"]["hit10_any_pos"]
    ok = [(j[i] >= HIT_RULE and r[i] >= HIT_RULE) for i in range(len(layers))]
    best, cur = [], []
    for layer, flag in zip(layers, ok):
        if flag and layer < max(layers):  # never use the identity layer
            cur.append(layer)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    return best


def record(handle, base, logp, clean_logp, a, s, stats, model):
    top1 = int(logp.argmax().item())
    handle.write(json.dumps({
        **base,
        "delta_margin": float((logp[s] - logp[a]) - (clean_logp[s] - clean_logp[a])),
        "delta_logp_answer": float(logp[a] - clean_logp[a]),
        "delta_logp_swap_answer": float(logp[s] - clean_logp[s]),
        "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
        "top1_str": model.tokenizer.decode([top1]),
        "kl": kl_divergence(clean_logp, logp), "total_dh2": stats.total_dh2,
    }) + "\n")


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block27_4b")
    t0 = time.time()
    model, lenses = load_4b()
    header = {"model": model_summary(model), "lenses": {k: lens_summary(v) for k, v in lenses.items()}}
    print("[4b] header:", json.dumps(header, indent=1), flush=True)
    items = load_items()

    clean_rows = clean.run(model, items, out_dir)
    summary = readout_scan.run(model, lenses, items, clean_rows, out_dir)
    band = pick_band(summary)
    override = os.environ.get("MATS_4B_BAND")  # e.g. "8-20": the 9B band, for a depth-matched arm
    if override:
        lo, hi = (int(x) for x in override.split("-"))
        band = list(range(lo, hi + 1))
        out_dir = results_dir(f"block27_4b_band{lo}_{hi}")
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    meta = {**header, "band": band, "n_eligible": len(eligible), "eligible": eligible,
            "hit_rule": HIT_RULE, "load_seconds": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[4b] band={band} eligible={len(eligible)}", flush=True)
    if len(band) < 2:
        raise SystemExit("band too short; inspect readout_summary.json")

    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    widths = list(range(1, len(band) + 1))
    handle = (out_dir / "replication.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean_logp = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        names = {layer: _resid_post_hook_name(layer) for layer in band}
        wanted = set(names.values())
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        contrast = model.W_U[:, s].float() - model.W_U[:, a].float()
        base = {"index": index, "name": r.item.name, "category": r.item.category}
        for kind in ("J", "R", "logit"):
            bases, coords, units = {}, {}, {}
            for layer in band:
                if kind == "logit":
                    bases[layer] = logit_lens_vectors(model, ids)
                    units[layer] = contrast
                else:
                    bases[layer] = lenses[kind].lens_vectors(model, ids, layer)
                    matrix = lenses[kind].jacobians[layer].to(device=contrast.device, dtype=torch.float32)
                    units[layer] = matrix.T @ contrast
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            # (1) full band: involution vs clamp, plus (3) direct-push control on both
            for arm in ("involution", "clamp"):
                for control in ("full", "ortho_rescaled"):
                    if kind == "logit" and control != "full":
                        continue
                    stats = EnergyStats()
                    hooks = []
                    for layer in band:
                        ortho = units[layer] if control == "ortho_rescaled" else None
                        if arm == "clamp":
                            hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats,
                                                 orthogonalize_to=ortho, rescale_after_orthogonalize=ortho is not None)
                        else:
                            hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats,
                                                          orthogonalize_to=ortho, rescale_after_orthogonalize=ortho is not None)
                    logp = final_logprobs(model, tokens, hooks)
                    record(handle, {**base, "stage": "band", "lens": kind, "arm": arm, "control": control,
                                    "width": len(band)}, logp, clean_logp, a, s, stats, model)
            # (2) width ladder from the band start
            if kind == "logit":
                continue
            for width in widths:
                layers = band[:width]
                for arm in ("involution", "clamp"):
                    stats = EnergyStats()
                    hooks = []
                    for layer in layers:
                        if arm == "clamp":
                            hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats)
                        else:
                            hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    record(handle, {**base, "stage": "ladder", "lens": kind, "arm": arm, "control": "full",
                                    "width": width}, logp, clean_logp, a, s, stats, model)
        if n % 5 == 0:
            print(f"[4b] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    print("[block27] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
