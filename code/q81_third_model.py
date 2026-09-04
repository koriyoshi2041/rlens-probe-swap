#!/usr/bin/env python
"""Block 81: a THIRD model, standard dense attention — Qwen3-4B (36 layers, d_model 2560, no linear attention).

Both models so far (Qwen3.5-9B, Qwen3.5-4B) are hybrid gated-delta/attention models from one
training recipe. This block fits a J-lens on Qwen3-4B with the reference estimator
(JacobianLens.fit: 25 pile-10k prompts, skip_first=4, max 128 tokens — the published artifact's
convention, block 34) and reruns the methods-level replication of block 27 with J only (the R
estimator is not available in the library): clean eligibility, readout scan, band pick (J
hit10 >= 0.60), band arms (involution vs clamp, full vs orthogonalised), width ladder for the
parity signature, plus the logit-lens baseline. Predictions (pre-registered here): the
involution's flip rate alternates with band width parity and the clamp does not; clamp > involution
on the full band; the orthogonalised clamp retains most of the effect.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q30_4b_replication import HIT_RULE, record
from q37_refit_pile import load_pile
from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks, logit_lens_vectors
from rlens.model import lens_summary, model_summary
from rlens.paths import results_dir
from rlens.stages import clean, readout_scan
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

from rlens.paths import MATS_ROOT as ROOT  # set MATS_ROOT; see rlens/paths.py
MODEL_DIR = pathlib.Path(os.environ.get("MATS_3RD_MODEL", ROOT / "assets" / "Qwen3-4B"))
LENS_PATH = pathlib.Path(os.environ.get("MATS_3RD_LENS", ROOT / "assets" / "workspace-lenses" / "qwen3-4b" / "j-lens" / "lens.pt"))
N_PROMPTS = int(os.environ.get("MATS_3RD_NPROMPTS", "25"))
DTYPE = torch.float32 if os.environ.get("MATS_3RD_DTYPE", "bf16") == "float32" else torch.bfloat16
OUT_NAME = os.environ.get("MATS_3RD_OUT", "block81_qwen3_4b")


def load_third():
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers(str(MODEL_DIR), device="cuda", dtype=DTYPE)
    model.eval()
    return model


def fit_or_load_lens(model):
    from transformer_lens.tools.analysis import JacobianLens

    if LENS_PATH.exists():
        lens = JacobianLens.load(str(LENS_PATH))
        lens.validate_model(model)
        print(f"[q81] loaded lens {LENS_PATH}", flush=True)
        return lens
    docs = load_pile(N_PROMPTS)
    t0 = time.time()
    lens = JacobianLens.fit(model, docs, corpus="NeelNanda/pile-10k", source_layers=list(range(0, model.cfg.n_layers - 1)),
                            dim_batch=32, max_seq_len=128, skip_first_positions=4, show_progress=False)  # metadata must not touch the fit's provenance keys
    print(f"[q81] lens fitted in {(time.time() - t0) / 60:.1f} min", flush=True)
    LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(lens, "save"):
        lens.save(str(LENS_PATH))
    else:
        torch.save(lens, str(LENS_PATH.with_suffix(".pkl")))
    return lens


def pick_band_j(summary):
    layers = summary["layers"][:-1]
    j = summary["per_lens"]["J"]["intermediate"]["hit10_any_pos"]
    best, cur = [], []
    for i, layer in enumerate(layers):
        if j[i] >= HIT_RULE and layer < max(layers):
            cur.append(layer)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    return best


def main() -> int:
    out_dir = results_dir(OUT_NAME)
    t0 = time.time()
    model = load_third()
    lens = fit_or_load_lens(model)
    lenses = {"J": lens}
    header = {"model": model_summary(model), "lenses": {"J": lens_summary(lens)}}
    print("[q81] header:", json.dumps(header, indent=1), flush=True)
    items = load_items()
    with torch.inference_mode():
        clean_rows = clean.run(model, items, out_dir)
        summary = readout_scan.run(model, lenses, items, clean_rows, out_dir)
        band = pick_band_j(summary)
        override = os.environ.get("MATS_3RD_BAND")
        if override:
            lo, hi = (int(x) for x in override.split("-"))
            band = list(range(lo, hi + 1))
        eligible = eligible_hybrid(clean_rows)
        modes = hybrid_rows(clean_rows)
        meta = {**header, "band": band, "n_eligible": len(eligible), "eligible": eligible, "hit_rule": HIT_RULE, "seconds_to_here": round(time.time() - t0, 1)}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[q81] band={band} eligible={len(eligible)}", flush=True)
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
            for kind in ("J", "logit"):
                bases, coords, units = {}, {}, {}
                for layer in band:
                    if kind == "logit":
                        bases[layer] = logit_lens_vectors(model, ids)
                        units[layer] = contrast
                    else:
                        bases[layer] = lens.lens_vectors(model, ids, layer)
                        matrix = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32)
                        units[layer] = matrix.T @ contrast
                    pinv = torch.linalg.pinv(bases[layer].T.float())
                    coords[layer] = cache[names[layer]].float() @ pinv.T
                for arm in ("involution", "clamp"):
                    for control in ("full", "ortho_rescaled"):
                        if kind == "logit" and control != "full":
                            continue
                        stats = EnergyStats()
                        hooks = []
                        for layer in band:
                            ortho = units[layer] if control == "ortho_rescaled" else None
                            if arm == "clamp":
                                hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats, orthogonalize_to=ortho, rescale_after_orthogonalize=ortho is not None)
                            else:
                                hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats, orthogonalize_to=ortho, rescale_after_orthogonalize=ortho is not None)
                        logp = final_logprobs(model, tokens, hooks)
                        record(handle, {**base, "stage": "band", "lens": kind, "arm": arm, "control": control, "width": len(band)}, logp, clean_logp, a, s, stats, model)
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
                        record(handle, {**base, "stage": "ladder", "lens": kind, "arm": arm, "control": "full", "width": width}, logp, clean_logp, a, s, stats, model)
            if n % 10 == 0:
                print(f"[q81] {n + 1}/{len(eligible)}", flush=True)
        handle.close()
    print(f"[block81] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
