#!/usr/bin/env python
"""Block 75: a DONOR-FREE editing recipe — position by the transport head, content by the 2-D
lens clamp, routing by forcing the head's attention (no donor run anywhere).

Blocks 66/69/73: the routing-assisted clamp (donor keys at L19/L23 + 2-D clamp) reaches 0.96 on
flippable items, but the keys come from a donor run. Here the routing step uses only the
attention pattern: the final-row pattern of L23 h8 (or h8+h9+h0, or all L23 heads, or L19+L23)
is forced onto the edit position p with weight w (w = 1: one-hot; w = 0.8: 80% onto p, the rest
rescaled), and the 2-D clamp x4 is applied at p over L8-20. The edit position is p_h8, the
position h8 attends to most in the clean run (block 69), with p_best (lens readability) as the
comparison. Controls: forcing alone (clean content -> should push toward the original answer),
the clamp alone, and the donor-key version from block 66/69 as the reference ceiling.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q30_4b_replication import load_4b
from q69_key_value import capture, hook_name, patch_position
from q70_position_choice import clamp_at, donor_vector
from q82_third_model_mechanism import clamp_band
from rlens.data import load_items
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
FOURB = os.environ.get("MATS_MODEL", "9b") == "4b"  # block80: the same donor-free recipe on Qwen3.5-4B
THIRD = os.environ.get("MATS_MODEL", "9b") == "3rd"  # block83: Qwen3-4B (dense), band/positions from blocks 81/82, head from MATS_HEAD
HEAD = int(os.environ.get("MATS_HEAD", "8"))
ATTN_LAYERS = (19, 23)
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def force_pattern(position, heads, weight):
    """Final-row attention pattern of the given heads (None = all): weight onto `position`, rest rescaled by 1-weight."""
    def fn(tensor, hook):
        out = tensor.clone()
        hs = list(range(out.shape[1])) if heads is None else heads
        for h in hs:
            row = out[:, h, -1, :] * (1.0 - weight)
            row[:, position] = row[:, position] + weight
            out[:, h, -1, :] = row
        return out
    return fn


def forcing_hooks(position, spec, weight):
    if spec == "h8@23":
        return [(hook_name(23, "hook_pattern"), force_pattern(position, [HEAD], weight))]
    if spec == "h8h9h0@23":
        return [(hook_name(23, "hook_pattern"), force_pattern(position, [8, 9, 0], weight))]
    if spec == "all@23":
        return [(hook_name(23, "hook_pattern"), force_pattern(position, None, weight))]
    if spec == "all@19,23":
        return [(hook_name(l, "hook_pattern"), force_pattern(position, None, weight)) for l in ATTN_LAYERS]
    raise KeyError(spec)


def donor_vector_band(model, text, entity, wanted, names, band):
    """Donor residuals at the entity's last token for the given band (q70's helper is bound to L8-20)."""
    from q35_donor_paste import find_last
    from rlens.tokens import encode_variant
    tokens = model.to_tokens(text, prepend_bos=False)
    seq = tokens[0].tolist()
    pos = None
    for variant in (" " + entity, entity):
        pos = find_last(seq, list(encode_variant(model.tokenizer, variant).ids))
        if pos is not None:
            break
    if pos is None:
        return None
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {layer: cache[names[layer]][0, pos] for layer in band}


def run(model, tokens, hooks, position, s, a):
    wanted = {hook_name(23, "hook_pattern")}
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    top1 = int(lp.argmax().item())
    pat = cache[hook_name(23, "hook_pattern")][0, HEAD, -1, :].float()
    return {"margin": float(lp[s] - lp[a]), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
            "top1_str": model.tokenizer.decode([top1]), "h8_to_p": float(pat[position].item())}


@torch.inference_mode()
def main() -> int:
    band = BAND
    if THIRD:
        from q81_third_model import LENS_PATH, load_third
        from transformer_lens.tools.analysis import JacobianLens
        out_dir = results_dir("block83_qwen3_4b_donor_free_recipe")
        model = load_third()
        lens = JacobianLens.load(str(LENS_PATH))
        band = json.loads((RESULTS_DIR / "block81_qwen3_4b" / "meta.json").read_text(encoding="utf-8"))["band"]
        clean_rows = json.loads((RESULTS_DIR / "block81_qwen3_4b" / "clean_rows.json").read_text(encoding="utf-8"))
        best_pos = {json.loads(l)["index"]: int(json.loads(l)["position"]) for l in (RESULTS_DIR / "block82_qwen3_4b_mechanism" / "mechanism.jsonl").read_text().splitlines()}
    else:
        out_dir = results_dir("block80_4b_donor_free_recipe" if FOURB else "block75_donor_free_recipe")
        if FOURB:
            model, lenses = load_4b()
            lens = lenses["J"]
        else:
            model = load_model()
            lens = load_lenses()["J"]
        clean_rows = json.loads((RESULTS_DIR / ("block27_4b" if FOURB else "block01") / "clean_rows.json").read_text(encoding="utf-8"))
        best_pos = {}
        for line in (RESULTS_DIR / ("block35_4b_donor_paste" if FOURB else "block31_donor_paste") / "donor_paste.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["position_set"] == "best" and r["arm"] == "clamp2d":
                best_pos[r["index"]] = int(r["position"])
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in band}
    print(f"[q75] model={'3rd' if THIRD else ('4b' if FOURB else '9b')} band={band[0]}-{band[-1]} head=L23 h{HEAD} items={len(eligible)}", flush=True)
    wanted = set(names.values()) | {hook_name(23, "hook_pattern")}
    handle = (out_dir / "donor_free.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        T = int(tokens.shape[1])
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        if index not in best_pos or str(index) not in templates:
            continue
        p_best = best_pos[index]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        pat = cache[hook_name(23, "hook_pattern")][0, 8, -1, :].float()
        p_h8 = int(pat[: T - 1].argmax().item())
        bases = {l: lens.lens_vectors(model, ids, l) for l in band}
        coords = {l: cache[names[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in band}
        donor = donor_vector_band(model, templates[str(index)].format(r.item.swap_to), r.item.swap_to, wanted, names, band)
        rec = {"index": index, "name": r.item.name, "n_prompt": T, "p_best": p_best, "p_h8": p_h8, "h8_clean_at_h8": float(pat[p_h8]), "h8_clean_at_best": float(pat[p_best]), "arms": {}}
        arms = {"clean": []}
        for tag, p in (("h8", p_h8), ("best", p_best)):
            c4 = clamp_band(model, bases, coords, band, [p], 4.0)
            arms[f"clamp@1_{tag}"] = clamp_band(model, bases, coords, band, [p], 1.0)
            arms[f"clamp@4_{tag}"] = c4
            for spec in (("h8@23", "all@23", "all@19,23") if THIRD else ("h8@23", "h8h9h0@23", "all@23", "all@19,23")):
                for w in ((1.0, 0.8) if spec == "h8@23" else (1.0,)):
                    arms[f"F[{spec},{w:g}]+clamp@4_{tag}"] = forcing_hooks(p, spec, w) + c4
            arms[f"F[h8@23,1]_{tag}"] = forcing_hooks(p, "h8@23", 1.0)
            arms[f"F[all@23,1]_{tag}"] = forcing_hooks(p, "all@23", 1.0)
            if donor is not None:
                from q66_energy_of_arms import scaled_paste_hooks
                S = capture(model, tokens, scaled_paste_hooks(model, donor, band, p, 0.25, bases, "orth"))
                arms[f"K<-donor+clamp@4_{tag}"] = [(hook_name(l, "hook_k"), patch_position(S[hook_name(l, "hook_k")], p)) for l in ATTN_LAYERS] + c4
        arms["clamp@4_all"] = clamp_band(model, bases, coords, band, list(range(T)), 4.0)
        arms["F[h8@23,1]_h8+clamp@4_all"] = forcing_hooks(p_h8, "h8@23", 1.0) + arms["clamp@4_all"]
        results = {}
        for arm, hooks in arms.items():
            p = p_h8 if arm.endswith("_h8") or "_h8+" in arm or arm.endswith("_all") else p_best
            results[arm] = run(model, tokens, hooks, p, s, a)
        for arm in results:
            results[arm]["delta_margin"] = results[arm]["margin"] - results["clean"]["margin"]
        rec["arms"] = results
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 10 == 0:
            print(f"[q75] {n + 1}/{len(eligible)} p_h8={p_h8} " + " ".join(f"{k}={v['delta_margin']:+.2f}" for k, v in results.items() if k in ("clamp@4_h8", "F[h8@23,1]+clamp@4_h8", "F[all@23,1]+clamp@4_h8", "K<-donor+clamp@4_h8")), flush=True)
    handle.close()
    print("[block83] done" if THIRD else ("[block80] done" if FOURB else "[block75] done"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
