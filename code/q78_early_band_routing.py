#!/usr/bin/env python
"""Block 78: is R's early-band advantage a routing effect?

Blocks 48/52/61: the R clamp over L3-8 (all positions) flips 0.25 vs J 0.16 at scale 1, the
advantage survives self-consistent energy matching (+0.86), and no single mechanism explains
it. Block 66/69/73: for the working band, the complement of the plane carries a key-side
signal that redirects L23 h8; the 2-D clamp does not. Question: does the R early clamp redirect
the transport head where the J early clamp does not?  For every item, under J@1, R@1, J@2
(energy-matched to R, block 61) and R@2 early clamps at all positions:
  - L23 h8's attention onto p_best and p_h8 and its argmax position, L19/L23 all-head sums;
  - K/V of the clamped run transplanted into the clean run at p_best and p_h8 (L19/L23):
    the margin/flip they alone reproduce, and the attention outputs Z at the final position;
  - the clamp's own margin/flip.
If R's clamp raises h8's attention onto the entity positions more than J's, and the KV
transplant from the R run reproduces more of the effect, R's early advantage is (partly) routing.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q69_key_value import capture, hook_name, patch_position
from q70_position_choice import clamp_at
from rlens.data import load_items
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

EARLY = list(range(3, 9))
BAND = list(range(8, 21))
ATTN_LAYERS = (19, 23)


def run(model, tokens, hooks, p_best, p_h8, s, a):
    wanted = {hook_name(l, "hook_pattern") for l in ATTN_LAYERS}
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    top1 = int(lp.argmax().item())
    pat23 = cache[hook_name(23, "hook_pattern")][0, :, -1, :].float()
    pat19 = cache[hook_name(19, "hook_pattern")][0, :, -1, :].float()
    T = tokens.shape[1]
    return {"margin": float(lp[s] - lp[a]), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
            "h8_best": float(pat23[8, p_best]), "h8_h8pos": float(pat23[8, p_h8]), "h8_argmax": int(pat23[8, : T - 1].argmax().item()),
            "h8_final": float(pat23[8, -1]), "L23_sum_best": float(pat23[:, p_best].sum()), "L19_sum_best": float(pat19[:, p_best].sum()),
            "L23_sum_h8pos": float(pat23[:, p_h8].sum())}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block78_early_band_routing")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    enames = {layer: _resid_post_hook_name(layer) for layer in EARLY}
    wanted = set(enames.values()) | {hook_name(23, "hook_pattern")}
    handle = (out_dir / "early_routing.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        T = int(tokens.shape[1])
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        p_best = best_pos[index]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        pat = cache[hook_name(23, "hook_pattern")][0, 8, -1, :].float()
        p_h8 = int(pat[: T - 1].argmax().item())
        arms = {"clean": []}
        for kind in ("J", "R"):
            bases = {l: lenses[kind].lens_vectors(model, ids, l) for l in EARLY}
            coords = {l: cache[enames[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in EARLY}
            for sc in (1.0, 2.0):
                hooks = []
                for l in EARLY:
                    c = coords[l]
                    target = c + sc * (c[..., [1, 0]] - c)
                    from rlens.interventions import clamp_hooks
                    hooks += clamp_hooks(model, bases[l], [l], {l: target}, exchange=False, positions=list(range(T)))
                arms[f"{kind}@{sc:g}"] = hooks
        results = {arm: run(model, tokens, hooks, p_best, p_h8, s, a) for arm, hooks in arms.items()}
        for arm in ("J@1", "R@1", "J@2", "R@2"):
            S = capture(model, tokens, arms[arm])
            for tag, p in (("best", p_best), ("h8", p_h8)):
                kv = [(hook_name(l, k), patch_position(S[hook_name(l, k)], p)) for l in ATTN_LAYERS for k in ("hook_k", "hook_v")]
                results[f"KV<-{arm}_{tag}"] = run(model, tokens, kv, p_best, p_h8, s, a)
            z = [(hook_name(l, "hook_z"), patch_position(S[hook_name(l, "hook_z")], T - 1)) for l in ATTN_LAYERS]
            results[f"Z<-{arm}"] = run(model, tokens, z, p_best, p_h8, s, a)
        for arm in results:
            results[arm]["delta_margin"] = results[arm]["margin"] - results["clean"]["margin"]
        handle.write(json.dumps({"index": index, "name": r.item.name, "n_prompt": T, "p_best": p_best, "p_h8": p_h8, "arms": results}) + "\n")
        if n % 10 == 0:
            print(f"[q78] {n + 1}/{len(eligible)} " + " ".join(f"{k}={v['delta_margin']:+.2f}/h8b {v['h8_best']:.2f}" for k, v in results.items() if k in ("J@1", "R@1", "J@2")), flush=True)
    handle.close()
    print("[block78] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
