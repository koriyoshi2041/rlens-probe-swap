#!/usr/bin/env python
"""Block 11: three mechanism tests, each aimed at a claim we cannot yet support.

M1. Does the Jacobian transport specifically entangle an entity pair with its
    answer pair, or does it contract angles between any two directions? The
    measured cosine between the entity-difference and the answer-contrast rises
    from 0.143 in the unembedding basis to 0.28 after transport. That is only
    interesting if the same map leaves unrelated pairs alone. Control: the same
    statistic for random token pairs and for shuffled (entity, answer) pairings.

M2. Why is R better than J in the early band? The hypothesis that R's direction
    is better aligned with the true counterfactual activation difference was
    tested against the patching data and rejected -- R's alignment is equal at L6
    and significantly worse from L8 on. The remaining hypothesis is that clamping
    in R's coordinates leaves an internal state that reads more cleanly as the
    replacement entity. That is testable by cross-reading the early-band state.

M3. Is the involution's weakness really the layer-to-layer alternation? If so the
    behavioural effect should depend on the parity of the band width, and the
    dependence should vanish for the idempotent clamp. Sweeping widths 1..13 at
    alpha = 1 tests exactly that.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

M1_LAYERS = [6, 12, 18]
M3_WIDTHS = list(range(1, 14))
M3_START = 8


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a[None].float(), b[None].float()).item())


@torch.inference_mode()
def m1_entanglement(model, lenses, resolved, eligible, out_dir):
    """Is the transport's angle contraction specific to related pairs?"""
    wu = model.W_U.float()
    d_vocab = wu.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(0)
    rand_ids = torch.randint(0, d_vocab, (4, 200), generator=gen)
    items = [resolved[i] for i in eligible]
    rows = []
    for kind, lens in lenses.items():
        for layer in M1_LAYERS:
            matrix = lens.jacobians[layer].to(device=wu.device, dtype=torch.float32).T
            real_before, real_after = [], []
            shuf_before, shuf_after = [], []
            for n, r in enumerate(items):
                other = items[(n + 7) % len(items)]
                ent = wu[:, r.tgt.first_id] - wu[:, r.src.first_id]
                ans = wu[:, r.swap_answer.first_id] - wu[:, r.answer.first_id]
                oth = wu[:, other.swap_answer.first_id] - wu[:, other.answer.first_id]
                real_before.append(cos(ent, ans)); real_after.append(cos(matrix @ ent, matrix @ ans))
                shuf_before.append(cos(ent, oth)); shuf_after.append(cos(matrix @ ent, matrix @ oth))
            rb, ra = [], []
            for k in range(rand_ids.shape[1]):
                a = wu[:, int(rand_ids[0, k])] - wu[:, int(rand_ids[1, k])]
                b = wu[:, int(rand_ids[2, k])] - wu[:, int(rand_ids[3, k])]
                rb.append(cos(a, b)); ra.append(cos(matrix @ a, matrix @ b))
            rows.append({
                "lens": kind, "layer": layer,
                "real_before": float(np.median(np.abs(real_before))), "real_after": float(np.median(np.abs(real_after))),
                "shuffled_before": float(np.median(np.abs(shuf_before))), "shuffled_after": float(np.median(np.abs(shuf_after))),
                "random_before": float(np.median(np.abs(rb))), "random_after": float(np.median(np.abs(ra))),
                "real_before_signed": float(np.median(real_before)), "real_after_signed": float(np.median(real_after)),
            })
            print(f"[M1] {kind} L{layer:02d} |cos| 真实对 {rows[-1]['real_before']:.3f}->{rows[-1]['real_after']:.3f}  "
                  f"错配对 {rows[-1]['shuffled_before']:.3f}->{rows[-1]['shuffled_after']:.3f}  "
                  f"随机 token 对 {rows[-1]['random_before']:.3f}->{rows[-1]['random_after']:.3f}", flush=True)
    (out_dir / "m1_entanglement.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


@torch.inference_mode()
def m2_early_state(model, lenses, resolved, eligible, out_dir):
    """After clamping in the early band, which entity does the other lens read?"""
    band = cfg.BANDS["early_L3_8"]
    names = {layer: _resid_post_hook_name(layer) for layer in band}
    wanted = set(names.values())
    handle = (out_dir / "m2_early_state.jsonl").open("w", encoding="utf-8")
    read_layers = list(range(3, 25))
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            bases, coords = {}, {}
            for layer in band:
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            hooks = []
            stats = EnergyStats()
            for layer in band:
                hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats)
            reader = "R" if kind == "J" else "J"
            with model.hooks(fwd_hooks=hooks):
                ranks, _ = rank_readout(model, lenses[reader], tokens, r.tracked_ids, read_layers)
            best = ranks[:-1].min(dim=1).values.numpy()
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "clamped_with": kind, "reader": reader,
                "layers": read_layers, "best_intermediate": best[:, 0].tolist(),
                "best_swap_to": best[:, 1].tolist(), "total_dh2": stats.total_dh2,
            }) + "\n")
        if n % 15 == 0:
            print(f"[M2] {n + 1}/{len(eligible)}", flush=True)
    handle.close()


@torch.inference_mode()
def m3_parity(model, lenses, resolved, eligible, out_dir):
    """Band width 1..13: does the involution's effect depend on parity, and the clamp's not?"""
    handle = (out_dir / "m3_parity.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolve = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        for width in M3_WIDTHS:
            layers = list(range(M3_START, M3_START + width))
            names = {layer: _resid_post_hook_name(layer) for layer in layers}
            wanted = set(names.values())
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            for kind, lens in lenses.items():
                bases, coords = {}, {}
                for layer in layers:
                    bases[layer] = lens.lens_vectors(model, ids, layer)
                    pinv = torch.linalg.pinv(bases[layer].T.float())
                    coords[layer] = cache[names[layer]].float() @ pinv.T
                for arm in ("involution", "clamp"):
                    stats = EnergyStats()
                    hooks = []
                    for layer in layers:
                        if arm == "clamp":
                            hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats)
                        else:
                            hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    top1 = int(logp.argmax().item())
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "width": width, "lens": kind, "arm": arm,
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "top1_is_swap": top1 == s, "kl": kl_divergence(clean, logp),
                        "total_dh2": stats.total_dh2,
                    }) + "\n")
        if n % 10 == 0:
            print(f"[M3] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()


def main() -> int:
    out_dir = results_dir("block11_mechanism")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    m1_entanglement(model, lenses, resolved, eligible, out_dir)
    m3_parity(model, lenses, resolved, eligible, out_dir)
    m2_early_state(model, lenses, resolved, eligible, out_dir)
    print("[block11] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
