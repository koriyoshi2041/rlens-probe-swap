#!/usr/bin/env python
"""Block 20: three experiments the findings so far demand.

S1 -- the control this project has been missing. Decomposing the true counterfactual
   activation difference showed that BOTH its lens-parallel and its lens-orthogonal
   halves flip the answer once rescaled to the full norm. If any large perturbation
   in this space flips the answer to the replacement, then "the lens direction is
   causally effective" means very little. The random controls run so far were all at
   the lens edit's own scale, where nothing happens. This one matches the scale of
   the full patch: a random direction, and another item's patch direction, both at
   the true difference's per-position norm. The prediction that would vindicate
   specificity is that these destroy the answer -- top-1 becomes neither word --
   rather than installing the replacement.

S2 -- did we rewrite a thought, or nudge one token? The whole project scores a
   single next token. If the clamp genuinely changes the model's state, the free
   continuation should stay in the rewritten world: say Rome and then keep talking
   about Italy. If it says Rome and reverts to France in the next clause, it was a
   surface blip. The clamp is held on the prompt positions only; generated tokens
   run unclamped, so the continuation is the model's own.

S3 -- who rebuilds the entity? Clamping five layers and reading the twelve above
   shows whether the coordinate climbs back, but not what puts it back. Splitting
   each layer's residual update into its attention half (resid_mid - resid_pre) and
   its MLP half (out - resid_mid), and projecting each onto the source direction,
   attributes the repair. Attention re-reading the prompt tokens and an MLP
   re-deriving the entity locally are different stories about what the "bridge
   variable" is.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks, swapped_fraction
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
PARTIAL = list(range(8, 13))
ABOVE = list(range(13, 21))
S1_LAYERS = [8, 12, 16]
S2_N = 12
S2_TOKENS = 24


def add_delta_hooks(model, deltas, stats=None):
    hooks = []
    for layer, delta in deltas.items():
        def transform(selected, delta=delta, layer=layer):
            d = delta.to(selected.device).float()
            h = selected.float()
            if stats is not None:
                stats.record(layer, h, d)
            return h + d
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def s1_scale_matched_controls(model, lenses, resolved, out_dir):
    pairs = json.loads((RESULTS_DIR / "block01" / "counterfactual_pairs.json").read_text(encoding="utf-8"))
    names = {layer: _resid_post_hook_name(layer) for layer in S1_LAYERS}
    wanted = set(names.values())
    handle = (out_dir / "s1_scale_matched.jsonl").open("w", encoding="utf-8")
    gen = torch.Generator(device="cpu").manual_seed(0)
    cached = {}
    for i, j in pairs:
        ri, rj = resolved[i], resolved[j]
        ti = model.to_tokens(ri.prompt, prepend_bos=False)
        tj = model.to_tokens(rj.prompt, prepend_bos=False)
        if ti.shape != tj.shape:
            continue
        _, ci = model.run_with_cache(ti, names_filter=lambda x: x in wanted)
        _, cj = model.run_with_cache(tj, names_filter=lambda x: x in wanted)
        cached[i] = {L: (cj[names[L]].float() - ci[names[L]].float()) for L in S1_LAYERS}
    keys = sorted(cached)
    for n, (i, j) in enumerate(pairs):
        if i not in cached:
            continue
        ri = resolved[i]
        ti = model.to_tokens(ri.prompt, prepend_bos=False)
        clean = final_logprobs(model, ti)
        a, s = ri.answer.first_id, ri.swap_answer.first_id
        other = keys[(keys.index(i) + 5) % len(keys)]
        for L in S1_LAYERS:
            true_delta = cached[i][L]
            norms = true_delta.norm(dim=-1, keepdim=True)
            rand = torch.randn(true_delta.shape, generator=gen).to(true_delta.device)
            rand = rand / rand.norm(dim=-1, keepdim=True) * norms
            shuffled = cached[other][L]
            if shuffled.shape != true_delta.shape:
                shuffled = None
            else:
                shuffled = shuffled / shuffled.norm(dim=-1, keepdim=True).clamp_min(1e-9) * norms
            arms = {"true_patch": true_delta, "random_matched": rand}
            if shuffled is not None:
                arms["other_item_patch_matched"] = shuffled
            for arm, delta in arms.items():
                stats = EnergyStats()
                logp = final_logprobs(model, ti, add_delta_hooks(model, {L: delta}, stats))
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": i, "partner": j, "name": ri.item.name, "layer": L, "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "delta_logp_answer": float(logp[a] - clean[a]),
                    "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "top1_str": model.tokenizer.decode([top1]),
                    "kl": kl_divergence(clean, logp), "dh_norm": float(stats.total_dh2 ** 0.5),
                }) + "\n")
        print(f"[S1] pair {n + 1}/{len(pairs)}", flush=True)
    handle.close()


@torch.inference_mode()
def s2_continuations(model, lenses, resolved, eligible, out_dir):
    rng = np.random.default_rng(0)
    chosen = sorted(rng.choice(eligible, size=min(S2_N, len(eligible)), replace=False).tolist())
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    rows = []
    for index in chosen:
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        n_prompt = tokens.shape[1]
        ids = r.tracked_ids[:2]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind in ("clean", "J", "R"):
            hooks = []
            if kind != "clean":
                lens = lenses[kind]
                for layer in BAND:
                    basis = lens.lens_vectors(model, ids, layer)
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords = cache[names[layer]].float() @ pinv.T
                    hooks += clamp_hooks(model, basis, [layer], {layer: coords},
                                         positions=list(range(n_prompt)))
            seq = tokens
            for _ in range(S2_TOKENS):
                logp = final_logprobs(model, seq, hooks)
                seq = torch.cat([seq, logp.argmax().view(1, 1)], dim=1)
            rows.append({
                "index": index, "name": r.item.name, "arm": kind, "prompt": r.prompt,
                "answer": r.answer.text, "swap_answer": r.swap_answer.text,
                "intermediate": r.item.intermediate, "swap_to": r.item.swap_to,
                "continuation": model.tokenizer.decode(seq[0, n_prompt:].tolist()),
            })
            print(f"[S2] {r.item.name:28s} {kind:5s} -> {rows[-1]['continuation']!r}", flush=True)
    (out_dir / "s2_continuations.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


@torch.inference_mode()
def s3_repair_attribution(model, lenses, resolved, eligible, out_dir):
    hook_names = {}
    for layer in ABOVE:
        hook_names[(layer, "pre")] = f"blocks.{layer}.hook_resid_pre"
        hook_names[(layer, "mid")] = f"blocks.{layer}.hook_resid_mid"
        hook_names[(layer, "out")] = f"blocks.{layer}.hook_out"
    wanted = set(hook_names.values()) | {_resid_post_hook_name(l) for l in PARTIAL}
    handle = (out_dir / "s3_repair.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        ids = r.tracked_ids[:2]
        _, clean_cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            bases, pinvs = {}, {}
            for layer in sorted(set(PARTIAL) | set(ABOVE)):
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinvs[layer] = torch.linalg.pinv(bases[layer].T.float())
            hooks = []
            for layer in PARTIAL:
                coords = clean_cache[_resid_post_hook_name(layer)].float() @ pinvs[layer].T
                hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords})
            probe = {}

            def watch(name):
                def fn(act, hook):
                    probe[name] = act.detach().float()
                    return act
                return fn

            with model.hooks(fwd_hooks=hooks + [(v, watch(v)) for v in hook_names.values()]):
                model(tokens, return_type="logits")
            rows = []
            for layer in ABOVE:
                pinv = pinvs[layer]
                pre = probe[hook_names[(layer, "pre")]] @ pinv.T
                mid = probe[hook_names[(layer, "mid")]] @ pinv.T
                out = probe[hook_names[(layer, "out")]] @ pinv.T
                clean_c = clean_cache[_resid_post_hook_name(layer)].float() @ pinv.T
                gap = clean_c[..., 1] - clean_c[..., 0]
                keep = gap.abs() >= gap.abs().median()

                def in_gap_units(delta_coord):
                    return float((delta_coord[keep] / gap[keep]).median().item())

                rows.append({
                    "layer": layer,
                    "attn_push_on_source": in_gap_units(mid[..., 0] - pre[..., 0]),
                    "mlp_push_on_source": in_gap_units(out[..., 0] - mid[..., 0]),
                    "attn_push_on_target": in_gap_units(mid[..., 1] - pre[..., 1]),
                    "mlp_push_on_target": in_gap_units(out[..., 1] - mid[..., 1]),
                    "swapped_fraction_out": swapped_fraction(out, clean_c),
                })
            handle.write(json.dumps({"index": index, "name": r.item.name, "lens": kind, "profile": rows}) + "\n")
        if n % 10 == 0:
            print(f"[S3] {n + 1}/{len(eligible)}", flush=True)
    handle.close()


def main() -> int:
    out_dir = results_dir("block20_specificity")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    s1_scale_matched_controls(model, lenses, resolved, out_dir)
    s2_continuations(model, lenses, resolved, eligible, out_dir)
    s3_repair_attribution(model, lenses, resolved, eligible, out_dir)
    print("[block20] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
