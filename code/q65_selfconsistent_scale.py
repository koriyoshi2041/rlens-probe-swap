#!/usr/bin/env python
"""Block 60: (a) self-consistent energy matching of R vs J in the early band (audit M-7); (b) clean-run
mean-ablation of the transfer stage (audit C15 control).

(a) A real clamp with a global scale s (coordinates pushed s of the way to the exchange) at
L3-8: R at s chosen per item so that its total energy equals J's at s=1; J at s = 1.5 and 2.0
(to reach R's KL). Flip rates and paired Δmargin vs the J clamp are recorded.
(b) In the CLEAN run (no clamp), the attention outputs of L17-24, or the MLP outputs of
L26-29, are mean-ablated at the final position (replaced by their mean over prompt positions);
two-hop accuracy vs single-hop accuracy tells whether the stage the clamp needs is also the
stage the model's own two-hop needs.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from q44_stage_necessity import sublayer_hook_name
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import resolve_next_token

EARLY = list(range(3, 9))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def capturing_scaled_clamp(model, bases, coords, layers, scale, store):
    hooks = []
    for layer in layers:
        matrix = bases[layer].T.float(); pinv = torch.linalg.pinv(matrix)
        c = coords[layer].float(); target = c + scale * (c[..., [1, 0]] - c)

        def transform(selected, matrix=matrix, pinv=pinv, target=target, layer=layer):
            h = selected.float(); cc = h @ pinv.to(h.device).T
            delta = (target.to(h.device) - cc) @ matrix.to(h.device).T
            store[layer] = delta.detach().clone(); return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def mean_ablate(names_list, position):
    def fn(act, hook):
        out = act.clone(); out[:, position, :] = act[:, :, :].mean(dim=1); return out
    return [(n, fn) for n in names_list]


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block60_selfconsistent")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in EARLY}
    attn_names = [sublayer_hook_name(model, l, "attn") for l in range(17, 25)]
    mlp_names = [f"blocks.{l}.hook_mlp_out" for l in range(26, 30)]
    handle = (out_dir / "selfconsistent.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in set(names.values()))
        clean = final_logprobs(model, tokens); base = float(clean[s] - clean[a])
        # (a)
        vec = {k: {l: lenses[k].lens_vectors(model, ids, l) for l in EARLY} for k in ("J", "R")}
        coords = {k: {l: cache[names[l]].float() @ torch.linalg.pinv(vec[k][l].T.float()).T for l in EARLY} for k in ("J", "R")}
        energy = {}
        for k in ("J", "R"):
            store = {}
            logp = final_logprobs(model, tokens, capturing_scaled_clamp(model, vec[k], coords[k], EARLY, 1.0, store))
            energy[k] = sum(float((store[l] ** 2).sum()) for l in EARLY)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({"index": index, "name": r.item.name, "part": "a", "arm": f"{k}@1", "scale": 1.0, "energy": energy[k],
                                     "delta_margin": float((logp[s] - logp[a]) - base), "top1_is_swap": top1 == s, "kl": kl_divergence(clean, logp)}) + "\n")
        s_R = (energy["J"] / max(energy["R"], 1e-9)) ** 0.5
        for k, sc in (("R", s_R), ("J", 1.5), ("J", 2.0), ("R", 0.8)):
            store = {}
            logp = final_logprobs(model, tokens, capturing_scaled_clamp(model, vec[k], coords[k], EARLY, sc, store))
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({"index": index, "name": r.item.name, "part": "a", "arm": f"{k}@{sc:.3g}" if k == "J" or sc == 0.8 else "R@energy_of_J", "scale": sc,
                                     "energy": sum(float((store[l] ** 2).sum()) for l in EARLY),
                                     "delta_margin": float((logp[s] - logp[a]) - base), "top1_is_swap": top1 == s, "kl": kl_divergence(clean, logp)}) + "\n")
        # (b) clean-run mean ablation of the transfer stage: two-hop prompt and single-hop prompt
        prompts = {"two_hop": (r.prompt, a)}
        sh = templates[str(index)].format(r.item.intermediate)
        prompts["single_hop"] = (sh, resolve_next_token(model.tokenizer, sh, r.item.answer).first_id)
        for kind, (text, ans) in prompts.items():
            t = model.to_tokens(text, prepend_bos=False); last = t.shape[1] - 1
            lp0 = final_logprobs(model, t)
            for group, hooks in (("attn_17_24", mean_ablate(attn_names, last)), ("mlp_26_29", mean_ablate(mlp_names, last)), ("attn_9_16", mean_ablate([sublayer_hook_name(model, l, "attn") for l in range(9, 17)], last))):
                lp = final_logprobs(model, t, hooks)
                handle.write(json.dumps({"index": index, "name": r.item.name, "part": "b", "prompt_kind": kind, "arm": group,
                                         "clean_correct": int(lp0.argmax().item()) == ans, "still_correct": int(lp.argmax().item()) == ans,
                                         "delta_logp_answer": float(lp[ans] - lp0[ans]), "top1_str": model.tokenizer.decode([int(lp.argmax().item())])}) + "\n")
        if n % 10 == 0:
            print(f"[q65] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block60] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
