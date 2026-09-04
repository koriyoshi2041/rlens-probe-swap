#!/usr/bin/env python
"""Block 45: layer-resolved necessity - restore ONE attention layer (L17-24) or ONE MLP layer (L24-29) at a time.

Block 30 decomposed the new-answer signal at the final position into an attention half
that appears at L17-24 and an MLP half at L26-29. That is a decomposition, not a
necessity claim. Here the clamp stays on (J-lens, L8-20, all positions) and one sublayer
group's OUTPUT is restored to its clean-run value, either at the final position only or at
all positions:

  attn_17_24     restore attention / linear-attention outputs of L17-24
  mlp_26_29      restore MLP outputs of L26-29
  attn_9_16      control: attention outputs inside the band (should matter little)
  mlp_17_24      control: MLP outputs of L17-24
  attn_25_31     late attention (L29 also contributed)
  mlp_21_25      MLP outputs between the two stages

If restoring attn_17_24 at the final position alone removes most flips while the controls
do not, the transfer through those layers is necessary for the rewrite to reach the
answer; likewise for the late MLPs.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
GROUPS = {f"attn_L{l}": ("attn", [l]) for l in range(17, 25)}
GROUPS.update({f"mlp_L{l}": ("mlp", [l]) for l in range(24, 30)})
GROUPS["attn_17_24"] = ("attn", list(range(17, 25)))


def sublayer_hook_name(model, layer, kind):
    if kind == "mlp":
        return f"blocks.{layer}.hook_mlp_out"
    for cand in (f"blocks.{layer}.hook_attn_out", f"blocks.{layer}.linear_attn.hook_out", f"blocks.{layer}.attn.hook_out"):
        if cand in model.hook_dict:
            return cand
    raise KeyError(f"no attention output hook for layer {layer}")


def restore_hooks(clean_values, positions):
    hooks = []
    for name, value in clean_values.items():
        def fn(act, hook, value=value):
            out = act.clone()
            if positions is None:
                out[:] = value.to(act.dtype)
            else:
                out[:, positions, :] = value[:, positions, :].to(act.dtype)
            return out
        hooks.append((name, fn))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block45_layer_necessity")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    band_names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    group_names = {g: {layer: sublayer_hook_name(model, layer, kind) for layer in layers} for g, (kind, layers) in GROUPS.items()}
    all_sub = {n for d in group_names.values() for n in d.values()}
    wanted = set(band_names.values()) | all_sub
    print("[q44] hooks:", sorted(all_sub)[:6], "...", flush=True)
    handle = (out_dir / "stage_necessity.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clamp = []
        for layer in BAND:
            basis = lens.lens_vectors(model, ids, layer)
            pinv = torch.linalg.pinv(basis.T.float())
            coords = cache[band_names[layer]].float() @ pinv.T
            clamp += clamp_hooks(model, basis, [layer], {layer: coords})
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        arms = {"clamp_only": []}
        for g, names in group_names.items():
            values = {name: cache[name].float() for name in names.values()}
            arms[f"{g}@final"] = restore_hooks(values, [tokens.shape[1] - 1])
        for arm, extra in arms.items():
            logp = final_logprobs(model, tokens, clamp + extra)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm,
                "delta_margin": float((logp[s] - logp[a]) - base),
                "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "top1_str": model.tokenizer.decode([top1]),
                "kl": kl_divergence(clean, logp),
            }) + "\n")
        if n % 10 == 0:
            print(f"[q44] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block39] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
