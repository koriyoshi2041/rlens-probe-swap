#!/usr/bin/env python
"""Block 79: are L23 h8/h9/h0 the model's own second-hop retrieval heads?  (clean-run, both models)

Block 61(b): mean-ablating the whole L17-24 attention stage at the final position drops the
model's native two-hop accuracy 1.00 -> 0.54 and leaves single-hop intact. Blocks 67/74: the
heads whose attention the complement paste redirects most are L23 h8, h9, h0 in BOTH models.
Here, without any lens intervention, the z-output of individual L23 heads at the final position
is mean-ablated (replaced by the head's mean over prompt positions) on the two-hop prompt and on
the single-hop prompt (repaired template with the intermediate):
  h8 | h9 | h0 | {h8,h9,h0} | 3 random other heads (seeded per item) | all L23 heads | all L19 heads
Reports accuracy after ablation and delta log p(answer). If {h8,h9,h0} hurts two-hop but not
single-hop while random heads do neither, the transport heads found by the editing experiments
are the native second-hop retrieval heads.  MATS_MODEL=4b runs the same on Qwen3.5-4B.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

from q30_4b_replication import load_4b
from q69_key_value import hook_name
from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import resolve_next_token

FOURB = os.environ.get("MATS_MODEL", "9b") == "4b"
THIRD = os.environ.get("MATS_MODEL", "9b") == "3rd"
HEADS = [int(h) for h in os.environ.get("MATS_HEADS", "8,9,0").split(",")]
ABLATION = os.environ.get("MATS_ABLATION", "mean")  # mean | zero | resample (block 81b: stronger ablations than the mean)
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
HEAD_SETS = {**{f"h{h}": [h] for h in HEADS}, "".join(f"h{h}" for h in HEADS): list(HEADS)}


def mean_ablate_heads(layer, heads):
    """hook_z [batch, pos, heads, d_head]: replace the final position's z of `heads` (None = all) with their mean over
    positions (ABLATION=mean), with zeros (zero), or with the z of a random other prompt position (resample)."""
    def fn(tensor, hook):
        out = tensor.clone()
        hs = list(range(out.shape[2])) if heads is None else heads
        n_pos = tensor.shape[1]
        for h in hs:
            if ABLATION == "zero":
                out[:, -1, h, :] = 0
            elif ABLATION == "resample":
                src = int(torch.randint(0, max(n_pos - 1, 1), (1,)).item())
                out[:, -1, h, :] = tensor[:, src, h, :]
            else:
                out[:, -1, h, :] = tensor[:, :, h, :].mean(dim=1)
        return out
    return [(hook_name(layer, "hook_z"), fn)]


@torch.inference_mode()
def main() -> int:
    suffix = "" if ABLATION == "mean" else f"_{ABLATION}"
    if THIRD:
        from q81_third_model import load_third
        out_dir = results_dir("block84_qwen3_4b_clean_head_ablation" + suffix)
        model = load_third()
        clean_rows = json.loads((RESULTS_DIR / "block81_qwen3_4b" / "clean_rows.json").read_text(encoding="utf-8"))
    else:
        out_dir = results_dir(("block79_4b_clean_head_ablation" if FOURB else "block79_clean_head_ablation") + suffix)
        if FOURB:
            model, _ = load_4b()
        else:
            model = load_model()
        clean_rows = json.loads((RESULTS_DIR / ("block27_4b" if FOURB else "block01") / "clean_rows.json").read_text(encoding="utf-8"))
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    n_heads = None
    handle = (out_dir / "head_ablation.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        if str(index) not in templates:
            continue
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        prompts = {"two_hop": (r.prompt, r.answer.first_id)}
        sh = templates[str(index)].format(r.item.intermediate)
        prompts["single_hop"] = (sh, resolve_next_token(model.tokenizer, sh, r.item.answer).first_id)
        rng = np.random.default_rng(index)
        torch.manual_seed(index)
        for kind, (text, ans) in prompts.items():
            t = model.to_tokens(text, prepend_bos=False)
            if n_heads is None:
                _, c = model.run_with_cache(t, names_filter=lambda x: x == hook_name(23, "hook_z"))
                n_heads = int(c[hook_name(23, "hook_z")].shape[2])
            lp0 = final_logprobs(model, t)
            others = [h for h in range(n_heads) if h not in HEADS]
            rand3 = [int(h) for h in rng.choice(others, size=3, replace=False)]
            arms = {name: mean_ablate_heads(23, hs) for name, hs in HEAD_SETS.items()}
            arms["rand3@23"] = mean_ablate_heads(23, rand3)
            arms["all@23"] = mean_ablate_heads(23, None)
            arms["all@19"] = mean_ablate_heads(19, None)
            for arm, hooks in arms.items():
                lp = final_logprobs(model, t, hooks)
                handle.write(json.dumps({"index": index, "name": r.item.name, "prompt_kind": kind, "arm": arm, "rand3": rand3 if arm == "rand3@23" else None,
                                         "clean_correct": int(lp0.argmax().item()) == ans, "still_correct": int(lp.argmax().item()) == ans,
                                         "delta_logp_answer": float(lp[ans] - lp0[ans]), "top1_str": model.tokenizer.decode([int(lp.argmax().item())])}) + "\n")
        if n % 10 == 0:
            print(f"[q79] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block79] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
