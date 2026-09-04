#!/usr/bin/env python
"""Block 42: is L23.h8 the model's own two-hop carrier, or an artefact of the clamp?

Block 37 found that under the clamp, L23 head 8 carries most of the new answer into the
final position. If that head is the model's native "bridge entity -> answer" transfer, then
ablating it in the CLEAN run should hurt the model's own two-hop answers more than its
single-hop answers (where the entity is stated and no bridge transfer is needed).

Arms (clean run, head output z set to zero at the final position only):
  two-hop prompt   : the 59 eligible items; measure clean answer log-prob drop and top-1 loss
  single-hop prompt: the repaired single-hop template filled with the intermediate; same measure
Heads: L23.h8, L23.h9, L23.h0, L23.random, L23.all, L31.h14, L19.h11. Also mean-ablation
(replace z with the head's mean over prompt positions) as a gentler variant for h8.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import resolve_next_token

TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
TARGETS = {"L23.h8": (23, [8]), "L23.h9": (23, [9]), "L23.h0": (23, [0]), "L23.all": (23, list(range(16))), "L31.h14": (31, [14]), "L19.h11": (19, [11])}


def zero_heads(heads, position, mean=False):
    def fn(act, hook):
        out = act.clone()
        if mean:
            out[:, position, heads, :] = act[:, :, heads, :].mean(dim=1)
        else:
            out[:, position, heads, :] = 0
        return out
    return fn


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block42_head_native_role")
    model = load_model()
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "head_native_role.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        prompts = {"two_hop": (r.prompt, r.answer.first_id, r.swap_answer.first_id)}
        single = templates[str(index)].format(r.item.intermediate)
        tgt = resolve_next_token(model.tokenizer, single, r.item.answer)
        prompts["single_hop"] = (single, tgt.first_id, r.swap_answer.first_id)
        gen = torch.Generator(device="cpu").manual_seed(index)
        rand_head = int(torch.randint(0, 16, (1,), generator=gen).item())
        while rand_head in (8, 9):
            rand_head = int(torch.randint(0, 16, (1,), generator=gen).item())
        for kind, (text, ans_id, swap_id) in prompts.items():
            tokens = model.to_tokens(text, prepend_bos=False)
            last = tokens.shape[1] - 1
            clean = final_logprobs(model, tokens)
            arms = {}
            for name, (layer, heads) in TARGETS.items():
                arms[name] = [(f"blocks.{layer}.attn.hook_z", zero_heads(heads, last))]
            arms["L23.random"] = [(f"blocks.{23}.attn.hook_z", zero_heads([rand_head], last))]
            arms["L23.h8_mean"] = [(f"blocks.{23}.attn.hook_z", zero_heads([8], last, mean=True))]
            for arm, hooks in arms.items():
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "prompt_kind": kind, "arm": arm,
                    "clean_correct": int(clean.argmax().item()) == ans_id,
                    "still_correct": top1 == ans_id,
                    "delta_logp_answer": float(logp[ans_id] - clean[ans_id]),
                    "delta_margin_swap_minus_answer": float((logp[swap_id] - logp[ans_id]) - (clean[swap_id] - clean[ans_id])),
                    "top1_str": model.tokenizer.decode([top1]),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q47] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block42] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
