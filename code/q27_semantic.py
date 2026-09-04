#!/usr/bin/env python
"""Block 23: score the rewrite semantically, not by one token.

The sampled continuations exposed a scoring flaw. Clamping Canada->China makes the
model answer "Renminbi"; the benchmark's expected token is " yuan", so the strict
next-token metric calls that a failure. Clamping Seattle->Philadelphia makes it
answer "the state where the Empire State Building is located" -- New York, given as
a description. Both are the intended rewrite, both scored zero. Two of twelve
sampled items were mis-scored this way, so the reported flip rate is a lower bound.

This run produces, for every eligible item, the clean continuation and the
continuation under each lens's clamp, long enough to read. It also asks the model,
with the clamp still held on the original prompt positions, to name the bridge
entity itself: the two-hop prompt is extended with its own clean answer and a probe
clause, so what comes next is the model's own account of which entity it is using.
The continuation from the sampled run already did this spontaneously -- after
Brazil->Mexico it asked itself which country the Amazon ends in and answered Peru --
and this makes that measurement systematic rather than incidental.

Nothing here is auto-judged. The output is a file to be read.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
GEN_TOKENS = 20
PROBE_TOKENS = 8


@torch.inference_mode()
def generate(model, seq, hooks, n_tokens):
    start = seq.shape[1]
    for _ in range(n_tokens):
        logp = final_logprobs(model, seq, hooks)
        seq = torch.cat([seq, logp.argmax().view(1, 1)], dim=1)
    return model.tokenizer.decode(seq[0, start:].tolist())


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block23_semantic")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    rows = []
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        n_prompt = tokens.shape[1]
        ids = r.tracked_ids[:2]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        probe_text = r.prompt + r.answer.text + ". The " + ("country" if r.item.category.startswith(("city-", "language-", "river-")) else "thing") + " referred to above is"
        probe_tokens = model.to_tokens(probe_text, prepend_bos=False)
        record = {"index": index, "name": r.item.name, "category": r.item.category, "prompt": r.prompt,
                  "intermediate": r.item.intermediate, "swap_to": r.item.swap_to,
                  "answer": r.answer.text.strip(), "swap_answer": r.swap_answer.text.strip(), "arms": {}}
        for kind in ("clean", "J", "R"):
            hooks, probe_hooks = [], []
            if kind != "clean":
                lens = lenses[kind]
                for layer in BAND:
                    basis = lens.lens_vectors(model, ids, layer)
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords = cache[names[layer]].float() @ pinv.T
                    hooks += clamp_hooks(model, basis, [layer], {layer: coords}, positions=list(range(n_prompt)))
                    probe_hooks += clamp_hooks(model, basis, [layer], {layer: coords}, positions=list(range(n_prompt)))
            record["arms"][kind] = {
                "continuation": generate(model, tokens, hooks, GEN_TOKENS),
                "entity_probe": generate(model, probe_tokens, probe_hooks, PROBE_TOKENS),
            }
        rows.append(record)
        print(f"[semantic] {n + 1}/{len(eligible)} {r.item.name}", flush=True)
    (out_dir / "semantic.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = []
    for rec in rows:
        lines.append(f"### {rec['name']}  [{rec['intermediate']} -> {rec['swap_to']}]  expect {rec['answer']} -> {rec['swap_answer']}")
        lines.append(f"    prompt: {rec['prompt']}")
        for kind in ("clean", "J", "R"):
            a = rec["arms"][kind]
            lines.append(f"    {kind:5s} cont : {a['continuation']!r}")
            lines.append(f"    {kind:5s} probe: {a['entity_probe']!r}")
        lines.append("")
    (out_dir / "semantic_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[block23] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
