#!/usr/bin/env python
"""Block 33: raw continuations for the donor-paste result (the reviewer's "show me the samples").

For every item that the full-band J clamp never flips but the single-position full-vector
donor paste does (block31), generate 16 greedy tokens under: clean, the 2-D clamp at the
best bridge position, and the donor paste at the same position (hooks only on the
original prompt positions; generated tokens are free). Also for 8 fixed random items
from the rest, so the sample is not cherry-picked. Nothing is auto-judged.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last, paste_hooks
from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
GEN = 16
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


@torch.inference_mode()
def generate(model, seq, hooks, n_tokens):
    start = seq.shape[1]
    for _ in range(n_tokens):
        logp = final_logprobs(model, seq, hooks)
        seq = torch.cat([seq, logp.argmax().view(1, 1)], dim=1)
    return model.tokenizer.decode(seq[0, start:].tolist())


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block33_donor_continuations")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    paste_rows = [json.loads(l) for l in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines() if l.strip()]
    sup = [json.loads(l) for l in (RESULTS_DIR / "block12_suppression" / "suppression.jsonl").read_text().splitlines() if l.strip()]
    band_flip = {r["index"]: r["top1_is_swap"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp"}
    best = {r["index"]: r for r in paste_rows if r["position_set"] == "best" and r["arm"] == "paste_full"}
    rescued = sorted(i for i, r in best.items() if r["top1_is_swap"] and not band_flip.get(i, False))
    rng = np.random.default_rng(0)
    others = sorted(rng.choice([i for i in eligible if i not in rescued], size=8, replace=False).tolist())
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    lines = [f"# rescued (band clamp never flips, donor paste flips): {len(rescued)} items; plus 8 random others (seed 0)\n"]
    records = []
    for group, indices in (("rescued", rescued), ("random_other", others)):
        for index in indices:
            r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
            tokens = model.to_tokens(r.prompt, prepend_bos=False)
            seq = tokens[0].tolist()
            ids = [r.src.first_id, r.tgt.first_id]
            position = int(best[index]["position"])
            donor_text = templates[str(index)].format(r.item.swap_to)
            donor_tokens = model.to_tokens(donor_text, prepend_bos=False)
            dseq = donor_tokens[0].tolist()
            donor_pos = None
            for variant in (" " + r.item.swap_to, r.item.swap_to):
                donor_pos = find_last(dseq, list(encode_variant(model.tokenizer, variant).ids))
                if donor_pos is not None:
                    break
            _, dcache = model.run_with_cache(donor_tokens, names_filter=lambda x: x in wanted)
            donor_vec = {layer: dcache[names[layer]][0, donor_pos] for layer in BAND}
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            bases, coords = {}, {}
            for layer in BAND:
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            arms = {
                "clean": [],
                "clamp2d": sum((clamp_hooks(model, bases[l], [l], {l: coords[l][:, position:position + 1, :]}, positions=[position]) for l in BAND), []),
                "paste_full": paste_hooks(model, donor_vec, BAND, position, "full"),
            }
            rec = {"group": group, "index": index, "name": r.item.name, "prompt": r.prompt, "position": position,
                   "pos_token": model.tokenizer.decode([seq[position]]), "intermediate": r.item.intermediate, "swap_to": r.item.swap_to,
                   "answer": r.answer.text.strip(), "swap_answer": r.swap_answer.text.strip(), "continuations": {}}
            lines.append(f"### [{group}] {r.item.name}  [{r.item.intermediate} -> {r.item.swap_to}]  expect {rec['answer']} -> {rec['swap_answer']}  (paste at pos {position} = {rec['pos_token']!r})")
            lines.append(f"    prompt: {r.prompt}")
            for arm, hooks in arms.items():
                text = generate(model, tokens, hooks, GEN)
                rec["continuations"][arm] = text
                lines.append(f"    {arm:10s}: {text!r}")
            lines.append("")
            records.append(rec)
            print(f"[q39] {group} {r.item.name}", flush=True)
    (out_dir / "continuations.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "continuations_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[block33] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
