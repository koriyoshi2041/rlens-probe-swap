#!/usr/bin/env python
"""Block 55: OUT-OF-SAMPLE test of "linear relatedness predicts editability" on new non-geographic items.

Pre-registered prediction (written before running): on the new items in
data/new_items_nongeo.json, the J clamp (L8-20, all positions) flips items whose
cos(v_swap_to - v_intermediate, v_swap_answer - v_answer) in unembedding space is high
(>= 0.25) more often than items with low cos (< 0.10); the single-position full-vector
donor paste (neutral carrier) flips more items than the clamp; and no geography is needed
for high-cos items to flip. Everything else follows the block-01/02 protocol: clean
baseline in both prompt modes, hybrid eligibility rule, single-hop knowledge ceiling,
strict next-token scoring, 12-token greedy continuations as raw samples.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last
from rlens.data import SwapItem
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import results_dir
from rlens.readout import rank_readout
from rlens.stages.common import resolve_item
from rlens.tokens import encode_variant, resolve_next_token

BAND = list(range(8, 21))
import os
DATA = pathlib.Path(__file__).resolve().parent / "data" / os.environ.get("MATS_NEWITEMS", "new_items_nongeo.json")
CARRIER = "Fact: Consider the following: {}"
GEN = 12


@torch.inference_mode()
def generate(model, seq, hooks, n_tokens):
    start = seq.shape[1]
    for _ in range(n_tokens):
        logp = final_logprobs(model, seq, hooks)
        seq = torch.cat([seq, logp.argmax().view(1, 1)], dim=1)
    return model.tokenizer.decode(seq[0, start:].tolist())


def donor_vec(model, entity, wanted, names):
    tokens = model.to_tokens(CARRIER.format(entity), prepend_bos=False)
    seq = tokens[0].tolist()
    pos = None
    for variant in (" " + entity, entity):
        pos = find_last(seq, list(encode_variant(model.tokenizer, variant).ids))
        if pos is not None:
            break
    if pos is None:
        return None
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {layer: cache[names[layer]][0, pos] for layer in BAND}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block55_newitems" + ("_v2" if "v2" in os.environ.get("MATS_NEWITEMS", "") else ""))
    model = load_model()
    lenses = load_lenses()
    lens = lenses["J"]
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    items = [SwapItem(index=i, name=it["name"], category=it["category"], prompt=it["prompt"], intermediate=it["intermediate"],
                      answer=it["answer"], swap_to=it["swap_to"], swap_answer=it["swap_answer"]) for i, it in enumerate(payload["items"])]
    single_hop = {i: it["single_hop"] for i, it in enumerate(payload["items"])}
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    wu = model.W_U.float()
    rows, lines = [], []
    for item in items:
        # hybrid prompt-mode rule as in block 01
        chosen = None
        for mode in ("rstrip", "as_is"):
            r = resolve_item(model.tokenizer, item, mode)
            if r.answers_single and r.answers_distinct:
                chosen = r
                break
        if chosen is None:
            chosen = resolve_item(model.tokenizer, item, "as_is")
        r = chosen
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        top1 = int(clean.argmax().item())
        rec = {"index": item.index, "name": item.name, "category": item.category, "mode": r.mode, "prompt": r.prompt,
               "intermediate": item.intermediate, "swap_to": item.swap_to, "answer": item.answer, "swap_answer": item.swap_answer,
               "clean_top1": model.tokenizer.decode([top1]), "clean_correct": top1 == a,
               "concept_feasible": r.concept_feasible, "answers_single": r.answers_single, "answers_distinct": r.answers_distinct,
               "clean_margin": float(clean[s] - clean[a])}
        rec["eligible"] = bool(rec["clean_correct"] and r.concept_feasible and r.answers_single and r.answers_distinct)
        # geometry: cos(entity diff, answer diff) in unembedding space (block05 definition)
        ed = wu[:, r.tgt.first_id] - wu[:, r.src.first_id]
        ad = wu[:, s] - wu[:, a]
        rec["cos_entitydiff_answerdiff"] = float(torch.nn.functional.cosine_similarity(ed[None], ad[None]).item())
        rec["cos_intermediate_swapto"] = float(torch.nn.functional.cosine_similarity(wu[:, r.src.first_id][None], wu[:, r.tgt.first_id][None]).item())
        # knowledge ceiling
        for side, entity, expected in (("intermediate", item.intermediate, item.answer), ("swap_to", item.swap_to, item.swap_answer)):
            p = single_hop[item.index].format(entity)
            t = model.to_tokens(p, prepend_bos=False)
            lp = final_logprobs(model, t)
            tgt = resolve_next_token(model.tokenizer, p, expected)
            rec[f"knowledge_{side}_correct"] = int(lp.argmax().item()) == tgt.first_id
            rec[f"knowledge_{side}_top1"] = model.tokenizer.decode([int(lp.argmax().item())])
        if rec["eligible"]:
            ids = [r.src.first_id, r.tgt.first_id]
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            # band clamp (J and R), all positions
            for kind in ("J", "R"):
                hooks = []
                for layer in BAND:
                    basis = lenses[kind].lens_vectors(model, ids, layer)
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords = cache[names[layer]].float() @ pinv.T
                    hooks += clamp_hooks(model, basis, [layer], {layer: coords})
                lp = final_logprobs(model, tokens, hooks)
                t1 = int(lp.argmax().item())
                rec[f"clamp_{kind}_delta_margin"] = float((lp[s] - lp[a]) - (clean[s] - clean[a]))
                rec[f"clamp_{kind}_flip"] = t1 == s
                rec[f"clamp_{kind}_top1"] = model.tokenizer.decode([t1])
                rec[f"clamp_{kind}_kl"] = kl_divergence(clean, lp)
                if kind == "J":
                    rec["clamp_J_continuation"] = generate(model, tokens, [h for h in hooks], GEN) if False else None
                    prompt_hooks = []
                    for layer in BAND:
                        basis = lens.lens_vectors(model, ids, layer)
                        pinv = torch.linalg.pinv(basis.T.float())
                        coords = cache[names[layer]].float() @ pinv.T
                        prompt_hooks += clamp_hooks(model, basis, [layer], {layer: coords}, positions=list(range(tokens.shape[1])))
                    rec["clamp_J_continuation"] = generate(model, tokens, prompt_hooks, GEN)
            rec["clean_continuation"] = generate(model, tokens, [], GEN)
            # best bridge position and single-position donor paste (neutral carrier)
            ranks, _ = rank_readout(model, lens, tokens, ids, BAND)
            hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
            bp = int(max(range(len(hits)), key=lambda p: (hits[p], p)))
            rec["best_pos"] = bp; rec["best_pos_hits"] = int(hits[bp]); rec["pos_token"] = model.tokenizer.decode([tokens[0, bp].item()])
            dv = donor_vec(model, item.swap_to, wanted, names)
            if dv is not None:
                from q35_donor_paste import paste_hooks
                lp = final_logprobs(model, tokens, paste_hooks(model, dv, BAND, bp, "full"))
                t1 = int(lp.argmax().item())
                rec["paste_full_flip"] = t1 == s; rec["paste_full_top1"] = model.tokenizer.decode([t1])
                rec["paste_full_delta_margin"] = float((lp[s] - lp[a]) - (clean[s] - clean[a]))
                bases = {l: lens.lens_vectors(model, ids, l) for l in BAND}
                lp = final_logprobs(model, tokens, paste_hooks(model, dv, BAND, bp, "orth", bases))
                rec["paste_orth_flip"] = int(lp.argmax().item()) == s
        rows.append(rec)
        lines.append(f"### {item.name}  [{item.intermediate} -> {item.swap_to}]  expect {item.answer} -> {item.swap_answer}  cos={rec['cos_entitydiff_answerdiff']:+.2f}  eligible={rec['eligible']} (clean top1 {rec['clean_top1']!r})")
        if rec["eligible"]:
            lines.append(f"    clean : {rec['clean_continuation']!r}")
            lines.append(f"    Jclamp: {rec['clamp_J_continuation']!r}   flip={rec['clamp_J_flip']}  paste_full={rec.get('paste_full_flip')}")
        lines.append("")
        print(f"[q60] {item.name} eligible={rec['eligible']} cos={rec['cos_entitydiff_answerdiff']:+.2f}", flush=True)
    (out_dir / "newitems.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "newitems_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[block55] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
