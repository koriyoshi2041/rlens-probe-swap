#!/usr/bin/env python
"""Block 38: does the donor paste carry the ENTITY or a pre-computed ANSWER?

Block 31's donor prompt is the single-hop template with the target entity ("Fact: The
hard covering on the back of a snake is called a"); the relation words precede the
entity, so at the "snake" position the model may already have computed "scale". Pasting
that residual could be pasting the answer. This block repeats the paste with donors in
which the entity appears in RELATION-FREE contexts:

  N1  "Here is a word: {entity}"
  N2  "Fact: Consider the following: {entity}"

and adds two controls at the same bridge position:
  src_neutral   paste the SOURCE entity's own neutral residual (turtle) -> should be ~clean;
  relation      the block-31 relation donor (reference, re-run here for the same items).

If the neutral donors still flip the band-unflippable items (paste_full / paste_orth), the
flips come from entity features, not from an answer computed in the donor context.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last, paste_hooks
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
NEUTRAL = {"N1": "Here is a word: {}", "N2": "Fact: Consider the following: {}"}


def donor_vector(model, text, entity, wanted, names):
    tokens = model.to_tokens(text, prepend_bos=False)
    seq = tokens[0].tolist()
    pos = None
    for variant in (" " + entity, entity):
        pos = find_last(seq, list(encode_variant(model.tokenizer, variant).ids))
        if pos is not None:
            break
    if pos is None:
        return None, None
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {layer: cache[names[layer]][0, pos] for layer in BAND}, model.tokenizer.decode([seq[pos]])


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block38_neutral_donor")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "neutral_donor.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        position = best_pos[index]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        bases = {layer: lens.lens_vectors(model, ids, layer) for layer in BAND}
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        donors = {}
        for key, tmpl in NEUTRAL.items():
            donors[f"{key}_target"] = donor_vector(model, tmpl.format(r.item.swap_to), r.item.swap_to, wanted, names)
            donors[f"{key}_source"] = donor_vector(model, tmpl.format(r.item.intermediate), r.item.intermediate, wanted, names)
        donors["relation_target"] = donor_vector(model, templates[str(index)].format(r.item.swap_to), r.item.swap_to, wanted, names)
        for dname, (vec, tok) in donors.items():
            if vec is None:
                continue
            for mode in ("full", "orth"):
                hooks = paste_hooks(model, vec, BAND, position, mode, bases if mode == "orth" else None)
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "donor": dname, "donor_token": tok, "mode": mode, "position": position,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "top1_str": model.tokenizer.decode([top1]),
                    "kl": kl_divergence(clean, logp),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q43] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block38] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
