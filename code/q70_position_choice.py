#!/usr/bin/env python
"""Block 69: is the content elsewhere?  Editing at the position the transport head actually reads,
and is the key-side routing signal entity-agnostic (foil keys)?

Block 66: for the 27 never-flipped items the 2-D clamp cannot rescue, L23 h8 barely attends
to the best-readout bridge position, and forcing attention there with the donor's keys plus
the 2-D clamp rescues only 1/27 (the plane there lacks the content). Two follow-ups:
  (1) POSITION. Take the prompt position h8 attends to most in the clean run (p_h8, final
      position excluded). Is the intermediate readable there? Does the 2-D clamp at p_h8 (x1,
      x4) or at both positions flip these items? Does the donor complement at p_h8?
      Reference: the 2-D clamp at ALL prompt positions at x1 (= the band clamp) and x4.
  (2) FOIL KEYS. Replace the donor by a FOIL donor (another item's target entity, the seed-0
      derangement of block 31). If the complement of the foil also raises h8's attention onto
      the bridge, the key-side signal is "an entity mention is here", not entity-specific; and
      if the foil's keys + the 2-D clamp reproduce the routing-assisted clamp (block 66: 0.96
      on flippable items), routing is entity-agnostic while the content comes from the plane.
Records per item: p_best, p_h8, its token, h8's clean attention mass at p_best / p_h8 / final /
position 0, readability hits at both positions, and per arm margin, flip, top-1 string, and
h8 attention onto p_best and p_h8.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last
from q66_energy_of_arms import scaled_paste_hooks
from q69_key_value import capture, hook_name, patch_position
from rlens.data import load_items
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
ATTN_LAYERS = (19, 23)
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def donor_vector(model, text, entity, wanted, names):
    tokens = model.to_tokens(text, prepend_bos=False)
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


def clamp_at(model, bases, coords, positions, scale):
    hooks = []
    for l in BAND:
        c = coords[l][:, positions, :]
        target = c + scale * (c[..., [1, 0]] - c)
        hooks += clamp_hooks(model, bases[l], [l], {l: target}, exchange=False, positions=list(positions))
    return hooks


def run(model, tokens, hooks, p_best, p_h8, s, a):
    wanted = {hook_name(23, "hook_pattern")}
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    top1 = int(lp.argmax().item())
    pat = cache[hook_name(23, "hook_pattern")][0, 8, -1, :].float()
    return {"margin": float(lp[s] - lp[a]), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
            "top1_str": model.tokenizer.decode([top1]), "h8_best": float(pat[p_best].item()), "h8_h8pos": float(pat[p_h8].item())}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block69_position_choice")
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
    rng = np.random.default_rng(0)  # same derangement as block 31 foil
    perm = list(range(len(eligible)))
    while any(p == k for k, p in enumerate(perm)):
        rng.shuffle(perm)
    foil_of = {eligible[k]: eligible[p] for k, p in enumerate(perm)}
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values()) | {hook_name(23, "hook_pattern")}
    handle = (out_dir / "position_choice.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        T = int(tokens.shape[1])
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        p_best = best_pos[index]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        pat = cache[hook_name(23, "hook_pattern")][0, 8, -1, :].float()
        p_h8 = int(pat[: T - 1].argmax().item())
        ranks, _ = rank_readout(model, lens, tokens, ids, BAND)
        hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
        donor = donor_vector(model, templates[str(index)].format(r.item.swap_to), r.item.swap_to, wanted, names)
        fi = foil_of[index]
        foil = donor_vector(model, templates[str(fi)].format(items[fi].swap_to), items[fi].swap_to, wanted, names)
        if donor is None or foil is None:
            continue
        bases = {l: lens.lens_vectors(model, ids, l) for l in BAND}
        coords = {l: cache[names[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in BAND}
        rec = {"index": index, "name": r.item.name, "n_prompt": T, "p_best": p_best, "p_h8": p_h8,
               "tok_best": model.tokenizer.decode([tokens[0, p_best].item()]), "tok_h8": model.tokenizer.decode([tokens[0, p_h8].item()]),
               "h8_mass": {"best": float(pat[p_best]), "h8pos": float(pat[p_h8]), "final": float(pat[-1]), "pos0": float(pat[0])},
               "hits_best": int(hits[p_best]), "hits_h8": int(hits[p_h8]), "foil_index": fi, "foil_entity": items[fi].swap_to, "foil_answer": items[fi].swap_answer,
               "arms": {}}
        both = sorted({p_best, p_h8})
        arms = {
            "clean": [],
            "clamp@1_best": clamp_at(model, bases, coords, [p_best], 1.0), "clamp@4_best": clamp_at(model, bases, coords, [p_best], 4.0),
            "clamp@1_h8": clamp_at(model, bases, coords, [p_h8], 1.0), "clamp@4_h8": clamp_at(model, bases, coords, [p_h8], 4.0),
            "clamp@4_both": clamp_at(model, bases, coords, both, 4.0),
            "clamp@1_all": clamp_at(model, bases, coords, list(range(T)), 1.0), "clamp@4_all": clamp_at(model, bases, coords, list(range(T)), 4.0),
            "orth@0.25_h8": scaled_paste_hooks(model, donor, BAND, p_h8, 0.25, bases, "orth"),
            "full@0.25_h8": scaled_paste_hooks(model, donor, BAND, p_h8, 0.25, None, "full"),
            "orth@0.25_best": scaled_paste_hooks(model, donor, BAND, p_best, 0.25, bases, "orth"),
            "foil_orth@0.25_best": scaled_paste_hooks(model, foil, BAND, p_best, 0.25, bases, "orth"),
            "foil_full@0.25_best": scaled_paste_hooks(model, foil, BAND, p_best, 0.25, None, "full"),
        }
        S = capture(model, tokens, arms["orth@0.25_best"])
        F = capture(model, tokens, arms["foil_orth@0.25_best"])
        arms["K<-donor"] = [(hook_name(l, "hook_k"), patch_position(S[hook_name(l, "hook_k")], p_best)) for l in ATTN_LAYERS]
        arms["K<-foil"] = [(hook_name(l, "hook_k"), patch_position(F[hook_name(l, "hook_k")], p_best)) for l in ATTN_LAYERS]
        arms["K<-donor+clamp@4_best"] = arms["K<-donor"] + arms["clamp@4_best"]
        arms["K<-foil+clamp@4_best"] = arms["K<-foil"] + arms["clamp@4_best"]
        arms["K<-foil+clamp@1_best"] = arms["K<-foil"] + arms["clamp@1_best"]
        results = {arm: run(model, tokens, hooks, p_best, p_h8, s, a) for arm, hooks in arms.items()}
        for arm in results:
            results[arm]["delta_margin"] = results[arm]["margin"] - results["clean"]["margin"]
        rec["arms"] = results
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 10 == 0:
            print(f"[q70] {n + 1}/{len(eligible)} p_best={p_best} p_h8={p_h8} " + " ".join(f"{k}={v['delta_margin']:+.2f}" for k, v in results.items() if k in ("clamp@4_best", "clamp@4_h8", "clamp@4_all", "K<-foil+clamp@4_best")), flush=True)
    handle.close()
    print("[block69] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
