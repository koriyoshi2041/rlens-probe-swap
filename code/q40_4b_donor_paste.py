#!/usr/bin/env python
"""Block 35: the block-31 donor-paste experiment on Qwen3.5-4B (second model).

Question. 35/59 items never flip under the clamp although the intermediate direction is
load-bearing for them (ablation hurts) and the model knows the target's second hop
(52/59 single-hop). Two hypotheses:
  H1  the lens subspace is too small: "snake" has features outside span{v_turtle, v_snake}
      that the second-hop circuit keys on, so a 2-D coordinate edit never looks like a
      snake to the MLPs;
  H2  the second hop for these items is not driven by the bridge-entity representation at
      the positions we edit at all.

Test. Take a DONOR prompt in which the target entity literally appears (the repaired
single-hop template filled with swap_to), cache its residual at the target token, and at
the source prompt's best-readout bridge position compare, across the band L8-20:
  clamp2d     : the J-lens 2-D clamp at that position (what the paper's method can do);
  paste_full  : overwrite the residual with the donor's full vector;
  paste_sub   : set only the 2-D lens coordinates to the donor's (should ~ clamp2d);
  paste_orth  : take the donor's orthogonal complement, keep the source's lens coords.
If paste_full flips items that clamp2d cannot, and paste_orth carries most of that, H1
holds (the missing information is outside the lens plane). If paste_full also fails, H2.
Two position sets: the single best-readout position, and the final position, per item.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from q30_4b_replication import load_4b
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def paste_hooks(model, vectors, layers, position, mode, bases=None):
    """Overwrite (or partially overwrite) the residual at `position` with the donor vector per layer."""
    hooks = []
    for layer in layers:
        donor = vectors[layer].float()
        if bases is not None:
            matrix = bases[layer].T.float()
            pinv = torch.linalg.pinv(matrix)
        else:
            matrix = pinv = None

        def transform(selected, donor=donor, matrix=matrix, pinv=pinv):
            h = selected.float()
            d = donor.to(h.device).expand_as(h)
            if mode == "full":
                return d
            m = matrix.to(h.device)
            p = pinv.to(h.device)
            in_plane = ((d - h) @ p.T) @ m.T  # component of (donor - h) inside the lens plane
            if mode == "sub":
                return h + in_plane
            return h + (d - h) - in_plane  # "orth"

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, [position], model.cfg.d_model)))
    return hooks


def find_last(seq, ids):
    n = len(ids)
    for start in range(len(seq) - n, -1, -1):
        if seq[start:start + n] == ids:
            return start + n - 1
    return None


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block35_4b_donor_paste")
    model, lenses = load_4b()
    lens = lenses["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block27_4b" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    foil_mode = bool(os.environ.get("MATS_DONOR_FOIL"))
    if foil_mode:
        out_dir = results_dir("block35_4b_donor_paste_foil")
    handle = (out_dir / "donor_paste.jsonl").open("w", encoding="utf-8")
    skipped = []
    # derangement of eligible items for the foil donor (seed 0): item i takes the donor of eligible[perm[i]]
    rng = np.random.default_rng(0)
    perm = list(range(len(eligible)))
    while any(p == k for k, p in enumerate(perm)):
        rng.shuffle(perm)
    foil_of = {eligible[k]: eligible[p] for k, p in enumerate(perm)}
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        seq = tokens[0].tolist()
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        # best-readout bridge position: most band layers with intermediate rank <= 10 (ties -> latest)
        ranks, _ = rank_readout(model, lens, tokens, ids, BAND)  # [len(BAND)+1, pos, 2]
        hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
        best_pos = int(max(range(len(hits)), key=lambda p: (hits[p], p)))
        # donor: repaired single-hop template with the target entity
        donor_index = foil_of[index] if foil_mode else index
        donor_item = items[donor_index]
        if str(donor_index) not in templates:  # 4B-eligible items outside the 59 with single-hop templates
            skipped.append({"index": index, "name": r.item.name, "donor": f"no template for item {donor_index}"})
            continue
        donor_text = templates[str(donor_index)].format(donor_item.swap_to)
        donor_tokens = model.to_tokens(donor_text, prepend_bos=False)
        dseq = donor_tokens[0].tolist()
        donor_pos = None
        for variant in (" " + donor_item.swap_to, donor_item.swap_to):
            donor_pos = find_last(dseq, list(encode_variant(model.tokenizer, variant).ids))
            if donor_pos is not None:
                break
        if donor_pos is None:
            skipped.append({"index": index, "name": r.item.name, "donor": donor_text})
            continue
        _, dcache = model.run_with_cache(donor_tokens, names_filter=lambda x: x in wanted)
        donor_vec = {layer: dcache[names[layer]][0, donor_pos] for layer in BAND}
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        bases, coords = {}, {}
        for layer in BAND:
            bases[layer] = lens.lens_vectors(model, ids, layer)
            pinv = torch.linalg.pinv(bases[layer].T.float())
            coords[layer] = cache[names[layer]].float() @ pinv.T
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        for pos_name, position in (("best", best_pos), ("last", len(seq) - 1)):
            arms = {
                "clamp2d": sum((clamp_hooks(model, bases[l], [l], {l: coords[l][:, position:position + 1, :]}, positions=[position]) for l in BAND), []),
                "paste_full": paste_hooks(model, donor_vec, BAND, position, "full"),
                "paste_sub": paste_hooks(model, donor_vec, BAND, position, "sub", bases),
                "paste_orth": paste_hooks(model, donor_vec, BAND, position, "orth", bases),
            }
            for arm, hooks in arms.items():
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "category": r.item.category,
                    "position_set": pos_name, "position": position, "best_pos_hits": int(hits[best_pos]),
                    "n_prompt": len(seq), "pos_token": model.tokenizer.decode([seq[position]]),
                    "donor_prompt": donor_text, "donor_pos": donor_pos, "donor_token": model.tokenizer.decode([dseq[donor_pos]]),
                    "donor_index": donor_index, "foil": foil_mode,
                    "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "top1_str": model.tokenizer.decode([top1]),
                    "kl": kl_divergence(clean, logp),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q35] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[block31] done; skipped {len(skipped)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
