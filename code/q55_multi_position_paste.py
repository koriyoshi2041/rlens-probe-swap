#!/usr/bin/env python
"""Block 50: are the remaining unflippable items multi-position cases?

Block 31: full-vector donor paste at the single best-readout bridge position rescues 16 of
the 35 items the band clamp never flips; 19 stay put even with the full entity
representation at that one position. Hypothesis: for those items the bridge entity is
carried at several positions (the readout scan showed hits spread over the last 1-5 tokens)
and a single-position paste is out-voted. Arms (relation donor, full vector, L8-20):

  best1      the best-readout position (block 31 reference)
  best2      the two best positions
  best3      the three best positions
  hits>=3    every position with >= 3 band layers reading the intermediate in the top-10
  last5      the last five prompt positions
and, as the matching lens-plane arm, the 2-D J clamp at the same position sets.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from q35_donor_paste import find_last
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def multi_paste_hooks(model, donor_vec, layers, positions):
    hooks = []
    for layer in layers:
        donor = donor_vec[layer].float()

        def transform(selected, donor=donor):
            return donor.to(selected.device).expand_as(selected.float()).clone()

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, list(positions), model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block50_multi_position_paste")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "multi_position.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        seq = tokens[0].tolist()
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        ranks, _ = rank_readout(model, lens, tokens, ids, BAND)
        hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
        order = sorted(range(len(hits)), key=lambda p: (-hits[p], -p))
        pos_sets = {
            "best1": [order[0]], "best2": sorted(order[:2]), "best3": sorted(order[:3]),
            "hits>=3": sorted([p for p in range(len(hits)) if hits[p] >= 3]) or [order[0]],
            "last5": list(range(max(0, len(seq) - 5), len(seq))),
        }
        donor_text = templates[str(index)].format(r.item.swap_to)
        donor_tokens = model.to_tokens(donor_text, prepend_bos=False)
        dseq = donor_tokens[0].tolist()
        donor_pos = None
        for variant in (" " + r.item.swap_to, r.item.swap_to):
            donor_pos = find_last(dseq, list(encode_variant(model.tokenizer, variant).ids))
            if donor_pos is not None:
                break
        if donor_pos is None:
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
        for set_name, positions in pos_sets.items():
            arms = {
                "paste_full": multi_paste_hooks(model, donor_vec, BAND, positions),
                "clamp2d": sum((clamp_hooks(model, bases[l], [l], {l: coords[l][:, positions, :]}, positions=list(positions)) for l in BAND), []),
            }
            for arm, hooks in arms.items():
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "position_set": set_name, "positions": positions, "n_positions": len(positions),
                    "n_prompt": len(seq), "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "top1_str": model.tokenizer.decode([top1]),
                    "kl": kl_divergence(clean, logp),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q55] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block50] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
