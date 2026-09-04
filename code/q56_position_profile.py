#!/usr/bin/env python
"""Block 51: does R put its early-band displacement where the entity actually is?

Block 48: R's early-band clamp advantage survives total-energy matching but not
per-position matching, i.e. it is about WHERE the displacement lands. Prediction: R's
per-position clamp energy is more concentrated on the positions where the intermediate is
readable (and where the single-position clamp works, block 13 E2) than J's. Per item, band
L3-8 (and L8-20 as contrast): per-position ||delta||^2 summed over band layers for the J
and R clamps; per-position readability = number of band layers with intermediate rank<=10
(J readout and R readout); the single-position clamp effect from block13 E2 (L8-20 only).
Reports: per-item Spearman(energy profile, readability profile) for J and R, and the share
of each lens's energy on its top-readout position.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BANDS = {"early_L3_8": list(range(3, 9)), "primary_L8_20": list(range(8, 21))}


def capturing_clamp_hooks(model, bases, coords, layers, store):
    hooks = []
    for layer in layers:
        matrix = bases[layer].T.float()
        pinv = torch.linalg.pinv(matrix)
        target = coords[layer][..., [1, 0]].float()

        def transform(selected, matrix=matrix, pinv=pinv, target=target, layer=layer):
            h = selected.float()
            c = h @ pinv.to(h.device).T
            delta = (target.to(h.device) - c) @ matrix.to(h.device).T
            store[layer] = delta.detach().clone()
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    def avg_rank(a):  # average ranks for ties (audit 2 fix)
        order = np.argsort(a, kind="mergesort"); ranks = np.empty(len(a)); sa = a[order]; i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sa[j + 1] == sa[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
        return ranks
    rx = avg_rank(x); ry = avg_rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block51_position_profile")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "position_profile.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        ids = [r.src.first_id, r.tgt.first_id]
        for band_name, band in BANDS.items():
            names = {layer: _resid_post_hook_name(layer) for layer in band}
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in set(names.values()))
            rec = {"index": index, "name": r.item.name, "band": band_name, "n_prompt": int(tokens.shape[1])}
            for k in ("J", "R"):
                lens = lenses[k]
                ranks, _ = rank_readout(model, lens, tokens, ids, band)
                hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy().tolist()  # per position
                best_rank = ranks[:-1, :, 0].min(dim=0).values.numpy().tolist()
                bases, coords = {}, {}
                for l in band:
                    bases[l] = lens.lens_vectors(model, ids, l)
                    pinv = torch.linalg.pinv(bases[l].T.float())
                    coords[l] = cache[names[l]].float() @ pinv.T
                store = {}
                final_logprobs(model, tokens, capturing_clamp_hooks(model, bases, coords, band, store))
                energy = torch.stack([store[l][0].norm(dim=-1) ** 2 for l in band]).sum(dim=0).cpu().numpy()  # per position
                gap = torch.stack([(coords[l][0, :, 1] - coords[l][0, :, 0]).abs() for l in band]).mean(dim=0).cpu().numpy()
                rec[f"{k}_hits"] = hits; rec[f"{k}_best_rank"] = best_rank
                rec[f"{k}_energy"] = energy.tolist(); rec[f"{k}_gap"] = gap.tolist()
                rec[f"{k}_spearman_energy_hits"] = spearman(energy, hits)
                rec[f"{k}_energy_share_on_best_readout_pos"] = float(energy[int(np.argmax(hits))] / max(energy.sum(), 1e-9))
                rec[f"{k}_energy_share_last"] = float(energy[-1] / max(energy.sum(), 1e-9))
            # cross: R's energy vs J's readability and vice versa
            rec["R_energy_vs_J_hits"] = spearman(rec["R_energy"], rec["J_hits"])
            rec["J_energy_vs_R_hits"] = spearman(rec["J_energy"], rec["R_hits"])
            rec["R_vs_J_energy_profile_spearman"] = spearman(rec["R_energy"], rec["J_energy"])
            handle.write(json.dumps(rec) + "\n")
        if n % 10 == 0:
            print(f"[q56] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block51] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
