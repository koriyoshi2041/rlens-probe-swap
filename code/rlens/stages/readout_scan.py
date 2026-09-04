"""Stage: rank readout of the four tracked tokens at every (layer, position), J vs R vs logit lens."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch

from ..data import SwapItem
from ..readout import rank_readout
from .common import choose_mode, eligible_indices, resolve_item

TRACKED = ("intermediate", "swap_to", "answer", "swap_answer")
LENS_KEYS = ("J", "R", "logit")


@torch.inference_mode()
def run(model, lenses: Dict[str, object], items: List[SwapItem], clean_rows, out_dir) -> Dict[str, object]:
    mode = choose_mode(clean_rows)
    indices = eligible_indices(clean_rows, mode)
    layers = list(lenses["J"].source_layers)
    n_rows = len(layers) + 1
    tokenizer = model.tokenizer
    resolved = [resolve_item(tokenizer, items[i], mode) for i in indices]
    max_pos = max(model.to_tokens(r.prompt, prepend_bos=False).shape[1] for r in resolved)
    keys = [k for k in LENS_KEYS if k == "logit" or k in lenses]  # lenses not provided (e.g. no R for a third model) are skipped
    ranks = {k: np.full((len(resolved), n_rows, max_pos, len(TRACKED)), -1, dtype=np.int64) for k in keys}
    top1 = {k: np.full((len(resolved), n_rows, max_pos), -1, dtype=np.int64) for k in keys}
    lengths = np.zeros(len(resolved), dtype=np.int64)
    for n, r in enumerate(resolved):
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        seq = tokens.shape[1]
        lengths[n] = seq
        for key in keys:
            lens = None if key == "logit" else lenses[key]
            rk, tp = rank_readout(model, lens, tokens, r.tracked_ids, layers)
            ranks[key][n, :, :seq, :] = rk.numpy()
            top1[key][n, :, :seq] = tp.numpy()
        if n % 10 == 0:
            print(f"[readout] {n + 1}/{len(resolved)} items done")
    np.savez(
        out_dir / "readout_ranks.npz",
        layers=np.array(layers + [model.cfg.n_layers - 1]),
        indices=np.array(indices),
        lengths=lengths,
        **{f"ranks_{k}": v for k, v in ranks.items()},
        **{f"top1_{k}": v for k, v in top1.items()},
    )
    summary = summarize(ranks, lengths, layers + [model.cfg.n_layers - 1])
    summary["mode"] = mode
    summary["indices"] = indices
    summary["n_items"] = len(indices)
    (out_dir / "readout_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_tables(summary)
    return summary


def summarize(ranks: Dict[str, np.ndarray], lengths: np.ndarray, layer_ids: List[int]) -> Dict[str, object]:
    out: Dict[str, object] = {"layers": layer_ids, "per_lens": {}}
    for key, arr in ranks.items():
        n_items, n_rows, _, n_tok = arr.shape
        per_token: Dict[str, Dict[str, List[float]]] = {}
        for t, name in enumerate(TRACKED):
            hit1_any, hit10_any, hit10_final, med_best = [], [], [], []
            for row in range(n_rows):
                best = np.array([arr[i, row, : lengths[i], t].min() for i in range(n_items)])
                final = np.array([arr[i, row, lengths[i] - 1, t] for i in range(n_items)])
                hit1_any.append(float((best == 1).mean()))
                hit10_any.append(float((best <= 10).mean()))
                hit10_final.append(float((final <= 10).mean()))
                med_best.append(float(np.median(best)))
            per_token[name] = {"hit1_any_pos": hit1_any, "hit10_any_pos": hit10_any, "hit10_final_pos": hit10_final, "median_best_rank": med_best}
        out["per_lens"][key] = per_token
    return out


def print_tables(summary: Dict[str, object]) -> None:
    layers = summary["layers"]
    for name in ("intermediate", "answer"):
        print(f"[readout] token={name}: per layer, fraction of items with rank-1 hit at any position (J / R / logit), and rank<=10 at any position")
        for li, layer in enumerate(layers):
            cells = []
            for key in [k for k in LENS_KEYS if k in summary["per_lens"]]:
                d = summary["per_lens"][key][name]
                cells.append(f"{key}: r1={d['hit1_any_pos'][li]:.2f} r10={d['hit10_any_pos'][li]:.2f} med={d['median_best_rank'][li]:.0f}")
            print(f"  L{layer:02d}  " + " | ".join(cells))
