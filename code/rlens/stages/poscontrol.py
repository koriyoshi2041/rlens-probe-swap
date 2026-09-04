"""Stage: positive control — steer toward the swap_answer token; does the hook path move the target logit?"""
from __future__ import annotations

import json
from typing import Dict, List

import torch

from ..data import SwapItem
from ..forward import final_logprobs, kl_divergence
from .common import choose_mode, eligible_indices, resolve_item

ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
LAYER_SETS = {"mid": [10, 12, 14, 16], "late": [24, 26, 28, 30]}
POSITION_SETS = {"all": None, "final": [-1]}
N_ITEMS = 2


@torch.inference_mode()
def run(model, lenses: Dict[str, object], items: List[SwapItem], clean_rows, out_dir) -> List[Dict[str, object]]:
    mode = choose_mode(clean_rows)
    chosen = eligible_indices(clean_rows, mode)[:N_ITEMS]
    tokenizer = model.tokenizer
    rows: List[Dict[str, object]] = []
    for index in chosen:
        r = resolve_item(tokenizer, items[index], mode)
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        clean_margin = float(clean[s] - clean[a])
        for kind, lens in lenses.items():
            for set_name, layers in LAYER_SETS.items():
                for pos_name, positions in POSITION_SETS.items():
                    for alpha in ALPHAS:
                        hooks = lens.steering_hooks(model, s, layers, alpha=alpha, positions=positions)
                        logp = final_logprobs(model, tokens, hooks)
                        top1 = int(logp.argmax().item())
                        rows.append(
                            {
                                "index": index,
                                "name": r.item.name,
                                "lens": kind,
                                "layers": set_name,
                                "positions": pos_name,
                                "alpha": alpha,
                                "logp_answer": float(logp[a]),
                                "logp_swap_answer": float(logp[s]),
                                "delta_margin": float(logp[s] - logp[a]) - clean_margin,
                                "top1": tokenizer.decode([top1]),
                                "kl_clean_to_steered": kl_divergence(clean, logp),
                            }
                        )
                        last = rows[-1]
                        print(
                            f"[poscontrol] {r.item.name} {kind} {set_name} {pos_name} a={alpha:<4} "
                            f"Δmargin={last['delta_margin']:+.2f} top1={last['top1']!r} KL={last['kl_clean_to_steered']:.3f}"
                        )
    (out_dir / "poscontrol.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows
