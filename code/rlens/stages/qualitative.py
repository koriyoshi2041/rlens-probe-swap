"""Stage: fixed random qualitative sample — top-10 tokens and greedy continuations under the headline swaps."""
from __future__ import annotations

import json
import math
import random
from typing import Dict, List

import torch

from .. import config_block02 as cfg
from ..data import SwapItem
from ..forward import final_logprobs
from ..interventions import EnergyStats, swap_hooks_with_stats
from .common import hybrid_rows, resolve_item


@torch.inference_mode()
def run(model, lenses, items: List[SwapItem], clean_rows, eligible: List[int], out_dir) -> List[Dict[str, object]]:
    rng = random.Random(cfg.QUAL_SEED)
    sample = sorted(rng.sample(eligible, min(cfg.QUAL_N, len(eligible))))
    modes = hybrid_rows(clean_rows)
    tokenizer = model.tokenizer
    rows: List[Dict[str, object]] = []
    for index in sample:
        r = resolve_item(tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        n_prompt = tokens.shape[1]
        ids = (r.src.first_id, r.tgt.first_id)
        conditions = {"clean": None}
        for band_name, layers in cfg.BANDS.items():
            for lens_key in cfg.LENSES_CONTROL:
                conditions[f"{lens_key}_{band_name}"] = (lens_key, layers)
        for cond_name, spec in conditions.items():
            hooks = []
            if spec is not None:
                lens_key, layers = spec
                for layer in layers:
                    basis = lenses[lens_key].lens_vectors(model, list(ids), layer)
                    hooks += swap_hooks_with_stats(model, basis, [layer], alpha=cfg.HEADLINE_ALPHA, positions=list(range(n_prompt)), stats=EnergyStats())
            logp = final_logprobs(model, tokens, hooks)
            top = torch.topk(logp, 10)
            generated = tokens
            for _ in range(cfg.GEN_TOKENS):
                step = final_logprobs(model, generated, hooks)
                generated = torch.cat([generated, step.argmax().view(1, 1)], dim=1)
            rows.append(
                {
                    "index": index,
                    "name": r.item.name,
                    "prompt": r.prompt,
                    "condition": cond_name,
                    "answer": r.answer.text,
                    "swap_answer": r.swap_answer.text,
                    "top10": [[tokenizer.decode([i]), round(math.exp(v), 4)] for v, i in zip(top.values.tolist(), top.indices.tolist())],
                    "continuation": tokenizer.decode(generated[0, n_prompt:].tolist()),
                }
            )
            print(f"[qual] {r.item.name:32s} {cond_name:20s} top1={rows[-1]['top10'][0]} cont={rows[-1]['continuation']!r}")
    (out_dir / "qualitative.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows
