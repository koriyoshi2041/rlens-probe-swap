"""Stage: single-hop knowledge ceiling. Does the model produce answer from intermediate and swap_answer from swap_to directly?"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List

import torch

from ..data import SwapItem
from ..forward import final_logprobs
from ..tokens import rank_of, resolve_next_token

TEMPLATES_FILE = pathlib.Path(__file__).resolve().parents[2] / "data" / "single_hop_templates.json"


@torch.inference_mode()
def run(model, items: List[SwapItem], eligible: List[int], out_dir) -> List[Dict[str, object]]:
    templates = json.loads(TEMPLATES_FILE.read_text(encoding="utf-8"))
    tokenizer = model.tokenizer
    rows: List[Dict[str, object]] = []
    for index in eligible:
        item = items[index]
        template = templates[str(index)]
        for side, entity, expected in (("intermediate", item.intermediate, item.answer), ("swap_to", item.swap_to, item.swap_answer)):
            prompt = template.format(entity)
            tokens = model.to_tokens(prompt, prepend_bos=False)
            logp = final_logprobs(model, tokens)
            target = resolve_next_token(tokenizer, prompt, expected)
            top1 = int(logp.argmax().item())
            rows.append(
                {
                    "index": index,
                    "name": item.name,
                    "side": side,
                    "prompt": prompt,
                    "expected": target.text,
                    "expected_single": target.single,
                    "top1": tokenizer.decode([top1]),
                    "correct": top1 == target.first_id,
                    "logp_expected": float(logp[target.first_id]),
                    "rank_expected": rank_of(logp, target.first_id),
                }
            )
    (out_dir / "knowledge.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    summary = {
        side: {
            "n": sum(1 for r in rows if r["side"] == side),
            "n_correct": sum(1 for r in rows if r["side"] == side and r["correct"]),
            "n_rank_le5": sum(1 for r in rows if r["side"] == side and r["rank_expected"] <= 5),
            "failures": [(r["name"], r["top1"], r["expected"], r["rank_expected"]) for r in rows if r["side"] == side and not r["correct"]],
        }
        for side in ("intermediate", "swap_to")
    }
    print("[knowledge]", json.dumps(summary, ensure_ascii=False, indent=1))
    return rows
