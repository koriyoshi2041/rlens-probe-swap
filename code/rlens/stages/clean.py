"""Stage: clean baseline for all 90 items under both prompt modes."""
from __future__ import annotations

import json
from typing import Dict, List

from ..data import SwapItem
from ..forward import final_logprobs
from ..tokens import PROMPT_MODES, rank_of
from .common import resolve_item


def run(model, items: List[SwapItem], out_dir) -> List[Dict[str, object]]:
    tokenizer = model.tokenizer
    rows: List[Dict[str, object]] = []
    for mode in PROMPT_MODES:
        for item in items:
            resolved = resolve_item(tokenizer, item, mode)
            tokens = model.to_tokens(resolved.prompt, prepend_bos=False)
            logp = final_logprobs(model, tokens)
            top1 = int(logp.argmax().item())
            answer_id = resolved.answer.first_id
            swap_id = resolved.swap_answer.first_id
            rows.append(
                {
                    "index": item.index,
                    "name": item.name,
                    "category": item.category,
                    "mode": mode,
                    "n_prompt_tokens": int(tokens.shape[1]),
                    "last_prompt_tokens": model.to_str_tokens(resolved.prompt, prepend_bos=False)[-3:],
                    "top1_id": top1,
                    "top1_str": tokenizer.decode([top1]),
                    "answer_text": resolved.answer.text,
                    "answer_first_id": answer_id,
                    "swap_answer_text": resolved.swap_answer.text,
                    "swap_answer_first_id": swap_id,
                    "answers_single": resolved.answers_single,
                    "answers_distinct": resolved.answers_distinct,
                    "concept_feasible": resolved.concept_feasible,
                    "src_text": resolved.src.text,
                    "tgt_text": resolved.tgt.text,
                    "clean_correct": top1 == answer_id,
                    "logp_answer": float(logp[answer_id]),
                    "logp_swap_answer": float(logp[swap_id]),
                    "p_answer": float(logp[answer_id].exp()),
                    "rank_answer": rank_of(logp, answer_id),
                    "rank_swap_answer": rank_of(logp, swap_id),
                    "margin_swap_minus_answer": float(logp[swap_id] - logp[answer_id]),
                }
            )
    (out_dir / "clean_rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    summary = summarize(rows)
    (out_dir / "clean_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[clean] summary:", json.dumps(summary, indent=1, ensure_ascii=False))
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for mode in PROMPT_MODES:
        sub = [r for r in rows if r["mode"] == mode]
        correct = [r for r in sub if r["clean_correct"]]
        strict = [r for r in correct if r["concept_feasible"] and r["answers_single"] and r["answers_distinct"]]
        out[mode] = {
            "n_clean_correct": len(correct),
            "n_eligible_strict": len(strict),
            "median_p_answer_when_correct": sorted(r["p_answer"] for r in correct)[len(correct) // 2] if correct else None,
            "median_rank_swap_answer_when_correct": sorted(r["rank_swap_answer"] for r in correct)[len(correct) // 2] if correct else None,
            "incorrect": [{"name": r["name"], "top1": r["top1_str"], "answer": r["answer_text"], "rank_answer": r["rank_answer"]} for r in sub if not r["clean_correct"]],
        }
    return out
