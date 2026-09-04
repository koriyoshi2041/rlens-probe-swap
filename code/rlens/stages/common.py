"""Shared helpers for stages: per-item token resolution and eligibility."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..data import SwapItem
from ..tokens import TokenResolution, normalize_prompt, resolve_concept_token, resolve_next_token


@dataclass(frozen=True)
class ResolvedItem:
    item: SwapItem
    mode: str
    prompt: str
    src: TokenResolution
    tgt: TokenResolution
    answer: TokenResolution
    swap_answer: TokenResolution

    @property
    def concept_feasible(self) -> bool:
        return self.src.single and self.tgt.single

    @property
    def answers_distinct(self) -> bool:
        return self.answer.first_id != self.swap_answer.first_id

    @property
    def answers_single(self) -> bool:
        return self.answer.single and self.swap_answer.single

    @property
    def tracked_ids(self) -> List[int]:
        return [self.src.first_id, self.tgt.first_id, self.answer.first_id, self.swap_answer.first_id]


def resolve_item(tokenizer, item: SwapItem, mode: str) -> ResolvedItem:
    prompt = normalize_prompt(item.prompt, mode)
    return ResolvedItem(
        item=item,
        mode=mode,
        prompt=prompt,
        src=resolve_concept_token(tokenizer, item.intermediate),
        tgt=resolve_concept_token(tokenizer, item.swap_to),
        answer=resolve_next_token(tokenizer, prompt, item.answer),
        swap_answer=resolve_next_token(tokenizer, prompt, item.swap_answer),
    )


def choose_mode(clean_rows: List[Dict[str, object]]) -> str:
    """Provisional rule: keep the published prompts verbatim unless stripping is strictly better."""
    correct = {mode: sum(1 for r in clean_rows if r["mode"] == mode and r["clean_correct"]) for mode in ("as_is", "rstrip")}
    return "rstrip" if correct["rstrip"] > correct["as_is"] else "as_is"


def eligible_indices(clean_rows: List[Dict[str, object]], mode: str) -> List[int]:
    """Clean-correct, concept-feasible, single-token, distinct answer tokens, in file order."""
    return [
        int(r["index"])
        for r in clean_rows
        if r["mode"] == mode
        and r["clean_correct"]
        and r["concept_feasible"]
        and r["answers_single"]
        and r["answers_distinct"]
    ]


def clean_row_lookup(clean_rows: List[Dict[str, object]], mode: str) -> Dict[int, Dict[str, object]]:
    return {int(r["index"]): r for r in clean_rows if r["mode"] == mode}


def hybrid_rows(clean_rows: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    """Per-item prompt mode: strip trailing space unless the answers then stop being single/distinct tokens."""
    by = {(int(r["index"]), r["mode"]): r for r in clean_rows}
    out: Dict[int, Dict[str, object]] = {}
    for index in sorted({int(r["index"]) for r in clean_rows}):
        stripped = by[(index, "rstrip")]
        out[index] = stripped if (stripped["answers_single"] and stripped["answers_distinct"]) else by[(index, "as_is")]
    return out


def eligible_hybrid(clean_rows: List[Dict[str, object]]) -> List[int]:
    return [
        index
        for index, r in hybrid_rows(clean_rows).items()
        if r["clean_correct"] and r["concept_feasible"] and r["answers_single"] and r["answers_distinct"]
    ]
