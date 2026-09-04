"""Loading and auditing the 90 probe-swap items (no model needed)."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from .paths import ITEMS_FILE

CONCEPT_FIELDS = ("intermediate", "swap_to", "answer", "swap_answer")
EXPECTED_ITEM_COUNT = 90


@dataclass(frozen=True)
class SwapItem:
    index: int
    name: str
    category: str
    prompt: str
    intermediate: str
    answer: str
    swap_to: str
    swap_answer: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items(path: pathlib.Path = ITEMS_FILE) -> List[SwapItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = payload["items"]
    if len(raw_items) != EXPECTED_ITEM_COUNT:
        raise RuntimeError(f"expected {EXPECTED_ITEM_COUNT} items, found {len(raw_items)}")
    return [
        SwapItem(
            index=i,
            name=item["name"],
            category=item["category"],
            prompt=item["prompt"],
            intermediate=item["intermediate"],
            answer=item["answer"],
            swap_to=item["swap_to"],
            swap_answer=item["swap_answer"],
        )
        for i, item in enumerate(raw_items)
    ]


def _word_in(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text, flags=re.IGNORECASE) is not None


def leaks_in_prompt(item: SwapItem) -> Dict[str, bool]:
    """Whether each concept string literally appears in the prompt (whole word, case-insensitive)."""
    return {field: _word_in(item.prompt, getattr(item, field)) for field in CONCEPT_FIELDS}


def reverse_pairs(items: List[SwapItem]) -> List[List[int]]:
    """Index pairs (i, j) where item j swaps in the direction opposite to item i."""
    key = {(it.intermediate.lower(), it.swap_to.lower()): it.index for it in items}
    seen = set()
    pairs: List[List[int]] = []
    for it in items:
        other = key.get((it.swap_to.lower(), it.intermediate.lower()))
        if other is not None and other != it.index:
            pair = tuple(sorted((it.index, other)))
            if pair not in seen:
                seen.add(pair)
                pairs.append(list(pair))
    return pairs
