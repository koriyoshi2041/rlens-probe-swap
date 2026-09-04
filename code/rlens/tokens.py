"""Token resolution rules and rank helper.

Rule for scoring a continuation word: if the prompt ends with whitespace the next
token is the bare word; otherwise the next token carries a leading space. This is
what "prompt ends just before the answer; greedy next-token == answer" means for a
BPE tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch

PROMPT_MODES = ("as_is", "rstrip")


@dataclass(frozen=True)
class TokenResolution:
    text: str
    ids: Tuple[int, ...]
    first_id: int
    single: bool


def encode_variant(tokenizer: Any, text: str) -> TokenResolution:
    ids = tuple(int(i) for i in tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError(f"{text!r} encodes to zero tokens")
    return TokenResolution(text=text, ids=ids, first_id=ids[0], single=len(ids) == 1)


def resolve_next_token(tokenizer: Any, prompt: str, word: str) -> TokenResolution:
    """The token whose probability scores `word` as the continuation of `prompt`."""
    variant = word if prompt.endswith((" ", "\n", "\t")) else " " + word
    return encode_variant(tokenizer, variant)


def resolve_concept_token(tokenizer: Any, word: str) -> TokenResolution:
    """Single-token form of a concept for lens interventions.

    Prefer the leading-space form (how the word appears mid-sentence); fall back to
    the bare form. If neither is a single token the result has ``single=False``.
    """
    spaced = encode_variant(tokenizer, " " + word)
    if spaced.single:
        return spaced
    bare = encode_variant(tokenizer, word)
    if bare.single:
        return bare
    return spaced


def normalize_prompt(prompt: str, mode: str) -> str:
    if mode == "as_is":
        return prompt
    if mode == "rstrip":
        return prompt.rstrip()
    raise ValueError(f"unknown prompt mode {mode!r}; expected one of {PROMPT_MODES}")


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of ``token_id`` in a 1-D logits vector (ties count as better)."""
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    return int((logits > logits[token_id]).sum().item()) + 1


def ranks_rowwise(logits: torch.Tensor, token_ids: Tuple[int, ...]) -> torch.Tensor:
    """Ranks for several tokens in a ``[pos, vocab]`` logits matrix -> ``[pos, n]`` (1-indexed)."""
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {tuple(logits.shape)}")
    ids = torch.tensor(list(token_ids), device=logits.device)
    own = logits[:, ids]  # [pos, n]
    return (logits.unsqueeze(-1) > own.unsqueeze(1)).sum(dim=1) + 1
