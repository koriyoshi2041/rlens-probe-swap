"""CPU-only tests for token resolution, ranks, and item loading."""
import torch

from rlens.data import leaks_in_prompt, load_items, reverse_pairs
from rlens.tokens import normalize_prompt, rank_of, ranks_rowwise, resolve_next_token


def test_rank_of_orders_descending():
    logits = torch.tensor([0.1, 3.0, 2.0])
    assert rank_of(logits, 1) == 1
    assert rank_of(logits, 2) == 2
    assert rank_of(logits, 0) == 3


def test_ranks_rowwise_matches_rank_of():
    logits = torch.randn(4, 50)
    ids = (3, 17, 42)
    ranks = ranks_rowwise(logits, ids)
    for pos in range(4):
        for k, tid in enumerate(ids):
            assert int(ranks[pos, k]) == rank_of(logits[pos], tid)


def test_normalize_prompt_modes():
    assert normalize_prompt("abc ", "as_is") == "abc "
    assert normalize_prompt("abc ", "rstrip") == "abc"


def test_resolve_next_token_spacing(tokenizer):
    spaced = resolve_next_token(tokenizer, "The capital of France is", "Paris")
    bare = resolve_next_token(tokenizer, "The capital of France is ", "Paris")
    assert spaced.text == " Paris"
    assert bare.text == "Paris"
    assert spaced.first_id != bare.first_id


def test_load_items_is_90_with_fields():
    items = load_items()
    assert len(items) == 90
    first = items[0]
    assert first.name == "amazon-language"
    assert first.intermediate == "Brazil" and first.swap_to == "Mexico"
    assert isinstance(leaks_in_prompt(first)["intermediate"], bool)
    assert all(isinstance(p, list) and len(p) == 2 for p in reverse_pairs(items))
