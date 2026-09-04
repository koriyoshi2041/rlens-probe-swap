"""Own intervention hooks must match the library's, and the control bases must have the claimed geometry."""
import pytest
import torch

from rlens.interventions import (
    EnergyStats,
    ablation_hooks_with_stats,
    gram_matched_random_basis,
    swap_hooks_with_stats,
    unit_basis,
)

PROMPT = "Fact: The capital of the country where Hungarian is the primary language is"
SRC, TGT = " Hungary", " Poland"
LAYERS = [8, 12, 16]


def test_gram_matched_random_basis_has_same_gram():
    torch.manual_seed(0)
    basis = torch.randn(2, 64) * torch.tensor([[3.0], [0.5]])
    random_basis = gram_matched_random_basis(basis, seed=1)
    torch.testing.assert_close(random_basis @ random_basis.T, basis @ basis.T, rtol=1e-4, atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(random_basis[0:1], basis[0:1])
    assert abs(float(cos)) < 0.5
    assert torch.equal(gram_matched_random_basis(basis, seed=1), random_basis)


def test_unit_basis_rows_have_unit_norm():
    basis = torch.randn(2, 16) * 7
    torch.testing.assert_close(unit_basis(basis).norm(dim=-1), torch.ones(2))


@pytest.mark.parametrize("kind", ["J", "R"])
@pytest.mark.parametrize("positions", [None, [-2, -1]])
def test_own_swap_matches_library(model, lenses, kind, positions):
    lens = lenses[kind]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    basis = lens.lens_vectors(model, ids, LAYERS[0])  # per-layer bases differ; compare one layer at a time
    with torch.inference_mode():
        lib = lens.swap_hooks(model, SRC, TGT, layers=[LAYERS[0]], alpha=1.0, positions=positions)
        with model.hooks(fwd_hooks=lib):
            lib_logits = model(tokens, return_type="logits")
        stats = EnergyStats()
        own = swap_hooks_with_stats(model, basis, [LAYERS[0]], alpha=1.0, positions=positions, stats=stats)
        with model.hooks(fwd_hooks=own):
            own_logits = model(tokens, return_type="logits")
    assert torch.equal(lib_logits, own_logits)
    assert stats.total_dh2 > 0 and set(stats.sum_dh2) == {LAYERS[0]}


def test_own_ablation_matches_library(model, lenses):
    lens = lenses["J"]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    sid = model.to_single_token(SRC)
    with torch.inference_mode():
        lib = lens.ablation_hooks(model, [sid], layers=LAYERS)
        with model.hooks(fwd_hooks=lib):
            lib_logits = model(tokens, return_type="logits")
        stats = EnergyStats()
        own = []
        for layer in LAYERS:
            vec = lens.lens_vectors(model, [sid], layer)
            own += ablation_hooks_with_stats(model, vec, [layer], stats=stats)
        with model.hooks(fwd_hooks=own):
            own_logits = model(tokens, return_type="logits")
    assert torch.equal(lib_logits, own_logits)
    assert set(stats.sum_dh2) == set(LAYERS)


def test_swap_energy_is_alpha_scaled(model, lenses):
    lens = lenses["R"]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    basis = lens.lens_vectors(model, ids, 10)
    totals = {}
    with torch.inference_mode():
        for alpha in (1.0, 2.0):
            stats = EnergyStats()
            with model.hooks(fwd_hooks=swap_hooks_with_stats(model, basis, [10], alpha=alpha, stats=stats)):
                model(tokens, return_type="logits")
            totals[alpha] = stats.total_dh2
    assert abs(totals[2.0] / totals[1.0] - 4.0) < 1e-3
