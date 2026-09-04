"""GPU tests: the library hook path does exactly what the paper's formulas say.

These are the block-01 qualification checks: alpha=0 is a no-op, source==target
fails closed, the pseudoinverse swap matches a hand-written least-squares version,
positions are respected, lens vectors follow v_t = J^T W_U[:, t], and the final
block output + ln_final + unembed reproduces the model's own logits.
"""
import pytest
import torch
from transformer_lens.tools.analysis.jacobian_lens import _unembed

PROMPT = "Fact: The language spoken in the country where the Amazon River ends is"
SRC, TGT = " Brazil", " Mexico"
LAYER = 12


def _tokens(model):
    return model.to_tokens(PROMPT, prepend_bos=False)


def test_validate_model_accepts_both_lenses(model, lenses):
    for lens in lenses.values():
        lens.validate_model(model)


def test_forward_is_deterministic(model):
    tokens = _tokens(model)
    with torch.inference_mode():
        a = model(tokens, return_type="logits")
        b = model(tokens, return_type="logits")
    assert torch.equal(a, b)


def test_final_block_output_reproduces_model_logits(model):
    final = model.cfg.n_layers - 1
    name = f"blocks.{final}.hook_out"
    tokens = _tokens(model)
    with torch.inference_mode():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n == name)
        by_hand = _unembed(model, cache[name][0])
    diff = (by_hand - logits[0].float()).abs().max().item()
    assert diff < 0.05, f"max |diff| = {diff}"


@pytest.mark.parametrize("kind", ["J", "R"])
def test_lens_vector_formula(model, lenses, kind):
    lens = lenses[kind]
    tid = model.to_single_token(SRC)
    got = lens.lens_vectors(model, tid, LAYER)[0]
    matrix = lens.jacobians[LAYER].to(device=got.device, dtype=torch.float32)
    expected = matrix.T @ model.W_U[:, tid].float()
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["J", "R"])
def test_alpha_zero_swap_equals_clean(model, lenses, kind):
    tokens = _tokens(model)
    hooks = lenses[kind].swap_hooks(model, SRC, TGT, layers=[LAYER, LAYER + 8], alpha=0.0)
    with torch.inference_mode():
        clean = model(tokens, return_type="logits")
        with model.hooks(fwd_hooks=hooks):
            swapped = model(tokens, return_type="logits")
    assert torch.equal(clean, swapped)


def test_source_equals_target_fails_closed(model, lenses):
    with pytest.raises(ValueError):
        lenses["J"].swap_hooks(model, SRC, SRC, layers=[LAYER])


@pytest.mark.parametrize("kind", ["J", "R"])
@pytest.mark.parametrize("positions", [None, [-1], [3, 5]])
def test_swap_matches_handwritten_least_squares(model, lenses, kind, positions):
    lens = lenses[kind]
    alpha = 1.0
    tokens = _tokens(model)
    name = f"blocks.{LAYER}.hook_out"
    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == name)
        clean_act = cache[name].clone()  # [1, pos, d] bf16
        ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
        basis = lens.lens_vectors(model, ids, LAYER).T  # [d, 2] fp32
        gram = basis.T @ basis
        h = clean_act.float()
        coords = (h @ basis) @ torch.linalg.inv(gram)  # least-squares coordinates [1, pos, 2]
        delta = alpha * ((coords[..., [1, 0]] - coords) @ basis.T)
        expected = h.clone()
        seq = h.shape[1]
        sel = list(range(seq)) if positions is None else [p + seq if p < 0 else p for p in positions]
        expected[:, sel, :] = h[:, sel, :] + delta[:, sel, :]

        hooks = lens.swap_hooks(model, SRC, TGT, layers=[LAYER], alpha=alpha, positions=positions)
        captured = {}

        def recorder(act, hook):
            captured["act"] = act.clone()
            return act

        with model.hooks(fwd_hooks=hooks + [(name, recorder)]):
            model(tokens, return_type="logits")
    got = captured["act"].float()
    torch.testing.assert_close(got, expected, rtol=4e-3, atol=2e-2)
    exact = (captured["act"] == expected.to(captured["act"].dtype)).float().mean().item()
    assert exact > 0.99, f"only {exact:.4f} of elements bitwise-equal after bf16 cast"
    untouched = [p for p in range(seq) if p not in sel]
    if untouched:
        assert torch.equal(captured["act"][:, untouched, :], clean_act[:, untouched, :])
