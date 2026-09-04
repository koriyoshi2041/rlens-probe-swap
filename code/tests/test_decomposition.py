"""The swap is exactly remove + install, and the coordinate map's eigenvalues are (1, 1-2*alpha)."""
import pytest
import torch

from rlens.interventions import EnergyStats, coordinate_map_hooks, swap_hooks_with_stats

PROMPT = "Fact: The capital of the country where Lyon is located is"
SRC, TGT = " France", " Italy"
LAYER = 12


def test_coordinate_map_eigenvalues():
    """M(alpha) = I + alpha(S - I) has eigenvalues 1 (symmetric) and 1-2*alpha (antisymmetric)."""
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    for alpha in (0.5, 1.0, 2.0, 4.0):
        m = torch.eye(2) + alpha * (swap - torch.eye(2))
        eigs = sorted(float(e.real) for e in torch.linalg.eigvals(m))
        assert pytest.approx(sorted([1.0, 1.0 - 2 * alpha]), abs=1e-5) == eigs


def test_swap_equals_remove_plus_install_in_fp32(model, lenses):
    """swap = remove + install exactly, checked on the fp32 deltas the hooks compute.

    The identity is exact in fp32: with c = V^+ h,
        remove  = (c_t - c_s) v_s,  install = -(c_t - c_s) v_t,
        swap    = (c_t - c_s) (v_s - v_t) = remove + install.
    It cannot be verified to fp32 precision through captured activations: the hook
    casts its result back to bf16, whose ~0.4% relative step is taken against ||h||
    while the perturbation is only 5-12% of ||h||, so the rounding error on the
    delta is amplified about tenfold. test_decomposition_through_bf16_is_bounded
    measures that amplified error instead.
    """
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    basis = lenses["R"].lens_vectors(model, ids, LAYER)
    name = f"blocks.{LAYER}.hook_out"
    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == name)
        h = cache[name].float()
    matrix = basis.T.float()
    pinv = torch.linalg.pinv(matrix)
    coords = h @ pinv.T
    gap = (coords[..., 1] - coords[..., 0]).unsqueeze(-1)
    v_s, v_t = matrix[:, 0], matrix[:, 1]
    d_swap, d_remove, d_install = gap * (v_s - v_t), gap * v_s, -gap * v_t
    torch.testing.assert_close(d_swap, d_remove + d_install, rtol=1e-5, atol=1e-5)
    # and the coordinates land where the names say they do
    for delta, expected in ((d_swap, (1, 0)), (d_remove, (1, 1)), (d_install, (0, 0))):
        new_coords = (h + delta) @ pinv.T
        torch.testing.assert_close(new_coords[..., 0], coords[..., expected[0]], rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(new_coords[..., 1], coords[..., expected[1]], rtol=2e-3, atol=2e-3)


def test_decomposition_through_bf16_is_bounded(model, lenses):
    """Through the bf16 activation path the identity holds to the expected rounding budget."""
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    basis = lenses["R"].lens_vectors(model, ids, LAYER)
    name = f"blocks.{LAYER}.hook_out"
    captured = {}

    def recorder(key):
        def fn(act, hook):
            captured[key] = act.clone().float()
            return act

        return fn

    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == name)
        clean = cache[name].float()
        for mode in ("swap", "remove", "install"):
            hooks = coordinate_map_hooks(model, basis, [LAYER], mode=mode)
            with model.hooks(fwd_hooks=hooks + [(name, recorder(mode))]):
                model(tokens, return_type="logits")
    d_swap = captured["swap"] - clean
    d_sum = (captured["remove"] - clean) + (captured["install"] - clean)
    rel = float((d_swap - d_sum).norm() / d_swap.norm())
    amplification = float(clean.norm() / d_swap.norm())
    budget = 3 * 2**-8 * amplification  # three bf16 roundings, each ~2^-8 of ||h||
    assert rel < budget, f"relative mismatch {rel:.2e} exceeds bf16 budget {budget:.2e}"


def test_swap_mode_matches_pseudoinverse_implementation(model, lenses):
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    basis = lenses["J"].lens_vectors(model, ids, LAYER)
    with torch.inference_mode():
        with model.hooks(fwd_hooks=swap_hooks_with_stats(model, basis, [LAYER], alpha=1.0, stats=EnergyStats())):
            a = model(tokens, return_type="logits")
        with model.hooks(fwd_hooks=coordinate_map_hooks(model, basis, [LAYER], mode="swap", alpha=1.0)):
            b = model(tokens, return_type="logits")
    torch.testing.assert_close(a.float(), b.float(), rtol=1e-3, atol=5e-3)


def test_remove_and_install_are_idempotent_but_alpha2_is_not(model, lenses):
    """Remove/install settle after one application; alpha=2 swap grows across repeated layers."""
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    lens = lenses["J"]
    layers = list(range(8, 21))
    energies = {}
    with torch.inference_mode():
        for label, mode, alpha in (("remove", "remove", 1.0), ("install", "install", 1.0), ("swap1", "swap", 1.0), ("swap2", "swap", 2.0)):
            stats = EnergyStats()
            hooks = []
            for layer in layers:
                hooks += coordinate_map_hooks(model, lens.lens_vectors(model, ids, layer), [layer], mode=mode, alpha=alpha, stats=stats)
            with model.hooks(fwd_hooks=hooks):
                model(tokens, return_type="logits")
            energies[label] = stats
    last, first = layers[-1], layers[0]
    for label in ("remove", "install", "swap1"):
        growth = energies[label].sum_dh2[last] / energies[label].sum_dh2[first]
        assert growth < 50, f"{label} grew {growth:.1f}x across the band"
    growth2 = energies["swap2"].sum_dh2[last] / energies["swap2"].sum_dh2[first]
    assert growth2 > 100, f"alpha=2 only grew {growth2:.1f}x"
