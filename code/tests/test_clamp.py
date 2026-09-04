"""A clamp holds the swapped coordinates across a band; the involution alternates.

Note on the metric. An earlier version of this test measured, per position, how far
the source coordinate had moved toward the target as a fraction of the two clean
coordinates' gap. That blows up: at most token positions the two lens coordinates
are nearly equal, so the denominator is ~0 and the fraction is meaningless. The
tests below assert the algebraic property directly and restrict the alternation
metric to positions whose clean gap is above the median.
"""
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.interventions import clamp_hooks, coordinate_map_hooks

PROMPT = "Fact: The capital of the country where Lyon is located is"
SRC, TGT = " France", " Italy"
BAND = list(range(8, 21))


def _setup(model, lens, tokens, ids, layers):
    names = {layer: _resid_post_hook_name(layer) for layer in layers}
    wanted = set(names.values())
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    coords, bases = {}, {}
    for layer in layers:
        basis = lens.lens_vectors(model, ids, layer)
        bases[layer] = basis
        pinv = torch.linalg.pinv(basis.T.float())
        coords[layer] = cache[names[layer]].float() @ pinv.T
    return names, coords, bases


def _run_capturing(model, tokens, hooks, names):
    captured = {}

    def recorder(name):
        def fn(act, hook):
            captured[name] = act.clone().float()
            return act

        return fn

    probes = [(name, recorder(name)) for name in names.values()]
    with model.hooks(fwd_hooks=hooks + probes):
        model(tokens, return_type="logits")
    return captured


def test_clamp_sets_coordinates_to_the_exchanged_clean_values(model, lenses):
    lens = lenses["J"]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    with torch.inference_mode():
        names, coords, bases = _setup(model, lens, tokens, ids, BAND)
        hooks = []
        for layer in BAND:
            hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]})
        captured = _run_capturing(model, tokens, hooks, names)
        for layer in BAND:
            pinv = torch.linalg.pinv(bases[layer].T.float())
            got = captured[names[layer]] @ pinv.T
            want = coords[layer][..., [1, 0]]
            rel = float((got - want).abs().max() / want.abs().max())
            assert rel < 5e-2, f"layer {layer}: clamp missed its target by {rel:.3f} relative"


def test_involution_alternates_but_clamp_and_install_do_not(model, lenses):
    lens = lenses["J"]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    swings = {}
    with torch.inference_mode():
        names, coords, bases = _setup(model, lens, tokens, ids, BAND)
        for arm in ("swap", "clamp", "install"):
            hooks = []
            for layer in BAND:
                if arm == "clamp":
                    hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]})
                else:
                    hooks += coordinate_map_hooks(model, bases[layer], [layer], mode=arm, alpha=1.0)
            captured = _run_capturing(model, tokens, hooks, names)
            fractions = []
            for layer in BAND:
                pinv = torch.linalg.pinv(bases[layer].T.float())
                got = (captured[names[layer]] @ pinv.T)[0]
                clean = coords[layer][0]
                gap = clean[:, 1] - clean[:, 0]
                keep = gap.abs() > gap.abs().median()  # drop positions where the coordinates coincide
                frac = ((got[keep, 0] - clean[keep, 0]) / gap[keep]).median()
                fractions.append(float(frac))
            swings[arm] = max(fractions) - min(fractions)
    assert swings["clamp"] < 0.3, f"clamp swung {swings['clamp']:.2f}"
    assert swings["install"] < 0.3, f"install swung {swings['install']:.2f}"
    assert swings["swap"] > 2 * max(swings["clamp"], swings["install"]), (
        f"the involution should alternate far more: swap {swings['swap']:.2f} vs "
        f"clamp {swings['clamp']:.2f}, install {swings['install']:.2f}"
    )


def test_scale_one_clamp_is_the_plain_clamp_not_a_no_op(model, lenses):
    """Regression for the ladder bug.

    Building targets as ``c + scale * (exchanged - c)`` and then passing them to a
    clamp that exchanges again cancels the intervention exactly: at scale 1 the
    ladder reported zero effect and 1e-13 energy. The ``exchange`` flag exists so
    a caller that already built its targets does not pay the exchange twice.

    The assertion is on the quantity the study reports (the log-prob margin at the
    final position) rather than on raw logits. The two code paths are algebraically
    identical but not bitwise so: one forms ``c[..., [1, 0]]`` directly and the
    other computes ``c + 1.0 * (c[..., [1, 0]] - c)``, and that fp32 difference is
    cast to bf16 at thirteen intervened layers, which moves individual logits by up
    to ~0.1 out of a range of ~30.
    """
    lens = lenses["J"]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    answer = model.to_single_token(" Paris")
    swap_answer = model.to_single_token(" Rome")

    def margin(logits):
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        return float(logp[swap_answer] - logp[answer])

    with torch.inference_mode():
        names, coords, bases = _setup(model, lens, tokens, ids, BAND)
        built = {"base": [], "scaled": [], "double_exchanged": []}
        for layer in BAND:
            exchanged = coords[layer][..., [1, 0]]
            target = coords[layer] + 1.0 * (exchanged - coords[layer])
            built["base"] += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]})
            built["scaled"] += clamp_hooks(model, bases[layer], [layer], {layer: target}, exchange=False)
            built["double_exchanged"] += clamp_hooks(model, bases[layer], [layer], {layer: target})
        clean = margin(model(tokens, return_type="logits"))
        got = {}
        for name, hooks in built.items():
            with model.hooks(fwd_hooks=hooks):
                got[name] = margin(model(tokens, return_type="logits"))
    assert got["base"] - clean > 1.0, f"the clamp should move the margin, got {got['base'] - clean:+.3f}"
    assert abs(got["scaled"] - got["base"]) < 0.1, (
        f"exchange=False with pre-built targets must match the plain clamp: "
        f"{got['scaled']:.3f} vs {got['base']:.3f}"
    )
    assert abs(got["double_exchanged"] - clean) < 0.05, (
        f"exchanging twice must cancel to a no-op, got {got['double_exchanged'] - clean:+.4f}"
    )


def test_swapped_fraction_handles_negative_gaps(model, lenses):
    """Regression: clamping the denominator turned every negative gap into +1e-6.

    That produced 'swapped fractions' in the hundreds of thousands in the first
    depth-recovery run. The metric must read ~0 on the clean state and ~1 on the
    fully clamped one, whichever sign the gap has.
    """
    from rlens.interventions import swapped_fraction

    lens = lenses["J"]
    tokens = model.to_tokens(PROMPT, prepend_bos=False)
    ids = [model.to_single_token(SRC), model.to_single_token(TGT)]
    with torch.inference_mode():
        names, coords, bases = _setup(model, lens, tokens, ids, BAND)
        layer = BAND[0]
        clean = coords[layer]
        assert bool(((clean[..., 1] - clean[..., 0]) < 0).any()), "test needs some negative gaps"
        assert abs(swapped_fraction(clean, clean)) < 1e-4
        exchanged = clean[..., [1, 0]]
        assert abs(swapped_fraction(exchanged, clean) - 1.0) < 1e-3
        half = clean + 0.5 * (exchanged - clean)
        assert abs(swapped_fraction(half, clean) - 0.5) < 1e-3
