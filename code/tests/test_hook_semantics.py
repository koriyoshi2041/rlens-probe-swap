"""GPU tests for two hook-semantics assumptions the code audit flagged as untested but load-bearing.

1. Cross-lens readout under an intervention runs ``run_with_cache`` inside ``model.hooks``;
   the cached activation at the intervened hook point must be the POST-intervention value,
   otherwise every "internal rewrite" number read inside the band would be pre-clamp.
2. The attention/MLP attribution (q25 S3) assumes ``hook_resid_pre -> hook_resid_mid ->
   hook_out`` are the residual stream before the attention sublayer, between the two
   sublayers, and after the MLP, on every layer including the linear-attention ones.
   We check residual continuity (``resid_pre[l] == out[l-1]``) and that the two halves
   are produced by the two sublayers (zeroing the MLP output leaves ``mid`` unchanged
   and makes ``out == mid``; the analogous statement for attention).
"""
from __future__ import annotations

import re

import pytest
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.interventions import clamp_hooks

LAYER = 12  # linear-attention layer inside the band
FULL_ATTN_LAYER = 11  # a full-attention layer inside the band
PROMPT = "Fact: The capital of the country where Lyon is located is"


@pytest.fixture(scope="module")
def tokens(model):
    return model.to_tokens(PROMPT, prepend_bos=False)


def _sublayer_hooks(model, layer):
    names = [n for n in model.hook_dict if n.startswith(f"blocks.{layer}.")]
    attn = [n for n in names if re.search(r"(attn|attention|linear_attn|gated_delta).*hook_out$", n)]
    mlp = [n for n in names if re.search(r"mlp.*hook_out$", n)]
    return names, attn, mlp


def test_cache_inside_hooks_context_sees_post_intervention_activation(model, lenses, tokens):
    lens = lenses["J"]
    name = _resid_post_hook_name(LAYER)
    ids = [model.tokenizer.encode(" France", add_special_tokens=False)[0], model.tokenizer.encode(" Italy", add_special_tokens=False)[0]]
    _, clean_cache = model.run_with_cache(tokens, names_filter=lambda x: x == name)
    clean = clean_cache[name].float()
    basis = lens.lens_vectors(model, ids, LAYER)
    pinv = torch.linalg.pinv(basis.T.float())
    coords = clean @ pinv.T
    hooks = clamp_hooks(model, basis, [LAYER], {LAYER: coords})
    with model.hooks(fwd_hooks=hooks):
        _, hooked_cache = model.run_with_cache(tokens, names_filter=lambda x: x == name)
    seen = hooked_cache[name].float()
    expected = clean + (coords[..., [1, 0]] - coords) @ basis.T.float().T
    assert not torch.allclose(seen, clean), "cache returned the pre-intervention activation"
    # bf16 storage of the hooked activation: compare at bf16 resolution
    rel = (seen - expected).norm() / expected.norm()
    assert rel < 5e-3, f"cached activation is neither clean nor the clamped value (rel err {rel:.3g})"


@pytest.mark.parametrize("layer", [LAYER, FULL_ATTN_LAYER])
def test_residual_stream_hooks_are_pre_mid_out(model, tokens, layer):
    pre, mid, out = f"blocks.{layer}.hook_resid_pre", f"blocks.{layer}.hook_resid_mid", f"blocks.{layer}.hook_out"
    prev_out = _resid_post_hook_name(layer - 1)
    wanted = {pre, mid, out, prev_out}
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    assert torch.equal(cache[pre], cache[prev_out]), "hook_resid_pre is not the previous block's hook_out"
    names, attn, mlp = _sublayer_hooks(model, layer)
    assert mlp, f"no MLP output hook found among {names}"

    def zero(act, hook):
        return torch.zeros_like(act)

    with model.hooks(fwd_hooks=[(mlp[0], zero)]):
        _, no_mlp = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    assert torch.equal(no_mlp[mid], cache[mid]), "zeroing the MLP changed resid_mid: mid is not pre-MLP"
    assert torch.allclose(no_mlp[out].float(), no_mlp[mid].float(), atol=1e-2, rtol=1e-2), "with the MLP zeroed, out != mid: MLP is not the out-mid half"
    if attn:
        with model.hooks(fwd_hooks=[(attn[0], zero)]):
            _, no_attn = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        assert torch.equal(no_attn[pre], cache[pre])
        assert torch.allclose(no_attn[mid].float(), no_attn[pre].float(), atol=1e-2, rtol=1e-2), "with attention zeroed, mid != pre: attention is not the mid-pre half"
    else:
        pytest.skip(f"no attention-output hook matched on layer {layer}; hooks: {names}")
