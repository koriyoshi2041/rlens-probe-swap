"""Our column-wise estimator must equal the reference row-wise fit on the same prompt."""
import torch

from rlens.fit_vectors import lens_vectors_for_prompt

PASSAGE = (
    "The history of the region is documented in several surviving manuscripts, which describe "
    "trade routes, harvests, and the succession of local rulers over more than two centuries. "
    "Modern scholarship has revised many of the earlier estimates of population and yield."
)
LAYER = 12


def test_matches_reference_fit_on_one_prompt(model):
    from transformer_lens.tools.analysis import JacobianLens

    tokens = model.to_tokens(PASSAGE)[:, :64]
    assert tokens.shape[1] > 17, "passage must exceed the skipped prefix"
    reference = JacobianLens.fit(
        model, [PASSAGE], corpus="parity-test", source_layers=[LAYER],
        dim_batch=64, max_seq_len=64, show_progress=False,
    )
    ids = [model.to_single_token(" France"), model.to_single_token(" Italy")]
    expected = reference.lens_vectors(model, ids, LAYER).cpu()
    cotangents = model.W_U[:, ids].float().T.contiguous()
    with torch.enable_grad():
        got = lens_vectors_for_prompt(model, tokens, cotangents, [LAYER], dim_batch=2)[LAYER]
    rel = float((got - expected).norm() / expected.norm())
    assert rel < 2e-2, f"column-wise estimator differs from the reference by {rel:.4f} relative"
    cos = torch.nn.functional.cosine_similarity(got, expected, dim=-1)
    assert float(cos.min()) > 0.999, f"direction mismatch, min cosine {float(cos.min()):.5f}"
