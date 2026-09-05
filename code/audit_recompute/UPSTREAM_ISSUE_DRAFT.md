# Upstream issue for TransformerLens `tools.analysis.jacobian_lens`

Filed as [TransformerLens issue #1746](https://github.com/TransformerLensOrg/TransformerLens/issues/1746), with [PR #1747](https://github.com/TransformerLensOrg/TransformerLens/pull/1747). The text below is the draft the issue was written from.

**Title:** `JacobianLens.swap_hooks` re-reads the lens coordinates from the already-patched activation at each layer, so applying it across an even number of band layers cancels itself (the paper's protocol clamps to clean-pass coordinates)

**Summary.** The Jacobian-lens "swap" intervention `h ← h + α V (σ(c) − c)` (exchange the two lens
coordinates and add the difference back) is an involution on the lens plane. When the same swap is
applied at consecutive layers of a residual stream whose in-plane coordinates are approximately
preserved from layer to layer, the second application undoes the first, the third redoes it, and so on.
The net effect therefore alternates with the number of layers in the band: odd widths flip the
readout, even widths approximately restore the original. We observed this parity signature on two
models (Qwen3.5-9B and Qwen3.5-4B: the flip rate alternates high/low with band width 1…13; exact numbers in
the project log block 4 / figure F1) and it
disappears when the swap is replaced by an idempotent *clamp* (set the coordinates to the exchanged
target instead of adding the exchange difference), which never re-exchanges an already-exchanged state.

**Minimal reproduction (any model with a fitted lens; band = 2 consecutive layers):**
```python
import torch
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import JacobianLens
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

model = TransformerBridge.boot_transformers(MODEL_DIR, device="cuda", dtype=torch.bfloat16); model.eval()
lens = JacobianLens.load(LENS_PATH)
tokens = model.to_tokens(PROMPT, prepend_bos=False)          # a two-hop prompt whose bridge entity the lens reads
ids = [SRC_TOKEN_ID, TGT_TOKEN_ID]                              # e.g. " France" and " Italy"

def swap_hook(layer):
    basis = lens.lens_vectors(model, ids, layer)                # [2, d_model]
    pinv = torch.linalg.pinv(basis.T.float())
    def transform(h):
        h = h.float(); c = h @ pinv.T                           # coordinates
        return h + (c[..., [1, 0]] - c) @ basis.float()         # exchange and add back  (involution)
    return (_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model))

def margin(hooks):
    with model.hooks(fwd_hooks=hooks):
        lp = torch.log_softmax(model(tokens)[0, -1].float(), -1)
    return float(lp[TGT_ANSWER_ID] - lp[SRC_ANSWER_ID])

print("clean       ", margin([]))
print("swap @L      ", margin([swap_hook(L)]))                  # large positive change
print("swap @L,L+1  ", margin([swap_hook(L), swap_hook(L + 1)]))  # ≈ clean again (cancellation)
print("swap @L..L+2 ", margin([swap_hook(l) for l in (L, L + 1, L + 2)]))  # positive again
```
Expected: the two-layer band gives ≈ the clean margin while one- and three-layer bands give the flipped
margin. The docstring says "the paper clamps the swap across an intermediate-layer band" and the paper's own control uses coordinates "clamped to their clean-pass values"; the released hook instead recomputes `c = V⁺h` from the live (already-modified) activation at every hooked layer, which turns the exchange into an involution across the band. A fix that removes the parity dependence: make the intervention idempotent,
`h ← h + V (c_target − c)` with `c_target` fixed to the exchange of the *clean* coordinates (a clamp),
or apply the exchange at a single layer only. Suggested doc note: multi-layer application of
`coordinate_map(mode="swap")` should use the clamp form or an odd band width.

**Environment.** transformer_lens version as pinned in the project's environment (`activate.sh`), models Qwen3.5-9B / Qwen3.5-4B via `TransformerBridge`, lenses = published workspace lenses.

**Evidence in this project.** Research log blocks 1–4, 27 (4B), 81 (Qwen3-4B dense, running); figures F1 (parity ladder) and F3 (baseline ladder).
