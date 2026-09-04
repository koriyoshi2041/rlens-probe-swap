#!/usr/bin/env python
"""Block 82: the mechanism checks on the third model (Qwen3-4B, dense attention), J-lens only.

Reads the band and eligibility from block 81. For every eligible item:
  (1) best-readout bridge position (block-31 rule) and, at that position over the band,
      the 2-D clamp x1/x4, the donor complement paste x0.25 and the full paste x0.25
      (repaired single-hop template donor);
  (2) transport-head search: for every layer (dense model: all layers have patterns), every
      head, the rise of attention from the final position onto the bridge position under the
      complement paste vs clean (median over items reported offline);
  (3) clean-run stage necessity by sliding windows: mean-ablate the attention outputs at the
      final position over windows of 8 consecutive layers, on the two-hop and the single-hop
      prompt (block-61b protocol), to locate the transport stage without assuming layer indices.
Predictions: (1) clamp < complement paste, paste rescues items the clamp does not; (2) a small set
of late-middle heads shows the largest rise; (3) exactly one window band hurts two-hop and not
single-hop.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last
from q44_stage_necessity import sublayer_hook_name
from q65_selfconsistent_scale import mean_ablate
from q66_energy_of_arms import scaled_paste_hooks
from rlens.interventions import clamp_hooks
from q81_third_model import LENS_PATH, load_third
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant, resolve_next_token

TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
WINDOW = 8


def clamp_band(model, bases, coords, band, positions, scale):
    """2-D clamp over this model's band (q70's helper is bound to the 9B band L8-20)."""
    hooks = []
    for l in band:
        c = coords[l][:, positions, :]
        target = c + scale * (c[..., [1, 0]] - c)
        hooks += clamp_hooks(model, bases[l], [l], {l: target}, exchange=False, positions=list(positions))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block82_qwen3_4b_mechanism")
    model = load_third()
    from transformer_lens.tools.analysis import JacobianLens
    lens = JacobianLens.load(str(LENS_PATH)) if LENS_PATH.exists() else torch.load(str(LENS_PATH.with_suffix(".pkl")), weights_only=False)
    meta = json.loads((RESULTS_DIR / "block81_qwen3_4b" / "meta.json").read_text(encoding="utf-8"))
    band = meta["band"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block81_qwen3_4b" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    n_layers = model.cfg.n_layers
    pattern_layers = [l for l in range(n_layers) if f"blocks.{l}.attn.hook_pattern" in model.hook_dict]
    names = {layer: _resid_post_hook_name(layer) for layer in band}
    wanted = set(names.values())
    pat_names = {f"blocks.{l}.attn.hook_pattern" for l in pattern_layers}
    attn_out = {l: sublayer_hook_name(model, l, "attn") for l in range(n_layers)}
    windows = [list(range(s, s + WINDOW)) for s in range(1, n_layers - WINDOW + 1, 2)]
    handle = (out_dir / "mechanism.jsonl").open("w", encoding="utf-8")
    print(f"[q82] band={band} eligible={len(eligible)} pattern layers={len(pattern_layers)} windows={len(windows)}", flush=True)
    for n, index in enumerate(eligible):
        if str(index) not in templates:
            continue
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        T = int(tokens.shape[1])
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        ranks, _ = rank_readout(model, lens, tokens, ids, band)
        hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
        p = int(max(range(len(hits)), key=lambda q: (hits[q], q)))
        # donor
        donor_text = templates[str(index)].format(r.item.swap_to)
        dt = model.to_tokens(donor_text, prepend_bos=False)
        dseq = dt[0].tolist()
        dpos = None
        for variant in (" " + r.item.swap_to, r.item.swap_to):
            dpos = find_last(dseq, list(encode_variant(model.tokenizer, variant).ids))
            if dpos is not None:
                break
        if dpos is None:
            continue
        _, dcache = model.run_with_cache(dt, names_filter=lambda x: x in wanted)
        donor = {layer: dcache[names[layer]][0, dpos] for layer in band}
        with model.hooks(fwd_hooks=[]):
            logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted or x in pat_names)
        clean = torch.log_softmax(logits[0, -1].float(), dim=-1)
        base = float(clean[s] - clean[a])
        bases = {l: lens.lens_vectors(model, ids, l) for l in band}
        coords = {l: cache[names[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in band}
        clean_pat = {l: cache[f"blocks.{l}.attn.hook_pattern"][0, :, -1, p].float().tolist() for l in pattern_layers}
        arms = {"clamp@1": clamp_band(model, bases, coords, band, [p], 1.0), "clamp@4": clamp_band(model, bases, coords, band, [p], 4.0),
                "clamp@1_all": clamp_band(model, bases, coords, band, list(range(T)), 1.0), "clamp@4_all": clamp_band(model, bases, coords, band, list(range(T)), 4.0),
                "orth@0.25": scaled_paste_hooks(model, donor, band, p, 0.25, bases, "orth"), "full@0.25": scaled_paste_hooks(model, donor, band, p, 0.25, None, "full")}
        rec = {"index": index, "name": r.item.name, "n_prompt": T, "position": p, "hits": int(hits[p]), "clean_pattern_to_p": clean_pat, "arms": {}}
        for arm, hooks in arms.items():
            with model.hooks(fwd_hooks=hooks):
                lg, c2 = model.run_with_cache(tokens, names_filter=lambda x: x in pat_names)
            lp = torch.log_softmax(lg[0, -1].float(), dim=-1)
            top1 = int(lp.argmax().item())
            rec["arms"][arm] = {"delta_margin": float((lp[s] - lp[a]) - base), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, lp)}
            if arm == "orth@0.25":
                rec["orth_pattern_to_p"] = {l: c2[f"blocks.{l}.attn.hook_pattern"][0, :, -1, p].float().tolist() for l in pattern_layers}
        # (3) sliding-window stage necessity in the clean run
        prompts = {"two_hop": (r.prompt, a)}
        sh = templates[str(index)].format(r.item.intermediate)
        prompts["single_hop"] = (sh, resolve_next_token(model.tokenizer, sh, r.item.answer).first_id)
        rec["stage"] = {}
        for kind, (text, ans) in prompts.items():
            t = model.to_tokens(text, prepend_bos=False)
            lp0 = final_logprobs(model, t)
            rows = []
            for w in windows:
                lp = final_logprobs(model, t, mean_ablate([attn_out[l] for l in w], t.shape[1] - 1))
                rows.append({"start": w[0], "still_correct": int(lp.argmax().item()) == ans, "delta_logp_answer": float(lp[ans] - lp0[ans])})
            rec["stage"][kind] = {"clean_correct": int(lp0.argmax().item()) == ans, "windows": rows}
        handle.write(json.dumps(rec) + "\n")
        if n % 10 == 0:
            print(f"[q82] {n + 1}/{len(eligible)} p={p} " + " ".join(f"{k}={v['delta_margin']:+.2f}" for k, v in rec["arms"].items()), flush=True)
    handle.close()
    print("[block82] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
