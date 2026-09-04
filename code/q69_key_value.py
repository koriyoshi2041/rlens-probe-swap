#!/usr/bin/env python
"""Block 66: does the off-plane component carry the KEY-side "attend to me" signal?

Block 63: the 27 never-flipped items the 2-D clamp cannot rescue are the items where the
transport head L23 h8 barely attends to the bridge position in the clean run (0.10 vs
0.40 for flippable items). The clamp leaves that attention unchanged; the complement paste
(orth@0.25) raises it to 0.65-0.82 for every group and the rise tracks rescue.
Hypothesis: the complement carries the key-side signal that makes L19/L23 attend to the
bridge position; the lens plane carries (some of) the value-side content but cannot
redirect attention. Decisive test: force the attention (keys or pattern) from the
complement-paste run while applying only the 2-D clamp at L8-20.
Arms (patches at the bridge position p, or the final position for queries; source run S):
  from S = orth@0.25 (complement paste L8-20 at p):
    K[p]@19,23   V[p]@19,23   KV[p]@19,23   Q[-1]@19,23   KV[p]@23   KV[p]@19
    PAT@19,23 (final-row attention pattern forced)   PAT_h8@23
    K[p]@19,23 + clamp2d@4      PAT@19,23 + clamp2d@4      KV[p]@19,23 + clamp2d@4
  from S = clamp2d@4:  KV[p]@19,23   (control)
Self-check on the first item: patching clean tensors changes nothing; zeroing the final
pattern row equals zeroing z at the final position (pattern hook is write-effective).
Predictions. Key hypothesis: K[p] alone raises attention but not the margin (clean value
is the source entity); K[p] + clamp2d@4 rescues items the clamp alone never rescues, and
PAT + clamp2d@4 likewise. If instead only KV works, the plane does not carry the value
content the transport heads copy either.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q30_4b_replication import load_4b
from q35_donor_paste import find_last
from q66_energy_of_arms import scaled_paste_hooks
from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
FOURB = os.environ.get("MATS_MODEL", "9b") == "4b"  # block70: same test on Qwen3.5-4B (its transport head is also L23 h8, block67)
ATTN_LAYERS = (19, 23)
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
KINDS = ("hook_q", "hook_k", "hook_v", "hook_pattern", "hook_z")


def hook_name(layer, kind):
    return f"blocks.{layer}.attn.{kind}"


def patch_position(cached, position):
    """Replace tensor[:, position] (q/k/v/z layouts: [batch, pos, heads, d]) with the cached run's."""
    def fn(tensor, hook):
        out = tensor.clone()
        out[:, position] = cached[:, position].to(out.dtype)
        return out
    return fn


def patch_pattern(cached, head=None):
    """Replace the final query row of the attention pattern ([batch, heads, q, k]) with the cached run's."""
    def fn(tensor, hook):
        out = tensor.clone()
        if head is None:
            out[:, :, -1, :] = cached[:, :, -1, :].to(out.dtype)
        else:
            out[:, head, -1, :] = cached[:, head, -1, :].to(out.dtype)
        return out
    return fn


def zero_pattern_row(tensor, hook):
    out = tensor.clone()
    out[..., -1, :] = 0
    return out


def zero_last_z(tensor, hook):
    out = tensor.clone()
    out[:, -1] = 0
    return out


def run(model, tokens, hooks, position, s, a):
    wanted = {hook_name(l, "hook_pattern") for l in ATTN_LAYERS}
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    out = {"margin": float(lp[s] - lp[a]), "top1_is_swap": int(lp.argmax().item()) == s, "top1_is_answer": int(lp.argmax().item()) == a}
    for l in ATTN_LAYERS:
        pat = cache[hook_name(l, "hook_pattern")][0, :, -1, :].float()
        out[f"L{l}_sum_to_bridge"] = float(pat[:, position].sum().item())
        if l == 23:
            out["L23_h8_to_bridge"] = float(pat[8, position].item())
    return out


def capture(model, tokens, hooks):
    wanted = {hook_name(l, k) for l in ATTN_LAYERS for k in KINDS}
    with model.hooks(fwd_hooks=hooks):
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {name: cache[name].detach().clone() for name in wanted}


def self_check(model, tokens, position, s, a, clean_cache):
    clean = run(model, tokens, [], position, s, a)["margin"]
    same_k = run(model, tokens, [(hook_name(l, "hook_k"), patch_position(clean_cache[hook_name(l, "hook_k")], position)) for l in ATTN_LAYERS], position, s, a)["margin"]
    same_pat = run(model, tokens, [(hook_name(l, "hook_pattern"), patch_pattern(clean_cache[hook_name(l, "hook_pattern")])) for l in ATTN_LAYERS], position, s, a)["margin"]
    pat0 = run(model, tokens, [(hook_name(23, "hook_pattern"), zero_pattern_row)], position, s, a)["margin"]
    z0 = run(model, tokens, [(hook_name(23, "hook_z"), zero_last_z)], position, s, a)["margin"]
    report = {"clean": clean, "patch_clean_k": same_k, "patch_clean_pattern": same_pat, "zero_pattern_row_L23": pat0, "zero_z_last_L23": z0}
    report["pattern_write_effective"] = bool(abs(pat0 - z0) < 0.05 and abs(pat0 - clean) > 1e-3)
    report["identity_ok"] = bool(abs(same_k - clean) < 1e-2 and abs(same_pat - clean) < 1e-2)
    return report


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block70_4b_key_value" if FOURB else "block66_key_value")
    if FOURB:
        model, lenses = load_4b()
        lens = lenses["J"]
    else:
        model = load_model()
        lens = load_lenses()["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / ("block27_4b" if FOURB else "block01") / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / ("block35_4b_donor_paste" if FOURB else "block31_donor_paste") / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "key_value.jsonl").open("w", encoding="utf-8")
    checked = None
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        if index not in best_pos or str(index) not in templates:
            continue
        position = best_pos[index]
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
        donor = {layer: dcache[names[layer]][0, dpos] for layer in BAND}
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        bases = {l: lens.lens_vectors(model, ids, l) for l in BAND}
        coords = {l: cache[names[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in BAND}
        clamp = []
        for l in BAND:
            c = coords[l][:, position:position + 1, :]
            target = c + 4.0 * (c[..., [1, 0]] - c)
            clamp += clamp_hooks(model, bases[l], [l], {l: target}, exchange=False, positions=[position])
        orth = scaled_paste_hooks(model, donor, BAND, position, 0.25, bases, "orth")
        src = {"orth@0.25": capture(model, tokens, orth), "clamp2d@4": capture(model, tokens, clamp)}
        if checked is None:
            checked = self_check(model, tokens, position, s, a, capture(model, tokens, []))
            (out_dir / "self_check.json").write_text(json.dumps({"index": index, **checked}, indent=1), encoding="utf-8")
            print(f"[q69] self-check {checked}", flush=True)
        S = src["orth@0.25"]
        arms = {
            "clean": [], "clamp2d@4": clamp, "orth@0.25": orth,
            "K@19,23": [(hook_name(l, "hook_k"), patch_position(S[hook_name(l, "hook_k")], position)) for l in ATTN_LAYERS],
            "V@19,23": [(hook_name(l, "hook_v"), patch_position(S[hook_name(l, "hook_v")], position)) for l in ATTN_LAYERS],
            "KV@19,23": [(hook_name(l, k), patch_position(S[hook_name(l, k)], position)) for l in ATTN_LAYERS for k in ("hook_k", "hook_v")],
            "Q@19,23": [(hook_name(l, "hook_q"), patch_position(S[hook_name(l, "hook_q")], tokens.shape[1] - 1)) for l in ATTN_LAYERS],
            "KV@23": [(hook_name(23, k), patch_position(S[hook_name(23, k)], position)) for k in ("hook_k", "hook_v")],
            "KV@19": [(hook_name(19, k), patch_position(S[hook_name(19, k)], position)) for k in ("hook_k", "hook_v")],
            "PAT@19,23": [(hook_name(l, "hook_pattern"), patch_pattern(S[hook_name(l, "hook_pattern")])) for l in ATTN_LAYERS],
            "PAT_h8@23": [(hook_name(23, "hook_pattern"), patch_pattern(S[hook_name(23, "hook_pattern")], head=8))],
            "Z@19,23": [(hook_name(l, "hook_z"), patch_position(S[hook_name(l, "hook_z")], tokens.shape[1] - 1)) for l in ATTN_LAYERS],
        }
        arms["K@19,23+clamp2d@4"] = arms["K@19,23"] + clamp
        arms["PAT@19,23+clamp2d@4"] = arms["PAT@19,23"] + clamp
        arms["KV@19,23+clamp2d@4"] = arms["KV@19,23"] + clamp
        arms["V@19,23+clamp2d@4"] = arms["V@19,23"] + clamp
        C = src["clamp2d@4"]
        arms["KV@19,23<-clamp"] = [(hook_name(l, k), patch_position(C[hook_name(l, k)], position)) for l in ATTN_LAYERS for k in ("hook_k", "hook_v")]
        arms["Z@19,23<-clamp"] = [(hook_name(l, "hook_z"), patch_position(C[hook_name(l, "hook_z")], tokens.shape[1] - 1)) for l in ATTN_LAYERS]
        results = {arm: run(model, tokens, hooks, position, s, a) for arm, hooks in arms.items()}
        for arm in results:
            results[arm]["delta_margin"] = results[arm]["margin"] - results["clean"]["margin"]
        handle.write(json.dumps({"index": index, "name": r.item.name, "position": position, "n_prompt": int(tokens.shape[1]), "arms": results}) + "\n")
        if n % 10 == 0:
            print(f"[q69] {n + 1}/{len(eligible)} " + " ".join(f"{k}={v['delta_margin']:+.2f}" for k, v in results.items() if k in ("clamp2d@4", "orth@0.25", "K@19,23", "KV@19,23", "K@19,23+clamp2d@4", "PAT@19,23+clamp2d@4")), flush=True)
    handle.close()
    print("[block70] done" if FOURB else "[block66] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
