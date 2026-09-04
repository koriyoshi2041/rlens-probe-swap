#!/usr/bin/env python
"""Block 73: the transport layers L17-24 seen from the bridge position: how much of the paste
effect flows through their inputs at p, and does the plane's content suffice once they are fed?

Block 66 patched keys only at L19/L23. The linear-attention layers 17, 18, 20, 21, 22, 24 use a
fused kernel: their internal k/v/beta/decay hooks do not fire (probe 22:40), only
linear_attn.hook_in / hook_out do. So key vs value cannot be separated there; what can be done:
  lin_in_all      linear_attn.hook_in[p] at 17,18,20,21,22,24 from the complement-paste run S
  K19,23          attn.hook_k[p] from S (routing only, full-attention layers)
  KV19,23         attn.hook_k[p] + hook_v[p] from S
  K19,23+lin_in   full-attention routing from S + linear layers fed S's input at p
  KV19,23+lin_in  every transport layer sees S at p  (upper bound of the "at-p route")
  each of the above + clamp2d@4 (L8-20 at p)
  orth-restore    the complement paste with all transport-layer inputs at p restored to clean
                  (attn.hook_in[p] at 19,23 and linear_attn.hook_in[p] at the others): the paste
                  effect that does NOT flow through the transport layers' inputs at p
  KV19,23<-clamp + lin_in<-clamp   the clamp seen by the transport layers only
  PAT_h8@23+clamp (audit M-4)
Self-check on the first item: patching the clean run's own tensors changes nothing.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last
from q66_energy_of_arms import scaled_paste_hooks
from q69_key_value import patch_pattern, patch_position
from rlens.data import load_items
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TRANSPORT = list(range(17, 25))
FULL = (19, 23)
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


LINEAR = [l for l in TRANSPORT if l not in FULL]
K_FULL = [f"blocks.{l}.attn.hook_k" for l in FULL]
V_FULL = [f"blocks.{l}.attn.hook_v" for l in FULL]
IN_FULL = [f"blocks.{l}.attn.hook_in" for l in FULL]
IN_LIN = [f"blocks.{l}.linear_attn.hook_in" for l in LINEAR]
ALL_NAMES = K_FULL + V_FULL + IN_FULL + IN_LIN


def capture(model, tokens, hooks):
    wanted = set(ALL_NAMES) | {"blocks.23.attn.hook_pattern"}
    with model.hooks(fwd_hooks=hooks):
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {name: cache[name].detach().clone() for name in wanted if name in cache}


def patches(source, names, position):
    return [(n, patch_position(source[n], position)) for n in names if n in source]


def run(model, tokens, hooks, position, s, a):
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x == "blocks.23.attn.hook_pattern")
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    top1 = int(lp.argmax().item())
    pat = cache["blocks.23.attn.hook_pattern"][0, 8, -1, :].float()
    return {"margin": float(lp[s] - lp[a]), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "h8": float(pat[position].item())}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block73_routing_all_layers")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    missing = [n for n in ALL_NAMES if n not in model.hook_dict]
    print(f"[q73] missing hooks: {missing}", flush=True)
    handle = (out_dir / "routing_all_layers.jsonl").open("w", encoding="utf-8")
    checked = False
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
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
        S = capture(model, tokens, orth)
        Cc = capture(model, tokens, clamp)
        clean_cache = capture(model, tokens, [])
        if not checked:
            shapes = {k: tuple(v.shape) for k, v in clean_cache.items() if "hook_pattern" not in k}
            base = run(model, tokens, [], position, s, a)["margin"]
            fired = [n for n in ALL_NAMES if n in clean_cache]
            print(f"[q73] fired hooks: {fired}", flush=True)
            same = run(model, tokens, patches(clean_cache, ALL_NAMES, position), position, s, a)["margin"]
            check = {"index": index, "shapes": shapes, "clean_margin": base, "patch_clean_all_margin": same, "identity_ok": bool(abs(same - base) < 1e-2)}
            (out_dir / "self_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
            print(f"[q73] self-check identity_ok={check['identity_ok']} shapes={shapes}", flush=True)
            checked = True
        arms = {
            "clean": [], "clamp2d@4": clamp, "orth@0.25": orth,
            "K19,23": patches(S, K_FULL, position), "KV19,23": patches(S, K_FULL + V_FULL, position),
            "lin_in": patches(S, IN_LIN, position),
            "K19,23+lin_in": patches(S, K_FULL + IN_LIN, position),
            "KV19,23+lin_in": patches(S, K_FULL + V_FULL + IN_LIN, position),
            "transport<-clamp": patches(Cc, K_FULL + V_FULL + IN_LIN, position),
            "orth-restore": orth + patches(clean_cache, IN_FULL + IN_LIN, position),
        }
        arms["PAT_h8@23+clamp"] = [("blocks.23.attn.hook_pattern", patch_pattern(S["blocks.23.attn.hook_pattern"], head=8))] + clamp
        for base_arm in ("K19,23", "KV19,23", "lin_in", "K19,23+lin_in", "KV19,23+lin_in"):
            arms[base_arm + "+clamp"] = arms[base_arm] + clamp
        results = {arm: run(model, tokens, hooks, position, s, a) for arm, hooks in arms.items()}
        for arm in results:
            results[arm]["delta_margin"] = results[arm]["margin"] - results["clean"]["margin"]
        handle.write(json.dumps({"index": index, "name": r.item.name, "position": position, "arms": results}) + "\n")
        if n % 10 == 0:
            print(f"[q73] {n + 1}/{len(eligible)} " + " ".join(f"{k}={v['delta_margin']:+.2f}" for k, v in results.items() if k in ("clamp2d@4", "orth@0.25", "KV19,23+lin_in", "K19,23+lin_in+clamp", "orth-restore")), flush=True)
    handle.close()
    print("[block73] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
