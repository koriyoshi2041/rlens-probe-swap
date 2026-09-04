#!/usr/bin/env python
"""Block 77: (a) is the residual-PCA subspace also NECESSARY, and (b) which sublayers turn the lens
components into the rest of the donor difference?

(a) Block 72: pasting through residual-PCA top-256 carries the content as well as the lens
subspace. Necessity ladder as in block 71 (full paste x0.25 minus a subspace, best position):
full, full-PCA64, full-PCA256, full-PCA1024, full-R256, full-(R256 u PCA256), full-rand256.
(b) Block 72: only the lens subspace's realised share GROWS after a single L8 paste. Here, with
the L8-only paste through J+R256 (and J+PCA256, full, J+rand256 as comparisons), the change of
every sublayer output at the bridge position (attention or linear-attention, MLP; L9-L20)
between the pasted and the clean run is projected onto the donor difference: onto d_l (the
layer's own difference) and onto d_20 (the end-of-band difference). This locates which layers
and which sublayer type manufacture the missing components.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q44_stage_necessity import sublayer_hook_name
from q51_subspace_ladder import orthonormal
from q66_energy_of_arms import energy_wrapped
from q68_propagation import recording_projector_hooks
from q71_necessity_ladder import removal_paste_hooks
from q72_subspace_controls import donor_at
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
SCALE = 0.25


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block77_growth_pca")
    model = load_model()
    lenses = load_lenses()
    lens = lenses["J"]
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
    sub_names = {l: (sublayer_hook_name(model, l, "attn"), sublayer_hook_name(model, l, "mlp")) for l in BAND}
    wanted = set(names.values()) | {n for pair in sub_names.values() for n in pair}
    dev = model.W_U.device
    svd_R = {}
    for layer in BAND:
        j = lenses["R"].jacobians[layer].to(device=dev, dtype=torch.float32)
        _, _, vh = torch.linalg.svd(j, full_matrices=False)
        svd_R[layer] = vh[:256]
    resid = {layer: [] for layer in BAND}
    info = {}
    for index in eligible:
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for layer in BAND:
            resid[layer].append(cache[names[layer]][0].float())
        donor = donor_at(model, templates[str(index)].format(r.item.swap_to), r.item.swap_to, set(names.values()), names)
        if donor is None:
            continue
        p = best_pos[index]
        info[index] = {"tokens": tokens, "r": r, "p": p, "donor": donor,
                       "clean": {l: cache[names[l]][0, p].float() for l in BAND},
                       "sub": {l: (cache[sub_names[l][0]][0, p].float(), cache[sub_names[l][1]][0, p].float()) for l in BAND}}
    pca = {}
    for layer in BAND:
        x = torch.cat(resid[layer])
        x = x - x.mean(0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        pca[layer] = vh
    print(f"[q77] PCA ready; items {len(info)}", flush=True)
    handle = (out_dir / "growth_pca.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(sorted(info)):
        it = info[index]
        r, tokens, p, donor = it["r"], it["tokens"], it["p"], it["donor"]
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        bases = {l: lens.lens_vectors(model, ids, l).float() for l in BAND}
        gen = torch.Generator(device="cpu").manual_seed(11000 + index)
        d = {l: donor[l] - it["clean"][l] for l in BAND}
        rec = {"index": index, "name": r.item.name, "position": p, "necessity": {}, "growth": {}}
        # (a) necessity ladder with PCA removal
        removal = {"full": None}
        for k in (64, 256, 1024):
            removal[f"full-PCA{k}"] = {l: orthonormal(torch.cat([bases[l], pca[l][:k]])) for l in BAND}
        removal["full-R256"] = {l: orthonormal(torch.cat([bases[l], svd_R[l]])) for l in BAND}
        removal["full-(R256+PCA256)"] = {l: orthonormal(torch.cat([bases[l], svd_R[l], pca[l][:256]])) for l in BAND}
        removal["full-rand256"] = {l: orthonormal(torch.cat([bases[l], torch.randn(256, model.cfg.d_model, generator=gen).to(dev)])) for l in BAND}
        for arm, rem in removal.items():
            store = {}
            logp = final_logprobs(model, tokens, energy_wrapped(removal_paste_hooks(model, donor, BAND, p, SCALE, rem), p, store))
            rec["necessity"][arm] = {"delta_margin": float((logp[s] - logp[a]) - base), "top1_is_swap": int(logp.argmax().item()) == s,
                                     "kl": kl_divergence(clean, logp), "energy": store.get("energy", 0.0)}
        # (b) growth localisation after an L8-only paste
        subspaces = {"J+R256": {l: orthonormal(torch.cat([bases[l], svd_R[l]])) for l in BAND},
                     "J+PCA256": {l: orthonormal(torch.cat([bases[l], pca[l][:256]])) for l in BAND},
                     "J+rand256": {l: orthonormal(torch.cat([bases[l], torch.randn(256, model.cfg.d_model, generator=gen).to(dev)])) for l in BAND},
                     "full": None}
        d20 = d[BAND[-1]]
        for sub_name, sub in subspaces.items():
            hooks = recording_projector_hooks(model, donor, [BAND[0]], p, sub, {})
            with model.hooks(fwd_hooks=hooks):
                logits, icache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            layers = []
            for l in BAND[1:]:
                attn_d = icache[sub_names[l][0]][0, p].float() - it["sub"][l][0]
                mlp_d = icache[sub_names[l][1]][0, p].float() - it["sub"][l][1]
                resid_d = icache[names[l]][0, p].float() - it["clean"][l]
                layers.append({"layer": l,
                               "attn_on_dl": float((attn_d @ d[l]).item() / max(float((d[l] ** 2).sum().item()), 1e-9)),
                               "mlp_on_dl": float((mlp_d @ d[l]).item() / max(float((d[l] ** 2).sum().item()), 1e-9)),
                               "attn_on_d20": float((attn_d @ d20).item() / max(float((d20 ** 2).sum().item()), 1e-9)),
                               "mlp_on_d20": float((mlp_d @ d20).item() / max(float((d20 ** 2).sum().item()), 1e-9)),
                               "realised_dl": float((resid_d @ d[l]).item() / max(float((d[l] ** 2).sum().item()), 1e-9)),
                               "realised_d20": float((resid_d @ d20).item() / max(float((d20 ** 2).sum().item()), 1e-9)),
                               "attn_energy": float((attn_d ** 2).sum().item()), "mlp_energy": float((mlp_d ** 2).sum().item())})
            l8 = icache[names[BAND[0]]][0, p].float() - it["clean"][BAND[0]]
            rec["growth"][sub_name] = {"delta_margin": float((lp[s] - lp[a]) - base), "top1_is_swap": int(lp.argmax().item()) == s,
                                       "realised_L8_on_d8": float((l8 @ d[BAND[0]]).item() / max(float((d[BAND[0]] ** 2).sum().item()), 1e-9)),
                                       "realised_L8_on_d20": float((l8 @ d20).item() / max(float((d20 ** 2).sum().item()), 1e-9)), "layers": layers}
        handle.write(json.dumps(rec) + "\n")
        if n % 10 == 0:
            print(f"[q77] {n + 1}/{len(info)} " + " ".join(f"{k}:{v['top1_is_swap']:d}" for k, v in rec["necessity"].items()), flush=True)
    handle.close()
    print("[block77] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
