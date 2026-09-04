#!/usr/bin/env python
"""Block 44: what does the clamped state look like when the answer is DAMAGED (neither old nor new)?

Block 23 found 5/59 items whose answer becomes a third word under the clamp (China->France
continent: Asia -> Africa; ruby->pearl colour: red -> green). Hypothesis: the clamped
bridge-position state reads as a mixture / a third entity, and the second hop is applied
to that. Test: J-lens top-10 readout at the best bridge position, clean vs clamped, for
the damaged items and, as contrast, for 5 cleanly flipped items and 5 unchanged items.
Also the readout at the final position at L24 (after the transfer) to see which entity or
answer arrives there.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name, _unembed

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
READ_LAYERS = [8, 12, 16, 20, 24]
DAMAGED = ["paper-continent", "gem-color-ruby", "city-state-Philadelphia", "person-country-napoleon", "spider-legs"]
FLIPPED = ["ex-city-capital-Lyon-Naples", "ex2-city-capital-Munich", "greatwall-ocean", "amazon-language", "christmas-season"]
UNCHANGED = ["animal-cover-turtle", "person-firstname-einstein", "violin-strings", "ex-element-symbol-26-79", "fruit-grows-grape"]


def topk_strings(model, logits, k=10):
    vals, ids = torch.topk(logits, k)
    return [(model.tokenizer.decode([int(i)]), round(float(v), 2)) for v, i in zip(vals, ids)]


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block44_damage_readout")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    by_name = {items[i].name: i for i in eligible}
    best_pos = {}
    for line in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in sorted(set(BAND) | set(READ_LAYERS))}
    wanted = set(names.values())
    out = []
    lines = []
    for group, name_list in (("damaged", DAMAGED), ("flipped", FLIPPED), ("unchanged", UNCHANGED)):
        for name in name_list:
            if name not in by_name:
                continue
            index = by_name[name]
            r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
            tokens = model.to_tokens(r.prompt, prepend_bos=False)
            ids = [r.src.first_id, r.tgt.first_id]
            bp = best_pos[index]
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            hooks = []
            for layer in BAND:
                basis = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(basis.T.float())
                coords = cache[names[layer]].float() @ pinv.T
                hooks += clamp_hooks(model, basis, [layer], {layer: coords})
            with model.hooks(fwd_hooks=hooks):
                logits, hcache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            rec = {"group": group, "index": index, "name": name, "prompt": r.prompt, "best_pos": bp, "pos_token": model.tokenizer.decode([tokens[0, bp].item()]),
                   "intermediate": r.item.intermediate, "swap_to": r.item.swap_to, "answer": r.answer.text.strip(), "swap_answer": r.swap_answer.text.strip(),
                   "final_top5_clean": topk_strings(model, final_logprobs(model, tokens), 5), "final_top5_clamped": topk_strings(model, torch.log_softmax(logits[0, -1].float(), dim=-1), 5),
                   "readout": {}}
            lines.append(f"### [{group}] {name}  [{r.item.intermediate} -> {r.item.swap_to}]  expect {rec['answer']} -> {rec['swap_answer']}   bridge pos {bp} = {rec['pos_token']!r}")
            lines.append(f"    final clean  : {rec['final_top5_clean']}")
            lines.append(f"    final clamped: {rec['final_top5_clamped']}")
            for layer in READ_LAYERS:
                for tag, c in (("clean", cache), ("clamped", hcache)):
                    for pos_tag, pos in (("bridge", bp), ("final", tokens.shape[1] - 1)):
                        act = c[names[layer]][0, pos].float().unsqueeze(0)
                        lg = _unembed(model, lens.transport(act, layer))[0]
                        rec["readout"][f"L{layer}_{tag}_{pos_tag}"] = topk_strings(model, lg, 8)
                lines.append(f"    L{layer:02d} bridge clean  : {rec['readout'][f'L{layer}_clean_bridge']}")
                lines.append(f"    L{layer:02d} bridge clamped: {rec['readout'][f'L{layer}_clamped_bridge']}")
                lines.append(f"    L{layer:02d} final  clamped: {rec['readout'][f'L{layer}_clamped_final']}")
            lines.append("")
            out.append(rec)
            print(f"[q49] {group} {name}", flush=True)
    (out_dir / "damage_readout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "damage_readout_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[block44] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
