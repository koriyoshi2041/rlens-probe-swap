#!/usr/bin/env python
"""Block 26: an unconfounded entity-report probe for the answer/entity dissociation.

Why this exists. Block 23's probe appended the item's *clean* answer before asking
which entity the fact was about ("... is Paris. The country referred to above is").
Under the clamp the model would have said "Rome", but "Paris" sat in the probe
context (correctly left unclamped), so "answer changed, entity report unchanged"
is exactly what a clean "Paris" in context would produce regardless of internal
state. Its readable-name yield was also only about a third of items.

This block separates the two sources of an entity report:
  * q_noans    : prompt + question, NO answer in context -> report driven by the
                 (clamped) prompt-span state only;
  * q_cleanans : prompt + clean answer + question (block 23's confounded design,
                 kept as a comparison arm);
  * q_swapans  : prompt + the *target* answer + question, in the CLEAN run this arm
                 tells us how much a stated answer alone drags the entity report;
  * ref_noans  : the block-23 phrasing without an answer.
Category-specific nouns are used, the probe suffix is tokenised separately and
concatenated at the token level (so the prompt-span clamp targets align exactly,
asserted), and at the probe's final position log p(swap_to) − log p(intermediate)
is recorded directly.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
GEN_TOKENS = 10

NOUN_BY_PREFIX = [
    ("ex-city-", "country"), ("ex2-city-", "country"), ("ex2-language-", "country"), ("ex2-river-", "country"),
    ("amazon-", "country"), ("greatwall-", "country"), ("paper-", "country"), ("super-populous", "country"),
    ("super-smallest", "country"), ("spaceneedle-", "US state"), ("city-state-", "city"),
    ("animal-", "animal"), ("spider-", "animal"), ("food-animal", "food"), ("fruit-", "fruit"),
    ("chem-", "element"), ("ex-element-", "element"), ("element-", "element"),
    ("ex-planet-", "planet"), ("planet-", "planet"),
    ("person-", "person"), ("birthstone-", "month"), ("etym-", "day of the week"),
    ("christmas-", "holiday"), ("holiday-", "holiday"), ("month-3", "Roman god"),
    ("func-", "organ"), ("organ-", "organ"), ("gem-", "gemstone"), ("season-", "season"),
    ("vehicle-", "vehicle"), ("violin-", "instrument"),
]


def noun_for(name: str) -> str:
    for prefix, noun in NOUN_BY_PREFIX:
        if name.startswith(prefix):
            return noun
    return "thing"


def probe_suffixes(noun: str, clean_answer: str, swap_answer: str):
    q = f"\nQuestion: Which {noun} is this fact about?\nAnswer: The {noun} is"
    return {
        "q_noans": q,
        "q_cleanans": f"{clean_answer}." + q,
        "q_swapans": f"{swap_answer}." + q,
        "ref_noans": f"\nThe {noun} referred to in the fact above is",
    }


@torch.inference_mode()
def generate(model, seq, hooks, n_tokens):
    start = seq.shape[1]
    for _ in range(n_tokens):
        logp = final_logprobs(model, seq, hooks)
        seq = torch.cat([seq, logp.argmax().view(1, 1)], dim=1)
    return model.tokenizer.decode(seq[0, start:].tolist())


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block26_entity_probe")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "entity_probe.jsonl").open("w", encoding="utf-8")
    lines = []
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        n_prompt = tokens.shape[1]
        ids = r.tracked_ids[:2]
        a, s = r.answer.first_id, r.swap_answer.first_id
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        noun = noun_for(r.item.name)
        clean_main = final_logprobs(model, tokens)
        # concept tokens as they would appear after "... is" (leading space form)
        src_tok = encode_variant(model.tokenizer, " " + r.item.intermediate)
        tgt_tok = encode_variant(model.tokenizer, " " + r.item.swap_to)
        lines.append(f"### {r.item.name}  [{r.item.intermediate} -> {r.item.swap_to}]  expect {r.answer.text.strip()} -> {r.swap_answer.text.strip()}  noun={noun}")
        suffixes = probe_suffixes(noun, r.answer.text, r.swap_answer.text)
        for kind in ("clean", "J", "R"):
            hooks = []
            if kind != "clean":
                for layer in BAND:
                    basis = lenses[kind].lens_vectors(model, ids, layer)
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords = cache[names[layer]].float() @ pinv.T
                    hooks += clamp_hooks(model, basis, [layer], {layer: coords}, positions=list(range(n_prompt)))
            main_logp = final_logprobs(model, tokens, hooks)
            answer_margin = float(main_logp[s] - main_logp[a])
            for probe_name, suffix in suffixes.items():
                suffix_ids = torch.tensor([model.tokenizer.encode(suffix, add_special_tokens=False)], device=tokens.device)
                probe_tokens = torch.cat([tokens, suffix_ids], dim=1)
                assert torch.equal(probe_tokens[0, :n_prompt], tokens[0])
                logp = final_logprobs(model, probe_tokens, hooks)
                top1 = int(logp.argmax().item())
                text = generate(model, probe_tokens, hooks, GEN_TOKENS)
                row = {
                    "index": index, "name": r.item.name, "category": r.item.category, "noun": noun,
                    "arm": kind, "probe": probe_name,
                    "answer_margin": answer_margin,
                    "answer_margin_clean": float(clean_main[s] - clean_main[a]),
                    "top1_is_swap_answer": int(main_logp.argmax().item()) == s,
                    "entity_logp_intermediate": float(logp[src_tok.first_id]),
                    "entity_logp_swap_to": float(logp[tgt_tok.first_id]),
                    "entity_margin": float(logp[tgt_tok.first_id] - logp[src_tok.first_id]),
                    "entity_top1": model.tokenizer.decode([top1]),
                    "entity_top1_is_intermediate": top1 == src_tok.first_id,
                    "entity_top1_is_swap_to": top1 == tgt_tok.first_id,
                    "entity_tokens_single": src_tok.single and tgt_tok.single,
                    "entity_text": text,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                lines.append(f"    {kind:5s} {probe_name:10s} top1={row['entity_top1']!r:14s} margin={row['entity_margin']:+6.2f}  text={text!r}")
        lines.append("")
        print(f"[probe] {n + 1}/{len(eligible)} {r.item.name}", flush=True)
    handle.close()
    (out_dir / "entity_probe_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[block26] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
