#!/usr/bin/env python
"""Block 01 / step 1: audit the 90 probe-swap items with the tokenizer only (no GPU)."""
from __future__ import annotations

import collections
import json
import sys
from datetime import datetime, timezone

from transformers import AutoTokenizer

from rlens.data import CONCEPT_FIELDS, leaks_in_prompt, load_items, reverse_pairs, sha256_of
from rlens.paths import ITEMS_FILE, MODEL_DIR, results_dir
from rlens.tokens import resolve_concept_token, resolve_next_token


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    items = load_items()
    rows = []
    for item in items:
        src = resolve_concept_token(tokenizer, item.intermediate)
        tgt = resolve_concept_token(tokenizer, item.swap_to)
        stripped = item.prompt.rstrip()
        answer_as_is = resolve_next_token(tokenizer, item.prompt, item.answer)
        swap_as_is = resolve_next_token(tokenizer, item.prompt, item.swap_answer)
        answer_rs = resolve_next_token(tokenizer, stripped, item.answer)
        swap_rs = resolve_next_token(tokenizer, stripped, item.swap_answer)
        rows.append(
            {
                **item.as_dict(),
                "trailing_space": item.prompt != stripped,
                "n_prompt_tokens_as_is": len(tokenizer.encode(item.prompt, add_special_tokens=False)),
                "n_prompt_tokens_rstrip": len(tokenizer.encode(stripped, add_special_tokens=False)),
                "leak": leaks_in_prompt(item),
                "concept_src": src.__dict__,
                "concept_tgt": tgt.__dict__,
                "concept_feasible": src.single and tgt.single,
                "answer_as_is": answer_as_is.__dict__,
                "swap_answer_as_is": swap_as_is.__dict__,
                "answer_rstrip": answer_rs.__dict__,
                "swap_answer_rstrip": swap_rs.__dict__,
                "all_single_as_is": src.single and tgt.single and answer_as_is.single and swap_as_is.single,
                "all_single_rstrip": src.single and tgt.single and answer_rs.single and swap_rs.single,
                "same_first_token_answers_as_is": answer_as_is.first_id == swap_as_is.first_id,
                "same_first_token_answers_rstrip": answer_rs.first_id == swap_rs.first_id,
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items_file": str(ITEMS_FILE),
        "items_sha256": sha256_of(ITEMS_FILE),
        "n_items": len(rows),
        "categories": dict(collections.Counter(r["category"] for r in rows)),
        "n_trailing_space": sum(r["trailing_space"] for r in rows),
        "n_concept_feasible": sum(r["concept_feasible"] for r in rows),
        "n_all_single_as_is": sum(r["all_single_as_is"] for r in rows),
        "n_all_single_rstrip": sum(r["all_single_rstrip"] for r in rows),
        "n_leak_by_field": {f: sum(r["leak"][f] for r in rows) for f in CONCEPT_FIELDS},
        "n_same_first_token_answers_rstrip": sum(r["same_first_token_answers_rstrip"] for r in rows),
        "reverse_pairs": reverse_pairs(items),
        "concept_src_bare_fallback": [r["name"] for r in rows if r["concept_src"]["single"] and not r["concept_src"]["text"].startswith(" ")],
        "concept_tgt_bare_fallback": [r["name"] for r in rows if r["concept_tgt"]["single"] and not r["concept_tgt"]["text"].startswith(" ")],
        "infeasible_concepts": [r["name"] for r in rows if not r["concept_feasible"]],
        "multi_token_answers_rstrip": [r["name"] for r in rows if not (r["answer_rstrip"]["single"] and r["swap_answer_rstrip"]["single"])],
    }
    out = results_dir("block01")
    (out / "q01_data_audit_rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "q01_data_audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
