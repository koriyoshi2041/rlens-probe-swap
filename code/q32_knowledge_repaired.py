#!/usr/bin/env python
"""Block 04b: the single-hop knowledge check with the ten REPAIRED templates.

Why. `data/single_hop_templates.json` carries a `_repairs_2026_09_02` note and ten rewritten
templates, and the log says the sensitivity analysis uses the repaired run. But
`block04/knowledge.json` is byte-identical to `block02/knowledge.json`: the rerun happened
before the file was edited, so the repaired templates were never scored. This block runs
them and writes to a fresh directory so both versions remain on disk.
"""
from __future__ import annotations

import json
import sys

from rlens.data import load_items
from rlens.model import load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages import knowledge
from rlens.stages.common import eligible_hybrid


def main() -> int:
    out_dir = results_dir("block04b_knowledge_repaired")
    model = load_model()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    rows = knowledge.run(model, items, eligible, out_dir)
    reachable = sorted({r["index"] for r in rows if r["side"] == "swap_to" and r["correct"]})
    (out_dir / "knowledge_reachable_indices.json").write_text(json.dumps(reachable), encoding="utf-8")
    print(f"[block04b] swap_to reachable n={len(reachable)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
