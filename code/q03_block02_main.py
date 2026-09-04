#!/usr/bin/env python
"""Block 02 driver: knowledge ceiling, pre-registered swap run, qualitative sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import torch
import transformers
import transformer_lens

from rlens import config_block02 as cfg
from rlens.data import load_items, sha256_of
from rlens.model import lens_summary, load_lenses, load_model, model_summary
from rlens.paths import ITEMS_FILE, RESULTS_DIR, results_dir
from rlens.stages import knowledge, mainrun, qualitative
from rlens.stages.common import eligible_hybrid


def code_hashes() -> dict:
    root = pathlib.Path(__file__).resolve().parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(root.rglob("*.py")) if "__pycache__" not in str(p)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="knowledge,main,qual")
    parser.add_argument("--limit", type=int, default=None, help="only the first N eligible items (smoke test)")
    parser.add_argument("--out", default="block02")
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out_dir = results_dir(args.out)
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    t0 = time.time()
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(cfg).items() if k.isupper()},
        "n_eligible": len(eligible),
        "eligible": eligible,
        "limit": args.limit,
        "model": model_summary(model),
        "lenses": {k: lens_summary(v) for k, v in lenses.items()},
        "items_sha256": sha256_of(ITEMS_FILE),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "transformer_lens": transformer_lens.__file__},
        "code_sha256_prefix": code_hashes(),
        "load_seconds": round(time.time() - t0, 1),
        "stages": {},
    }
    if "knowledge" in stages:
        t = time.time()
        knowledge.run(model, items, eligible[: args.limit] if args.limit else eligible, out_dir)
        manifest["stages"]["knowledge"] = round(time.time() - t, 1)
    if "main" in stages:
        t = time.time()
        records = mainrun.run(model, lenses, items, clean_rows, eligible, out_dir, limit=args.limit)
        manifest["stages"]["main"] = round(time.time() - t, 1)
        manifest["n_records"] = len(records)
    if "qual" in stages:
        t = time.time()
        qualitative.run(model, lenses, items, clean_rows, eligible[: args.limit] if args.limit else eligible, out_dir)
        manifest["stages"]["qual"] = round(time.time() - t, 1)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("[driver] done:", json.dumps({k: manifest[k] for k in ("stages", "n_eligible", "load_seconds", "peak_cuda_allocated_gib")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
