#!/usr/bin/env python
"""Block 01 driver: load the model once, run the requested qualification stages."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import torch
import transformers
import transformer_lens

from rlens.data import load_items
from rlens.model import load_lenses, load_model, model_summary
from rlens.paths import results_dir
from rlens.stages import artifacts, clean, poscontrol, readout_scan

STAGES = ("clean", "artifacts", "readout", "poscontrol")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default=",".join(STAGES))
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")

    out_dir = results_dir("block01")
    timing = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": {}}
    t0 = time.time()
    model = load_model()
    lenses = load_lenses()
    timing["load_seconds"] = round(time.time() - t0, 1)
    header = {
        "model": model_summary(model),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformer_lens": transformer_lens.__file__,
    }
    print("[driver] header:", json.dumps(header, indent=1))
    items = load_items()

    clean_rows = None
    if "clean" in stages:
        t = time.time()
        clean_rows = clean.run(model, items, out_dir)
        timing["stages"]["clean"] = round(time.time() - t, 1)
    else:
        clean_rows = json.loads((out_dir / "clean_rows.json").read_text(encoding="utf-8"))
    if "artifacts" in stages:
        t = time.time()
        artifacts.run(model, lenses, items, out_dir)
        timing["stages"]["artifacts"] = round(time.time() - t, 1)
    if "readout" in stages:
        t = time.time()
        readout_scan.run(model, lenses, items, clean_rows, out_dir)
        timing["stages"]["readout"] = round(time.time() - t, 1)
    if "poscontrol" in stages:
        t = time.time()
        poscontrol.run(model, lenses, items, clean_rows, out_dir)
        timing["stages"]["poscontrol"] = round(time.time() - t, 1)
    timing["finished_at"] = datetime.now(timezone.utc).isoformat()
    timing["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    (out_dir / "timing.json").write_text(json.dumps({**header, **timing}, indent=1), encoding="utf-8")
    print("[driver] done:", json.dumps(timing, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
