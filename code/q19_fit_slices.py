#!/usr/bin/env python
"""Block 14: fit J-lens slices and merge them into a corpus-size ladder.

JacobianLens.merge combines lenses fitted on disjoint prompt slices into the lens
the whole corpus would have produced, so fitting four slices of 25 passages gives
n = 25, 50, 75 and 100 from a single pass instead of four. Each slice is saved as
it finishes, so an interrupted run still yields every ladder point below it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from rlens.model import load_model
from rlens.paths import MATS_ROOT, results_dir
from q18_fit_lens import load_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-size", type=int, default=25)
    parser.add_argument("--n-slices", type=int, default=4)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--layers", default="", help="comma-separated source layers; empty = all")
    args = parser.parse_args()

    from transformer_lens.tools.analysis import JacobianLens

    out_dir = results_dir("block14_fit")
    model = load_model()
    total = args.slice_size * args.n_slices
    passages = load_corpus(pathlib.Path(MATS_ROOT / "assets" / "corpus" / "passages.txt"), total)
    layers = [int(x) for x in args.layers.split(",") if x.strip()] or None
    print(f"[slices] {args.n_slices} x {args.slice_size} passages, dim_batch={args.dim_batch}, layers={layers or 'all'}", flush=True)
    parts = []
    for k in range(args.n_slices):
        chunk = passages[k * args.slice_size : (k + 1) * args.slice_size]
        t0 = time.time()
        part = JacobianLens.fit(
            model, chunk, corpus=f"wikitext103-slice{k}", dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len, source_layers=layers, show_progress=False,
            metadata={"fitted_by": "mats-r-lens block14", "slice": k},
        )
        elapsed = time.time() - t0
        path = out_dir / f"jlens_slice{k}.pt"
        part.save(str(path))
        parts.append(part)
        merged = JacobianLens.merge(parts) if len(parts) > 1 else parts[0]
        n = args.slice_size * (k + 1)
        merged.save(str(out_dir / f"jlens_n{n}.pt"))
        record = {"n": n, "slice_seconds": round(elapsed, 1), "per_passage_seconds": round(elapsed / len(chunk), 1),
                  "n_prompts": merged.n_prompts, "layers": [merged.source_layers[0], merged.source_layers[-1]]}
        (out_dir / f"fit_n{n}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"[slices] slice {k} done in {elapsed:.0f}s ({elapsed / len(chunk):.1f}s/passage); merged n={n} saved", flush=True)
    print("[slices] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
