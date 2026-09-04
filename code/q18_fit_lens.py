#!/usr/bin/env python
"""Block 14: fit our own J-lens at several corpus sizes.

Why. The published artifacts we have been using were fitted on 25 prompts. The
reference implementation's own README says the paper's lenses use 1000 sequences
of 128 tokens and that "~100 prompts is usable" -- so 25 is below the threshold
its authors state. Every absolute number we report inherits that. Fitting our own
lens at increasing corpus sizes turns an unquantified caveat into a measured
curve: if intervention success keeps climbing with n_prompts, the published
artifact is the binding constraint; if it plateaus by 25, the gap to the paper's
61-70% comes from somewhere else.

The corpus is a pretraining-like web-text slice, matching the reference recipe
(the estimator itself is deterministic given the prompts).
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


def load_corpus(path: pathlib.Path, n: int, min_chars: int = 900) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if len(line) >= min_chars:
            texts.append(line)
        if len(texts) >= n:
            break
    if len(texts) < n:
        raise RuntimeError(f"corpus has only {len(texts)} usable passages, need {n}")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dim-batch", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--corpus", default=str(MATS_ROOT / "assets" / "corpus" / "passages.txt"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--time-only", action="store_true")
    args = parser.parse_args()

    from transformer_lens.tools.analysis import JacobianLens

    model = load_model()
    prompts = load_corpus(pathlib.Path(args.corpus), args.n)
    print(f"[fit] {len(prompts)} passages, dim_batch={args.dim_batch}, max_seq_len={args.max_seq_len}")
    t0 = time.time()
    lens = JacobianLens.fit(
        model,
        prompts,
        corpus=f"local-webtext-n{args.n}",
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        show_progress=True,
        metadata={"fitted_by": "mats-r-lens block14", "n_requested": args.n},
    )
    elapsed = time.time() - t0
    print(f"[fit] done in {elapsed:.0f}s ({elapsed / len(prompts):.1f}s per passage)")
    if args.time_only:
        return 0
    out = pathlib.Path(args.out or (results_dir("block14_fit") / f"jlens_n{args.n}.pt"))
    lens.save(str(out))
    meta = {"n": args.n, "seconds": elapsed, "path": str(out), "d_model": lens.d_model,
            "n_prompts": lens.n_prompts, "source_layers": [lens.source_layers[0], lens.source_layers[-1]]}
    (results_dir("block14_fit") / f"fit_n{args.n}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[fit] saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
