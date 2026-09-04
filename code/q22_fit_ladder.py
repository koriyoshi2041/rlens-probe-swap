#!/usr/bin/env python
"""Block 17: fit our own lens vectors across a corpus-size ladder.

The artifacts we have been using were fitted on 25 prompts; the reference
implementation states the paper used 1000 sequences and that ~100 is usable. Every
absolute number in this project inherits that. Fitting only the columns the
intervention uses (validated against the reference in tests/test_fit_vectors.py)
makes the ladder affordable: ~13 backward passes per passage instead of 4096.

Saves at each rung so the ladder survives an interruption, and records the running
mean's stability -- the change from one rung to the next is the empirical
convergence rate of the estimator.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from rlens.data import load_items
from rlens.fit_vectors import FittedVectorLens, lens_vectors_for_prompt
from rlens.model import load_model
from rlens.paths import MATS_ROOT, RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from transformer_lens.tools.analysis.jacobian_lens import _frozen_parameters

RUNGS = [25, 50, 100, 200, 400]
LAYERS = list(range(3, 25))


def load_corpus(path: pathlib.Path, n: int, min_chars: int = 900):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if len(line) >= min_chars:
            out.append(line)
        if len(out) >= n:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-n", type=int, default=400)
    args = parser.parse_args()
    out_dir = results_dir("block17_fitladder")
    model = load_model()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    token_ids = sorted({t for r in resolved.values() for t in r.tracked_ids})
    print(f"[ladder] {len(token_ids)} unique tracked tokens, layers {LAYERS[0]}..{LAYERS[-1]}", flush=True)
    cotangents = model.W_U[:, token_ids].float().T.contiguous()
    passages = load_corpus(MATS_ROOT / "assets" / "corpus" / "passages.txt", args.max_n)
    print(f"[ladder] {len(passages)} passages", flush=True)
    totals = None
    done = 0
    prev = None
    t0 = time.time()
    rungs = [r for r in RUNGS if r <= args.max_n]
    with _frozen_parameters(model):
        for passage in passages:
            tokens = model.to_tokens(passage)[:, : args.max_seq_len]
            if tokens.shape[1] <= 17:
                continue
            part = lens_vectors_for_prompt(model, tokens, cotangents, LAYERS, dim_batch=args.dim_batch)
            if totals is None:
                totals = part
            else:
                for layer in LAYERS:
                    totals[layer] += part[layer]
            done += 1
            if done in rungs:
                mean = {layer: totals[layer] / done for layer in LAYERS}
                lens = FittedVectorLens(mean, token_ids, done,
                                        metadata={"corpus": "wikitext103", "max_seq_len": args.max_seq_len,
                                                  "dim_batch": args.dim_batch, "estimator": "column-wise, parity-tested"})
                lens.save(str(out_dir / f"vectors_n{done}.pt"))
                drift = None
                if prev is not None:
                    num = sum(float((mean[l] - prev[l]).norm() ** 2) for l in LAYERS) ** 0.5
                    den = sum(float(mean[l].norm() ** 2) for l in LAYERS) ** 0.5
                    drift = num / den
                prev = {l: mean[l].clone() for l in LAYERS}
                rec = {"n": done, "seconds": round(time.time() - t0, 1),
                       "per_passage": round((time.time() - t0) / done, 2),
                       "relative_drift_from_previous_rung": drift}
                (out_dir / f"rung_n{done}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
                print(f"[ladder] n={done} saved  {rec['per_passage']}s/passage  drift={drift}", flush=True)
            if done >= args.max_n:
                break
    print("[ladder] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
