#!/usr/bin/env python
"""Block 34: refit J-lens vectors under the PUBLISHED artifact's convention (pile-10k, skip_first=4).

Why. Block 17/19 fitted on wikitext-103 with the library default skip_first_positions=16
and found cos 0.764 to the published artifact but identical behaviour. The audit flagged
that the geometry gap conflates corpus, skip and n. This block fits n=25 and n=100 on
NeelNanda/pile-10k (the artifact's stated corpus) with skip_first=4 and t_max=128, so the
comparison with the published n=25 artifact is like-for-like except for the exact 25
documents. Evaluation (clamp / involution flip rates, cosine) is done by q38.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch
from transformer_lens.tools.analysis.jacobian_lens import _frozen_parameters

from rlens.data import load_items
from rlens.fit_vectors import FittedVectorLens, lens_vectors_for_prompt
from rlens.model import load_model
from rlens.paths import MATS_ROOT, RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

RUNGS = [25, 100]
LAYERS = list(range(3, 25))
SKIP = 4
T_MAX = 128


def load_pile(n: int):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                           repo_type="dataset", revision="127bfedcd5047750df5ccf3a12979a47bfa0bafa",
                           local_dir=str(MATS_ROOT / "assets" / "corpus" / "pile-10k"))
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["text"])
        texts = table.column("text").to_pylist()
    except Exception:
        import pandas as pd

        texts = pd.read_parquet(path, columns=["text"])["text"].tolist()
    return texts[:n]


def main() -> int:
    out_dir = results_dir("block34_refit_pile")
    model = load_model()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    token_ids = sorted({t for r in resolved.values() for t in r.tracked_ids})
    cotangents = model.W_U[:, token_ids].float().T.contiguous()
    docs = load_pile(400)
    print(f"[refit] {len(docs)} pile-10k docs loaded; skip_first={SKIP}, t_max={T_MAX}", flush=True)
    totals, done, t0 = None, 0, time.time()
    with _frozen_parameters(model):
        for doc in docs:
            tokens = model.to_tokens(doc)[:, :T_MAX]
            if tokens.shape[1] <= SKIP + 1:
                continue
            part = lens_vectors_for_prompt(model, tokens, cotangents, LAYERS, dim_batch=16, skip_first_positions=SKIP)
            totals = part if totals is None else {l: totals[l] + part[l] for l in LAYERS}
            done += 1
            if done in RUNGS:
                mean = {l: totals[l] / done for l in LAYERS}
                lens = FittedVectorLens(mean, token_ids, done, metadata={"corpus": "NeelNanda/pile-10k", "skip_first": SKIP, "t_max": T_MAX, "estimator": "column-wise"})
                lens.save(str(out_dir / f"vectors_pile_n{done}.pt"))
                print(f"[refit] n={done} saved ({(time.time() - t0) / done:.1f}s/doc)", flush=True)
            if done >= max(RUNGS):
                break
    print("[block34] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
