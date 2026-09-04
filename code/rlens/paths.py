"""Paths. Everything lives under MATS_ROOT (set the env var; on the original GPU box it was /root/autodl-tmp/mats-r-lens).
Fallback: <repo>/workspace, i.e. assets/, src/jacobian-lens/, work/results next to this code tree."""
from __future__ import annotations

import os
import pathlib

MATS_ROOT = pathlib.Path(os.environ.get("MATS_ROOT", str(pathlib.Path(__file__).resolve().parents[2] / "workspace")))
MODEL_DIR = MATS_ROOT / "assets" / "Qwen3.5-9B"
LENS_FILES = {
    "J": MATS_ROOT / "assets" / "workspace-lenses" / "qwen3.5-9b" / "j-lens" / "lens.pt",
    "R": MATS_ROOT / "assets" / "workspace-lenses" / "qwen3.5-9b" / "r-lens" / "lens.pt",
}
ITEMS_FILE = MATS_ROOT / "src" / "jacobian-lens" / "data" / "experiments" / "probe-swap.json"
RESULTS_DIR = MATS_ROOT / "work" / "results"
LOGS_DIR = MATS_ROOT / "work" / "logs"


def results_dir(block: str) -> pathlib.Path:
    out = RESULTS_DIR / block
    out.mkdir(parents=True, exist_ok=True)
    return out
