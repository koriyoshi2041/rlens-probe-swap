"""Regenerate every write-up figure into 正式研究/figures_paper/. CPU only.

Usage: python3 make_all.py            (from 正式研究/code/figpaper/)
"""
from __future__ import annotations

import importlib
import traceback

MODULES = [
    "fig_overview",
    "fig_parity",
    "fig_ladder",
    "fig_crossreadout",
    "fig_outcomes",
    "fig_entityprobe",
    "fig_rj_layers",
    "fig_rj_bands",
    "fig_paste",
    "fig_subspace_sufficiency",
    "fig_subspace_necessity",
    "fig_routing_attention",
    "fig_routing_transplant",
    "fig_recipe",
    "fig_stage",
    "fig_native",
    "fig_propagation",
    "fig_third_model",
    "fig_alpha",
]

if __name__ == "__main__":
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name).make()
        except Exception:  # keep going; report at the end
            failed.append(name)
            traceback.print_exc()
    if failed:
        print("FAILED:", failed)
