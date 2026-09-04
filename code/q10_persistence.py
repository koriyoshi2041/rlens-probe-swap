#!/usr/bin/env python
"""Why doesn't the model act on the rewrite? Does the rewrite survive past the band?

The cross-lens readout showed the swap really does exchange which entity the band
reads (53/59 items), yet only 6/59 items change their answer. One concrete
explanation is that the model repairs or ignores the edit downstream: the band is
rewritten, but by the layers that actually build the answer the original entity is
back. This run reads every layer from 0 to the top, inside and past the band, under
clean / swap / install, with the cross lens doing the reading.
"""
from __future__ import annotations

import json
import sys
import time

import torch

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.interventions import coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block06_persistence")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    band = cfg.BANDS[cfg.PRIMARY_BAND]
    all_layers = list(range(0, 31))
    handle = (out_dir / "persistence.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        tracked = r.tracked_ids
        arms = {"clean": None, "J_swap": ("J", "swap"), "R_swap": ("R", "swap"), "J_install": ("J", "install")}
        for arm, spec in arms.items():
            hooks = []
            if spec is not None:
                kind, mode = spec
                for layer in band:
                    basis = lenses[kind].lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    hooks += coordinate_map_hooks(model, basis, [layer], mode=mode, alpha=1.0)
            reader = "R" if (spec is None or spec[0] == "J") else "J"
            with model.hooks(fwd_hooks=hooks) if hooks else torch.no_grad():
                ranks, _ = rank_readout(model, lenses[reader], tokens, tracked, all_layers)
            best = ranks[:-1].min(dim=1).values.numpy()
            final_pos = ranks[:-1, -1, :].numpy()
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm, "reader": reader, "layers": all_layers,
                "best_intermediate": best[:, 0].tolist(), "best_swap_to": best[:, 1].tolist(),
                "final_intermediate": final_pos[:, 0].tolist(), "final_swap_to": final_pos[:, 1].tolist(),
                "final_answer": final_pos[:, 2].tolist(), "final_swap_answer": final_pos[:, 3].tolist(),
            }) + "\n")
        if n % 15 == 0:
            print(f"[persistence] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    print("[persistence] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
