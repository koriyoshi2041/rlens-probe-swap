"""Stage: the pre-registered swap run (block 02). One JSONL record per (item, condition)."""
from __future__ import annotations

import json
import random
import time
from typing import Dict, List, Optional, Sequence

import torch

from .. import config_block02 as cfg
from ..data import SwapItem
from ..forward import final_logprobs, kl_divergence
from ..interventions import (
    EnergyStats,
    ablation_hooks_with_stats,
    gram_matched_random_basis,
    logit_lens_vectors,
    swap_hooks_with_stats,
    unit_basis,
)
from .common import ResolvedItem, hybrid_rows, resolve_item


class Runner:
    def __init__(self, model, lenses: Dict[str, object], items: List[SwapItem], clean_rows, eligible: List[int], out_dir):
        self.model = model
        self.lenses = lenses
        self.tokenizer = model.tokenizer
        self.out_dir = out_dir
        modes = hybrid_rows(clean_rows)
        self.resolved: Dict[int, ResolvedItem] = {i: resolve_item(self.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
        self.eligible = list(eligible)
        self.tokens = {i: model.to_tokens(r.prompt, prepend_bos=False) for i, r in self.resolved.items()}
        self.clean = {i: final_logprobs(model, self.tokens[i]) for i in eligible}
        self.records: List[Dict[str, object]] = []
        self.foils = self._foil_map(items)
        self.shuffles = {seed: self._derangement(seed) for seed in cfg.SHUFFLE_SEEDS}
        self.handle = (out_dir / "block02_records.jsonl").open("w", encoding="utf-8")

    # ---------------------------------------------------------------- helpers
    def _foil_map(self, items: List[SwapItem]) -> Dict[int, Optional[int]]:
        out: Dict[int, Optional[int]] = {}
        for i in self.eligible:
            me = items[i]
            candidates = [
                j
                for j in self.eligible
                if j != i
                and items[j].category == me.category
                and items[j].swap_to.lower() not in (me.swap_to.lower(), me.intermediate.lower())
            ]
            after = [j for j in candidates if j > i]
            out[i] = (after or candidates or [None])[0]
        return out

    def _derangement(self, seed: int) -> Dict[int, int]:
        rng = random.Random(seed)
        order = list(self.eligible)
        while True:
            perm = order[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(order, perm)):
                return dict(zip(order, perm))

    def basis(self, lens_key: str, ids: Sequence[int], layer: int) -> torch.Tensor:
        if lens_key == "logit":
            return logit_lens_vectors(self.model, ids)
        return self.lenses[lens_key].lens_vectors(self.model, list(ids), layer)

    def band_hooks(self, layers: Sequence[int], basis_fn, alpha: float, positions, stats: EnergyStats):
        hooks = []
        for layer in layers:
            hooks += swap_hooks_with_stats(self.model, basis_fn(layer), [layer], alpha=alpha, positions=positions, stats=stats)
        return hooks

    def record(self, index: int, hooks, stats: EnergyStats, **meta) -> Dict[str, object]:
        r = self.resolved[index]
        clean = self.clean[index]
        a, s = r.answer.first_id, r.swap_answer.first_id
        t0 = time.time()
        logp = final_logprobs(self.model, self.tokens[index], hooks)
        wall_ms = (time.time() - t0) * 1000
        top1 = int(logp.argmax().item())
        topk = torch.topk(logp, cfg.TOPK_SAVE)
        row = {
            "index": index,
            "name": r.item.name,
            "category": r.item.category,
            "mode": r.mode,
            **meta,
            "logp_answer": float(logp[a]),
            "logp_swap_answer": float(logp[s]),
            "delta_logp_answer": float(logp[a] - clean[a]),
            "delta_logp_swap_answer": float(logp[s] - clean[s]),
            "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
            "top1_id": top1,
            "top1_str": self.tokenizer.decode([top1]),
            "top1_is_swap": top1 == s,
            "top1_is_answer": top1 == a,
            "neither_mass": float(1.0 - logp[a].exp() - logp[s].exp()),
            "kl_clean_to_hooked": kl_divergence(clean, logp),
            "total_dh2": stats.total_dh2,
            "energy": stats.as_rows(),
            "topk": [[int(i), float(v)] for v, i in zip(topk.values.tolist(), topk.indices.tolist())],
            "wall_ms": round(wall_ms, 1),
        }
        self.records.append(row)
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    # ---------------------------------------------------------------- conditions
    def run_item(self, index: int) -> None:
        r = self.resolved[index]
        ids = (r.src.first_id, r.tgt.first_id)
        base = {"condition": "clean", "lens": None, "band": None, "alpha": 0.0, "positions": "all", "seed": None}
        self.record(index, [], EnergyStats(), **base)
        for band_name, layers in cfg.BANDS.items():
            for lens_key in cfg.LENSES_SWAP:
                for alpha in cfg.ALPHAS_ALL:
                    stats = EnergyStats()
                    hooks = self.band_hooks(layers, lambda l, k=lens_key: self.basis(k, ids, l), alpha, None, stats)
                    self.record(index, hooks, stats, condition="swap_raw", lens=lens_key, band=band_name, alpha=alpha, positions="all", seed=None)
            for lens_key in cfg.LENSES_CONTROL:
                for alpha in cfg.CONTROL_ALPHAS:
                    stats = EnergyStats()
                    hooks = self.band_hooks(layers, lambda l, k=lens_key: unit_basis(self.basis(k, ids, l)), alpha, None, stats)
                    self.record(index, hooks, stats, condition="swap_unit", lens=lens_key, band=band_name, alpha=alpha, positions="all", seed=None)
                    for seed in cfg.SHUFFLE_SEEDS:
                        other = self.resolved[self.shuffles[seed][index]]
                        other_ids = (other.src.first_id, other.tgt.first_id)
                        stats = EnergyStats()
                        hooks = self.band_hooks(layers, lambda l, k=lens_key, o=other_ids: self.basis(k, o, l), alpha, None, stats)
                        self.record(index, hooks, stats, condition="swap_shuffled", lens=lens_key, band=band_name, alpha=alpha, positions="all", seed=seed, other_index=other.item.index, other_name=other.item.name)
                    for seed in cfg.GAUSS_SEEDS:
                        stats = EnergyStats()
                        hooks = self.band_hooks(layers, lambda l, k=lens_key, sd=seed: gram_matched_random_basis(self.basis(k, ids, l), seed=sd * 1000 + l), alpha, None, stats)
                        self.record(index, hooks, stats, condition="swap_gauss", lens=lens_key, band=band_name, alpha=alpha, positions="all", seed=seed)
                    foil = self.foils[index]
                    if foil is not None:
                        foil_ids = (r.src.first_id, self.resolved[foil].tgt.first_id)
                        stats = EnergyStats()
                        hooks = self.band_hooks(layers, lambda l, k=lens_key, f=foil_ids: self.basis(k, f, l), alpha, None, stats)
                        self.record(index, hooks, stats, condition="swap_foil", lens=lens_key, band=band_name, alpha=alpha, positions="all", seed=None, foil_index=foil, foil_name=self.resolved[foil].item.name, foil_target=self.resolved[foil].tgt.text)
                stats = EnergyStats()
                hooks = []
                for layer in layers:
                    hooks += ablation_hooks_with_stats(self.model, self.basis(lens_key, [ids[0]], layer), [layer], stats=stats)
                self.record(index, hooks, stats, condition="ablate_src", lens=lens_key, band=band_name, alpha=None, positions="all", seed=None)
        primary = cfg.BANDS[cfg.PRIMARY_BAND]
        for pos_name, positions in cfg.POSITION_SENSITIVITY.items():
            for lens_key in cfg.LENSES_CONTROL:
                stats = EnergyStats()
                hooks = self.band_hooks(primary, lambda l, k=lens_key: self.basis(k, ids, l), cfg.HEADLINE_ALPHA, positions, stats)
                self.record(index, hooks, stats, condition="swap_raw", lens=lens_key, band=cfg.PRIMARY_BAND, alpha=cfg.HEADLINE_ALPHA, positions=pos_name, seed=None)
        for layer in cfg.SINGLE_LAYER_PROFILE:
            for lens_key in cfg.LENSES_CONTROL:
                stats = EnergyStats()
                hooks = self.band_hooks([layer], lambda l, k=lens_key: self.basis(k, ids, l), cfg.HEADLINE_ALPHA, None, stats)
                self.record(index, hooks, stats, condition="swap_single_layer", lens=lens_key, band=f"L{layer}", alpha=cfg.HEADLINE_ALPHA, positions="all", seed=None, layer=layer)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


@torch.inference_mode()
def run(model, lenses, items, clean_rows, eligible, out_dir, limit: Optional[int] = None) -> List[Dict[str, object]]:
    chosen = eligible[:limit] if limit else eligible
    runner = Runner(model, lenses, items, clean_rows, chosen, out_dir)
    meta = {"eligible": chosen, "foils": runner.foils, "shuffles": {str(k): v for k, v in runner.shuffles.items()},
            "modes": {i: runner.resolved[i].mode for i in chosen}}
    (out_dir / "block02_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(chosen):
        runner.run_item(index)
        print(f"[mainrun] item {n + 1}/{len(chosen)} ({runner.resolved[index].item.name}) done; records={len(runner.records)}; elapsed={time.time() - t0:.0f}s", flush=True)
    runner.close()
    return runner.records
