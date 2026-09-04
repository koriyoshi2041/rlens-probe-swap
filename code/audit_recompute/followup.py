import sys, json, collections
sys.path.insert(0, ".")
from audit_lib import *
import numpy as np
P, E = "primary_L8_20", "early_L3_8"

print("== Claim 9 follow-up ==")
c8 = load_jsonl("block08_clamp/clamp.jsonl"); sup = load_jsonl("block12_suppression/suppression.jsonl")
cr = load_jsonl("block04/cross_readout.jsonl")
print("cross_readout sample:", {k: (v if not isinstance(v, list) else v[:5]) for k, v in cr[0].items()})
flJ = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens="J", arm="clamp")}
flR = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens="R", arm="clamp")}
for reader in "JR":
    rows = sel(cr, arm="clean", reader=reader)
    for agg, fn in (("min", min), ("median", np.median), ("first(L8)", lambda v: v[0])):
        rk = {r["index"]: fn(r["best_rank_intermediate"]) for r in rows}
        print(f"  reader {reader} clean intermediate rank ({agg}) -> flip: J-clamp {spearman([rk[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f} | R-clamp {spearman([rk[i] for i in ELIGIBLE],[flR[i] for i in ELIGIBLE]):+.3f} | median rank {np.median([rk[i] for i in ELIGIBLE]):.0f}")
# other readout: block08 clamp_readout? (post-clamp) ; block01 readout_summary? try hit-rate style
dmJ = per_item(sel(sup, lens="J", arm="clamp"))
print("  ΔM -> flip alternatives:")
print(f"    Spearman(clamp ΔM, flip) = {spearman([dmJ[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}")
x = np.array([dmJ[i] for i in ELIGIBLE]); y = np.array([flJ[i] for i in ELIGIBLE])
print(f"    Pearson(clamp ΔM, flip) = {np.corrcoef(x,y)[0,1]:+.3f}")
for arm in ("install", "remove", "clamp_plus_ablate"):
    d = per_item(sel(sup, lens="J", arm=arm)); print(f"    Spearman({arm} ΔM, clamp flip) = {spearman([d[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}")
c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
d = per_item(sel(c9, band=P, lens="J", pair="entity", arm="ortho_rescaled")); print(f"    Spearman(ortho_rescaled clamp ΔM, full flip) = {spearman([d[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}")
rec = load_jsonl("block02/block02_records.jsonl")
d = per_item(sel(rec, condition="swap_raw", band=P, lens="J", alpha=1.0, positions="all")); print(f"    Spearman(involution ΔM, clamp flip) = {spearman([d[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}")
geo = load_jsonl("block05_answer_swap/geometry.jsonl"); cos = {r["index"]: r["cos_entitydiff_answerdiff"] for r in geo}
print(f"    Spearman(cos, ortho_rescaled ΔM) = {spearman([cos[i] for i in ELIGIBLE],[d2 for d2 in [per_item(sel(c9, band=P, lens='J', pair='entity', arm='ortho_rescaled'))[i] for i in ELIGIBLE]]):+.3f}")
dsw = per_item(sel(sup, lens="J", arm="clamp"), "delta_logp_swap_answer"); print(f"    Spearman(Δlogp swap, flip) = {spearman([dsw[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}")
# excess over threshold?
clean = {r["index"]: r for r in sel(rec, condition="clean")}
req = {i: clean[i]["logp_answer"] - clean[i]["logp_swap_answer"] for i in ELIGIBLE}
print(f"    Spearman(ΔM - required, flip) = {spearman([dmJ[i]-req[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}")
# R lens
dmR = per_item(sel(sup, lens="R", arm="clamp")); print(f"    R: Spearman(cos, ΔM) = {spearman([cos[i] for i in ELIGIBLE],[dmR[i] for i in ELIGIBLE]):+.3f}; Spearman(ΔM, flip) = {spearman([dmR[i] for i in ELIGIBLE],[flR[i] for i in ELIGIBLE]):+.3f}")
# both lenses pooled
xs = [dmJ[i] for i in ELIGIBLE] + [dmR[i] for i in ELIGIBLE]; ys = [flJ[i] for i in ELIGIBLE] + [flR[i] for i in ELIGIBLE]
print(f"    pooled J+R Spearman(ΔM, flip) = {spearman(xs, ys):+.3f}")
# E3 actual?
e3 = load_jsonl("block13_deep/e3_gradient.jsonl")
a = {r["index"]: r["actual"] for r in sel(e3, lens="J")}; f3 = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(e3, lens="J")}
print(f"    E3: Spearman(actual, flip) = {spearman([a[i] for i in ELIGIBLE],[f3[i] for i in ELIGIBLE]):+.3f}; Spearman(pred, actual) = {spearman([r['predicted_linear'] for r in sel(e3,lens='J')],[r['actual'] for r in sel(e3,lens='J')]):+.3f}; mean pred {np.mean([r['predicted_linear'] for r in sel(e3,lens='J')]):+.2f} mean actual {np.mean([r['actual'] for r in sel(e3,lens='J')]):+.2f}")

print("\n== Claim 7 follow-up: odd 3..13 minus even 2..12 ==")
m3 = load_jsonl("block11_mechanism/m3_parity.jsonl")
for arm in ("involution", "clamp"):
    for lens in "JR":
        vals = {}
        for i in ELIGIBLE:
            byw = {x["width"]: x["delta_margin"] for x in m3 if x["index"] == i and x["lens"] == lens and x["arm"] == arm}
            vals[i] = float(np.mean([byw[w] for w in (3,5,7,9,11,13)]) - np.mean([byw[w] for w in (2,4,6,8,10,12)]))
        m, lo, hi, n, nc = cboot(vals)
        print(f"  {lens} {arm:10s}: odd(3..13)-even(2..12) = {m:+.3f} [{lo:+.3f},{hi:+.3f}]")

print("\n== Claim 10 E2 follow-up: additivity ==")
e2 = load_jsonl("block13_deep/e2_positions.jsonl")
offs = sorted({r["offset_from_end"] for r in e2})
per_off_med = {o: med(sel(e2, lens="J", offset_from_end=o), "delta_margin") for o in offs}
print(f"  sum over offsets of per-offset median ΔM (J): {sum(per_off_med.values()):.3f}; offsets -5..0 only: {sum(per_off_med[o] for o in (-5,-4,-3,-2,-1,0)):.3f}")
per_off_med_R = {o: med(sel(e2, lens="R", offset_from_end=o), "delta_margin") for o in offs}
print(f"  same for R: {sum(per_off_med_R.values()):.3f}")
sums = collections.defaultdict(float)
for x in e2:
    if x["lens"] == "J": sums[x["index"]] += x["delta_margin"]
full = per_item(sel(sup, lens="J", arm="clamp"))
ratio = {i: full[i] / sums[i] if abs(sums[i]) > 1e-9 else float('nan') for i in ELIGIBLE}
print(f"  per-item sum of single-position ΔM: mean {np.mean([sums[i] for i in ELIGIBLE]):.3f} median {np.median([sums[i] for i in ELIGIBLE]):.3f}; full clamp mean {np.mean([full[i] for i in ELIGIBLE]):.3f} median {np.median([full[i] for i in ELIGIBLE]):.3f}")
print(f"  mean(full)/mean(sum) = {np.mean([full[i] for i in ELIGIBLE])/np.mean([sums[i] for i in ELIGIBLE]):.2f}; median per-item full/sum ratio = {np.nanmedian(list(ratio.values())):.2f}")
m, lo, hi, n, nc = cboot({i: full[i] - sums[i] for i in ELIGIBLE}); print(f"  cluster bootstrap of (full - sum of singles): {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
flipsJ = {r["index"]: bool(r["top1_is_swap"]) for r in sel(sup, lens="J", arm="clamp")}
print(f"  among flippable items: mean sum {np.mean([sums[i] for i in ELIGIBLE if flipsJ[i]]):.2f} vs full {np.mean([full[i] for i in ELIGIBLE if flipsJ[i]]):.2f}; zero-flip: sum {np.mean([sums[i] for i in ELIGIBLE if not flipsJ[i]]):.2f} vs full {np.mean([full[i] for i in ELIGIBLE if not flipsJ[i]]):.2f}")
# leave-one-out vs solo at offset -3
pr = load_jsonl("block18_rederivation/position_repair.jsonl")
fullpr = per_item(sel(pr, spared=None))
for off in (-5, -3, -1, 0):
    spared = per_item(sel(pr, spared_offset=off)); solo = per_item(sel(e2, lens="J", offset_from_end=off))
    loss = {i: fullpr[i] - spared[i] for i in ELIGIBLE}
    print(f"  offset {off:3d}: marginal loss mean {np.mean(list(loss.values())):+.3f} median {np.median(list(loss.values())):+.3f} | solo mean {np.mean([solo[i] for i in ELIGIBLE]):+.3f} median {np.median([solo[i] for i in ELIGIBLE]):+.3f} | Spearman(solo, loss) {spearman([solo[i] for i in ELIGIBLE],[loss[i] for i in ELIGIBLE]):+.3f} | med(full)-med(spared) {np.median(list(fullpr.values()))-np.median(list(spared.values())):+.3f}")
# check position_repair full vs block12 clamp identical
d = paired(fullpr, full); print("  position_repair full - block12 clamp:", fmt(d))

print("\n== Claim 14 follow-up ==")
s1 = load_jsonl("block20_specificity/s1_scale_matched.jsonl")
for arm in ("true_patch", "other_item_patch_matched", "random_matched"):
    r = sel(s1, arm=arm); print(f"  {arm:26s}: meanΔM {mean(r,'delta_margin'):+.3f} medΔM {med(r,'delta_margin'):+.3f} meanKL {mean(r,'kl'):.3f} medKL {med(r,'kl'):.3f} items {sorted({x['index'] for x in r})}")
    print("     top1_str:", collections.Counter(x["top1_str"] for x in r).most_common(6))

print("\n== Claim 16 follow-up: modes over all 90 ==")
cr90 = json.load(open(RES / "block01" / "clean_rows.json")); by = {(r["index"], r["mode"]): r for r in cr90}
modes90 = collections.Counter()
for i in range(90):
    s = by[(i, "rstrip")]; modes90["rstrip" if (s["answers_single"] and s["answers_distinct"]) else "as_is"] += 1
print("  per-item mode over 90 items:", dict(modes90))
man = json.load(open(RES / "block02" / "block02_meta.json")); print("  manifest modes over eligible:", collections.Counter(man["modes"].values()))

print("\n== Oddities: block02 group sizes and energies ==")
groups = collections.Counter((r["condition"], r["band"], r["lens"], r["alpha"], r["positions"], r["seed"]) for r in rec)
odd = {k: v for k, v in groups.items() if v != 59}
print("  groups with n != 59:", len(odd), "of", len(groups)); 
for k, v in sorted(odd.items(), key=str)[:12]: print("   ", k, v)
print("  foil rows per (band,lens,alpha):", collections.Counter((r["band"], r["lens"], r["alpha"]) for r in sel(rec, condition="swap_foil")))
print("  median energy by condition/band/lens/alpha (only > 1e4 shown):")
eg = collections.defaultdict(list)
for r in rec: eg[(r["condition"], r["band"], r["lens"], r["alpha"])].append(r["total_dh2"])
for k, v in sorted(eg.items(), key=str):
    if np.median(v) > 1e4: print(f"    {k}: median {np.median(v):.3g} max {max(v):.3g} n={len(v)}")
print("  early band swap_raw energies by alpha (J):", {a: f"{np.median(eg[('swap_raw',E,'J',a)]):.3g}" for a in (0.5,1.0,2.0,4.0)})
print("  single-layer alpha=1 median energy J:", {L: round(float(np.median([r['total_dh2'] for r in sel(rec, condition='swap_single_layer', lens='J', layer=L)])),1) for L in (3,6,9,12,16,20)})
# any records where swap changed nothing (energy 0) other than clean?
z = [r for r in rec if r["condition"] != "clean" and r["total_dh2"] == 0.0]; print("  non-clean records with zero energy:", len(z), collections.Counter((r['condition'], r['lens']) for r in z).most_common(5))
# top1 flip at alpha=2 anywhere? swap_unit etc.
for cond in ("swap_unit", "swap_shuffled", "swap_gauss", "swap_foil"):
    for band in (P, E):
        r = sel(rec, condition=cond, band=band, alpha=2.0)
        print(f"  {cond:14s} {band:13s} a=2: n={len(r)} flip {rate(r)[0]:.2f} medΔM {med(r,'delta_margin'):+.1f} med energy {med(r,'total_dh2'):.3g}")
# wall_ms sanity, kl extremes
print("  max KL:", max(r["kl_clean_to_hooked"] for r in rec), " min KL:", min(r["kl_clean_to_hooked"] for r in rec))
print("  neither_mass range:", min(r["neither_mass"] for r in rec), max(r["neither_mass"] for r in rec))

print("\n== block21 finite difference quick check ==")
fd = load_jsonl("block21_finite_difference/finite_difference.jsonl")
for d_ in ("random", "clamp"):
    for t in sorted({r["target_rel"] for r in fd}):
        r = sel(fd, direction=d_, target_rel=t)
        if r: print(f"  {d_:6s} target_rel {t}: n={len(r)} median ratio_central {med(r,'ratio_central'):.3f} median ratio_one_sided {med(r,'ratio_one_sided'):.3f} med achieved_rel {med(r,'achieved_rel'):.4f}")

print("\n== block07 patch: frac var explained & diffmean flip (claim 6 ladder row 'diffmean 0.05') ==")
pat = load_jsonl("block07_causal/patch.jsonl")
for lens in "JR":
    for L in (6, 8, 12, 16, 20):
        v = [r["frac_var_explained_by_lens_dir"] for r in sel(pat, lens=lens, layer=L, arm="lens_swap") if r["frac_var_explained_by_lens_dir"] is not None]
        print(f"  {lens} L{L} frac var explained by lens dir: median {np.median(v) if v else float('nan'):.3f} (n={len(v)})", end=" | ")
    print()
dm = load_jsonl("block07_causal/diffmean.jsonl")
for arm in ("diffmean_matched", "lens_swap"):
    print(f"  diffmean {arm}: flip by layer", {L: round(rate(sel(dm, arm=arm, layer=L, lens='J'))[0], 2) for L in (6,8,10,12,14,16,20)})
