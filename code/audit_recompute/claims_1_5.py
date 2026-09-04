import sys, json
sys.path.insert(0, __import__('os').path.dirname(__file__))
from audit_lib import *
import numpy as np

P, E = "primary_L8_20", "early_L3_8"

# ---------------- Claim 1 & 2: block02 ----------------
rec = load_jsonl("block02/block02_records.jsonl")
print("block02 records:", len(rec), check_items(rec, "block02"))
def sw(lens, band, alpha, pos="all"):
    return sel(rec, condition="swap_raw", lens=lens, band=band, alpha=alpha, positions=pos)
print("\n== Claim 1 ==")
for lens in "JR":
    r1 = sw(lens, P, 1.0); print(f"top1_swap rate {lens} primary a=1: {rate(r1)}  meanΔM={mean(r1,'delta_margin'):+.3f} medΔM={med(r1,'delta_margin'):+.3f}")
d = paired(per_item(sw("R", P, 1.0)), per_item(sw("J", P, 1.0))); print("R-J primary a=1:", fmt(d))
d = paired(per_item(sw("R", E, 0.5)), per_item(sw("J", E, 0.5))); print("R-J early a=0.5:", fmt(d))
d = paired(per_item(sw("R", E, 1.0)), per_item(sw("J", E, 1.0))); print("R-J early a=1.0:", fmt(d))
d = paired(per_item(sw("R", P, 0.5)), per_item(sw("J", P, 0.5))); print("R-J primary a=0.5:", fmt(d))
for lens in "JR":
    r2 = sw(lens, P, 2.0); print(f"a=2 primary {lens}: top1_swap {rate(r2)} medΔM {med(r2,'delta_margin'):+.2f} med Δans {med(r2,'delta_logp_answer'):+.1f} med Δswap {med(r2,'delta_logp_swap_answer'):+.1f} medKL {med(r2,'kl_clean_to_hooked'):.1f} neither {1-rate(r2)[0]-rate(r2,'top1_is_answer')[0]:.2f}")
for a in (0.5, 1.0, 2.0, 4.0):
    rows = sw("J", P, a) + sw("R", P, a)
    print(f"  median energy primary a={a}: pooled J+R {med(rows,'total_dh2'):.4g} | J {med(sw('J',P,a),'total_dh2'):.4g} | R {med(sw('R',P,a),'total_dh2'):.4g}")
# max_rel per layer at a=1 and a=2 (J)
for a in (1.0, 2.0):
    rows = sw("J", P, a)
    per_layer = {}
    for r in rows:
        for e in r["energy"]:
            per_layer.setdefault(e["layer"], []).append(e["max_rel"])
    print(f"  a={a} J median max_rel per layer:", {l: round(float(np.median(v)), 3) for l, v in sorted(per_layer.items())})

print("\n== Claim 2 ==")
for lens in "JR":
    u = sel(rec, condition="swap_unit", lens=lens, band=P, alpha=1.0)
    d = paired(per_item(u), per_item(sw(lens, P, 1.0))); print(f"{lens} unit - raw primary a=1:", fmt(d))
for lens in "JR":
    g = sel(rec, condition="swap_gauss", lens=lens, band=P, alpha=1.0)
    m, lo, hi, n, nc = cboot(per_item(g)); print(f"{lens} gauss primary a=1 mean ΔM {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}, flip {rate(g)}; seeds={sorted({r['seed'] for r in g})}")
    g = sel(rec, condition="swap_gauss", lens=lens, band=E, alpha=1.0)
    m, lo, hi, n, nc = cboot(per_item(g)); print(f"{lens} gauss early   a=1 mean ΔM {m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}, flip {rate(g)}")
for lens in "JR":
    sh = sel(rec, condition="swap_shuffled", lens=lens, band=P, alpha=1.0)
    print(f"{lens} shuffled rows {len(sh)} seeds {sorted({r['seed'] for r in sh})}; mean ΔM shuffled {mean(sh,'delta_margin'):+.3f}")
    d = paired(per_item(sw(lens, P, 1.0)), per_item(sh)); print(f"{lens} swap - shuffled(mean of 3 seeds):", fmt(d))
    ab = sel(rec, condition="ablate_src", lens=lens, band=P)
    print(f"  ablate rows {len(ab)} alphas {sorted({r['alpha'] for r in ab})}")
    d = paired(per_item(sw(lens, P, 1.0)), per_item(ab)); print(f"{lens} swap - ablate_src:", fmt(d))
    fo = sel(rec, condition="swap_foil", lens=lens, band=P, alpha=1.0)
    d = paired(per_item(sw(lens, P, 1.0)), per_item(fo)); print(f"{lens} swap - foil (n foil rows {len(fo)}):", fmt(d))
# shuffle seeds distinct?
for lens in "J":
    sh = sel(rec, condition="swap_shuffled", lens=lens, band=P, alpha=1.0)
    by_seed = {}
    for r in sh:
        by_seed.setdefault(r["seed"], {})[r["index"]] = r["other_index"]
    print("  shuffle map item 0 by seed:", {s: m.get(0) for s, m in by_seed.items()})
    same = [(s1, s2) for s1 in by_seed for s2 in by_seed if s1 < s2 and by_seed[s1] == by_seed[s2]]
    print("  identical seed maps:", same)
# position sensitivity
for pos in ("all", "last5", "last3"):
    r = sw("J", P, 1.0, pos); print(f"  pos {pos} J: mean ΔM {mean(r,'delta_margin'):+.2f}, med {med(r,'delta_margin'):+.2f}, flip {rate(r)}")
# single-layer top1 rates L11, L12
for L in (11, 12):
    for lens in "JR":
        r = sel(rec, condition="swap_single_layer", lens=lens, layer=L)
        print(f"  single layer L{L} {lens}: flip {rate(r)} medΔM {med(r,'delta_margin'):+.2f}")

# ---------------- Claim 3: block03_explore ----------------
print("\n== Claim 3 ==")
dec = load_jsonl("block03_explore/explore_decomposition.jsonl")
print("decomp rows", len(dec), check_items(dec, "decomp"))
for band in (P, E):
    for lens in "JR":
        line = []
        for mode in ("swap", "remove", "install"):
            r = sel(dec, band=band, lens=lens, mode=mode)
            line.append(f"{mode}: mean {mean(r,'delta_margin'):+.3f} med {med(r,'delta_margin'):+.3f} flip {rate(r)[0]:.2f} (n={len(r)})")
        print(f"{band} {lens}: " + " | ".join(line))
    for lens in "JR":
        d = paired(per_item(sel(dec, band=band, lens=lens, mode="swap")), per_item(sel(dec, band=band, lens=lens, mode="install")))
        print(f"  {band} {lens} swap - install:", fmt(d))
    d = paired(per_item(sel(dec, band=band, lens="R", mode="remove")), per_item(sel(dec, band=band, lens="J", mode="remove")))
    print(f"  {band} R-J remove:", fmt(d))
r = sel(dec, band=P, lens="J", mode="install"); print(f"  primary J install: med Δlogp(answer) {med(r,'delta_logp_answer'):+.3f}, med Δlogp(swap) {med(r,'delta_logp_swap_answer'):+.3f}")
# consistency: block03 swap vs block02 swap_raw a=1
d = paired(per_item(sel(dec, band=P, lens="J", mode="swap")), per_item(sw("J", P, 1.0))); print("  block03 J swap - block02 J swap_raw a=1 (should be ~0):", fmt(d))
# remove vs block02 ablate_src
d = paired(per_item(sel(dec, band=P, lens="J", mode="remove")), per_item(sel(rec, condition="ablate_src", lens="J", band=P))); print("  J remove - ablate_src primary:", fmt(d))

# ---------------- Claim 4: block03_ortho ----------------
print("\n== Claim 4 ==")
ort = load_jsonl("block03_ortho/ortho.jsonl")
print("ortho rows", len(ort), check_items(ort, "ortho"))
for group in (P, E, "L6", "L12"):
    for lens in "JR":
        f = sel(ort, group=group, lens=lens, arm="full"); o = sel(ort, group=group, lens=lens, arm="ortho_rescaled"); o2 = sel(ort, group=group, lens=lens, arm="ortho")
        m, lo, hi, n, nc = cboot(per_item(o))
        mf = mean(f, "delta_margin")
        print(f"{group:13s} {lens}: full mean {mf:+.3f} (med {med(f,'delta_margin'):+.3f}) flip {rate(f)[0]:.2f} | ortho_rescaled mean {m:+.3f} [{lo:+.3f},{hi:+.3f}] flip {rate(o)[0]:.2f} | ratio {m/mf:.2f} | ortho(no rescale) mean {mean(o2,'delta_margin'):+.3f} | median cos {med(f,'median_cos_direction_contrast'):+.3f}")
    d = paired(per_item(sel(ort, group=group, lens="R", arm="ortho_rescaled")), per_item(sel(ort, group=group, lens="J", arm="ortho_rescaled")))
    print(f"   R-J ortho_rescaled {group}:", fmt(d))
    d = paired(per_item(sel(ort, group=group, lens="R", arm="full")), per_item(sel(ort, group=group, lens="J", arm="full")))
    print(f"   R-J full {group}:", fmt(d))

# ---------------- Claim 5: block03_sweep ----------------
print("\n== Claim 5 ==")
swp = load_jsonl("block03_sweep/sweep.jsonl")
print("sweep rows", len(swp), check_items(swp, "sweep"))
alphas = sorted({r["alpha"] for r in swp}); print("alphas", alphas)
# check linearity of dh_norm in alpha
lin = []
for r in swp:
    if r["alpha"] != 1.0:
        pass
by = {}
for r in swp:
    by[(r["index"], r["layer"], r["lens"], r["alpha"])] = r
ratios = []
for (i, L, lens, a), r in by.items():
    base = by[(i, L, lens, 1.0)]["dh_norm"]
    if base > 0:
        ratios.append(r["dh_norm"] / base / a)
print(f"dh_norm/alpha relative to alpha=1: min {min(ratios):.4f} max {max(ratios):.4f}")
for L in (5, 6, 9, 12, 16):
    same, matched, extrap = {}, {}, 0
    for i in ELIGIBLE:
        J1 = by[(i, L, "J", 1.0)]
        rrows = sorted([by[(i, L, "R", a)] for a in alphas], key=lambda r: r["dh_norm"])
        xs = np.array([r["dh_norm"] for r in rrows]); ys = np.array([r["delta_margin"] for r in rrows])
        target = J1["dh_norm"]
        if target < xs[0] or target > xs[-1]:
            extrap += 1
        # linear interpolation on the dh_norm axis (np.interp clamps outside range)
        matched[i] = float(np.interp(target, xs, ys)) - J1["delta_margin"]
        same[i] = by[(i, L, "R", 1.0)]["delta_margin"] - J1["delta_margin"]
    m1, lo1, hi1, n, nc = cboot(matched); m2, lo2, hi2, _, _ = cboot(same)
    # alternative: interpolate on alpha axis with alpha_R = dhJ/dhR (equivalent under linearity)
    alt = {}
    for i in ELIGIBLE:
        J1 = by[(i, L, "J", 1.0)]; R1 = by[(i, L, "R", 1.0)]
        aR = J1["dh_norm"] / R1["dh_norm"]
        xs = np.array(alphas); ys = np.array([by[(i, L, "R", a)]["delta_margin"] for a in alphas])
        alt[i] = float(np.interp(aR, xs, ys)) - J1["delta_margin"]
    m3, lo3, hi3, _, _ = cboot(alt)
    print(f"L{L:2d}: energy-matched R-J {m1:+.3f} [{lo1:+.3f},{hi1:+.3f}] | alt(alpha-axis) {m3:+.3f} [{lo3:+.3f},{hi3:+.3f}] | same-alpha R-J {m2:+.3f} [{lo2:+.3f},{hi2:+.3f}] | items outside R range: {extrap}; median dhJ/dhR at a=1 = {np.median([by[(i,L,'J',1.0)]['dh_norm']/by[(i,L,'R',1.0)]['dh_norm'] for i in ELIGIBLE]):.3f}")
