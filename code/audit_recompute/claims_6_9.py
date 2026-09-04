import sys, json
sys.path.insert(0, __import__('os').path.dirname(__file__))
from audit_lib import *
import numpy as np

P, E = "primary_L8_20", "early_L3_8"

# ---------------- Claim 6 ----------------
print("== Claim 6 ==")
c8 = load_jsonl("block08_clamp/clamp.jsonl"); print("block08 rows", len(c8), check_items(c8, "block08"))
c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl"); print("block09 rows", len(c9), check_items(c9, "block09"))
lad = load_jsonl("block10_ladder/ladder.jsonl"); print("block10 ladder rows", len(lad), check_items(lad, "ladder"))
sub = load_jsonl("block10_ladder/subspace_patch.jsonl"); print("block10 subspace rows", len(sub))
pat = load_jsonl("block07_causal/patch.jsonl"); print("block07 patch rows", len(pat))
for band in (P, E):
    for lens in "JR":
        for arm in ("clamp", "swap", "install"):
            r = sel(c8, band=band, lens=lens, arm=arm)
            print(f"  block08 {band} {lens} {arm}: flip {rate(r)} ({sum(bool(x['top1_is_swap']) for x in r)}/{len(r)}) meanΔM {mean(r,'delta_margin'):+.3f} medΔM {med(r,'delta_margin'):+.3f} med energy {med(r,'total_dh2'):.3g}")
a = {r["index"]: bool(r["top1_is_swap"]) for r in sel(c8, band=P, lens="J", arm="clamp")}
b = {r["index"]: bool(r["top1_is_swap"]) for r in sel(c9, band=P, lens="J", pair="entity", arm="full")}
print(f"  block09 entity/full J primary flip {sum(b.values())}/{len(b)} = {np.mean(list(b.values())):.4f}; per-item identical to block08: {a==b}; n disagreements {sum(a[i]!=b[i] for i in a)}")
da = {r["index"]: r["delta_margin"] for r in sel(c8, band=P, lens="J", arm="clamp")}
db = {r["index"]: r["delta_margin"] for r in sel(c9, band=P, lens="J", pair="entity", arm="full")}
print(f"  max |ΔM diff| block08 vs block09 J clamp: {max(abs(da[i]-db[i]) for i in da):.4f}")
for band in (P, E):
    for lens in "JR":
        for pair in ("entity", "answer_pair", "shuffled"):
            for arm in ("full", "ortho_rescaled"):
                r = sel(c9, band=band, lens=lens, pair=pair, arm=arm)
                print(f"  block09 {band} {lens} {pair:11s} {arm:14s}: flip {rate(r)[0]:.3f} (n={len(r)}) meanΔM {mean(r,'delta_margin'):+.3f} medΔM {med(r,'delta_margin'):+.3f}")
# early band R-J clamp
for arm in ("full", "ortho_rescaled"):
    d = paired(per_item(sel(c9, band=E, lens="R", pair="entity", arm=arm)), per_item(sel(c9, band=E, lens="J", pair="entity", arm=arm)))
    print(f"  early band R-J entity clamp {arm}:", fmt(d))
    d = paired(per_item(sel(c9, band=P, lens="R", pair="entity", arm=arm)), per_item(sel(c9, band=P, lens="J", pair="entity", arm=arm)))
    print(f"  primary band R-J entity clamp {arm}:", fmt(d))
# ladder
for lens in ("J", "R", "logit"):
    for sc in (0.5, 1.0, 1.5, 2.0, 3.0):
        r = sel(lad, lens=lens, arm="clamp", scale=sc)
        print(f"  ladder {lens:5s} clamp scale {sc}: flip {rate(r)[0]:.3f} (n={len(r)}) meanΔM {mean(r,'delta_margin'):+.3f} medΔM {med(r,'delta_margin'):+.3f} med energy {med(r,'total_dh2'):.4g}")
    r = sel(lad, lens=lens, arm="involution"); print(f"  ladder {lens:5s} involution: flip {rate(r)[0]:.3f} meanΔM {mean(r,'delta_margin'):+.3f} med energy {med(r,'total_dh2'):.4g}")
d = paired(per_item(sel(lad, lens="J", arm="clamp", scale=1.0)), per_item(sel(lad, lens="logit", arm="clamp", scale=1.0))); print("  J clamp - logit clamp (scale 1):", fmt(d))
# consistency block10 J clamp scale1 vs block08 J clamp
d = paired(per_item(sel(lad, lens="J", arm="clamp", scale=1.0)), per_item(sel(c8, band=P, lens="J", arm="clamp"))); print("  block10 J clamp s=1 - block08 J clamp:", fmt(d))
for lens in ("J", "R", "logit"):
    for arm in ("clamp_exchange", "clamp_counterfactual"):
        r = sel(sub, lens=lens, arm=arm); print(f"  subspace {lens:5s} {arm:20s}: flip {rate(r)} medΔM {med(r,'delta_margin'):+.2f}")
print("  block07 patch arms by layer (J lens rows):")
for arm in sorted({r["arm"] for r in pat}):
    line = []
    for L in sorted({r["layer"] for r in pat}):
        r = sel(pat, arm=arm, layer=L, lens="J"); line.append(f"L{L}:{rate(r)[0]:.2f}")
    r = sel(pat, arm=arm, lens="J"); print(f"    {arm:18s} pooled flip {rate(r)[0]:.2f} (n={len(r)}) | " + " ".join(line))
r = sel(pat, arm="patch_full"); print(f"    patch_full pooled J+R flip {rate(r)} ; identical across lens? J vs R per-item: ", {L: (sorted((x['index'],x['top1_is_swap']) for x in sel(pat,arm='patch_full',layer=L,lens='J'))==sorted((x['index'],x['top1_is_swap']) for x in sel(pat,arm='patch_full',layer=L,lens='R'))) for L in sorted({x['layer'] for x in pat})})

# ---------------- Claim 7 ----------------
print("\n== Claim 7 ==")
m3 = load_jsonl("block11_mechanism/m3_parity.jsonl"); print("m3 rows", len(m3), check_items(m3, "m3"))
for arm in ("involution", "clamp"):
    line = []
    for w in range(1, 14):
        r = sel(m3, lens="J", arm=arm, width=w); line.append(f"{rate(r)[0]:.2f}")
    print(f"  J {arm:10s} flip by width 1..13: " + " ".join(line))
    line = []
    for w in range(1, 14):
        r = sel(m3, lens="J", arm=arm, width=w); line.append(f"{mean(r,'delta_margin'):+.2f}")
    print(f"  J {arm:10s} meanΔM by width:      " + " ".join(line))
    for lens in "JR":
        vals = {}
        for i in ELIGIBLE:
            odd = [x["delta_margin"] for x in m3 if x["index"] == i and x["lens"] == lens and x["arm"] == arm and x["width"] % 2 == 1]
            even = [x["delta_margin"] for x in m3 if x["index"] == i and x["lens"] == lens and x["arm"] == arm and x["width"] % 2 == 0]
            vals[i] = float(np.mean(odd) - np.mean(even))
        m, lo, hi, n, nc = cboot(vals)
        # alternative: adjacent pairs (w odd) - (w+1 even) averaged over w=1..12
        vals2 = {}
        for i in ELIGIBLE:
            byw = {x["width"]: x["delta_margin"] for x in m3 if x["index"] == i and x["lens"] == lens and x["arm"] == arm}
            vals2[i] = float(np.mean([byw[w] - byw[w + 1] for w in range(1, 13, 2)]))
        m2, lo2, hi2, _, _ = cboot(vals2)
        print(f"  {lens} {arm:10s} odd-even (mean over odd widths - mean over even widths): {m:+.3f} [{lo:+.3f},{hi:+.3f}] | adjacent-pair version {m2:+.3f} [{lo2:+.3f},{hi2:+.3f}]")
# check m3 width13 clamp J equals block08 clamp
d = paired(per_item(sel(m3, lens="J", arm="clamp", width=13)), per_item(sel(c8, band=P, lens="J", arm="clamp"))); print("  m3 width13 J clamp - block08 J clamp:", fmt(d))
d = paired(per_item(sel(m3, lens="J", arm="involution", width=13)), per_item(sel(c8, band=P, lens="J", arm="swap"))); print("  m3 width13 J involution - block08 J swap:", fmt(d))

# ---------------- Claim 8 ----------------
print("\n== Claim 8 ==")
sup = load_jsonl("block12_suppression/suppression.jsonl"); print("block12 rows", len(sup), check_items(sup, "block12"))
supr = load_jsonl("block12_suppression/suppression_readout.jsonl"); print("block12 readout rows", len(supr))
for lens in "JR":
    for arm in ("install", "remove", "clamp", "clamp_plus_ablate", "clamp_plus_random"):
        r = sel(sup, lens=lens, arm=arm)
        print(f"  {lens} {arm:18s}: meanΔM {mean(r,'delta_margin'):+.3f} medΔM {med(r,'delta_margin'):+.3f} flip {rate(r)[0]:.3f} ({sum(bool(x['top1_is_swap']) for x in r)}/{len(r)}) med energy {med(r,'total_dh2'):.3g}")
    syn = {}
    ci = per_item(sel(sup, lens=lens, arm="clamp")); ii = per_item(sel(sup, lens=lens, arm="install")); rr = per_item(sel(sup, lens=lens, arm="remove"))
    for i in ci: syn[i] = ci[i] - (ii[i] + rr[i])
    m, lo, hi, n, nc = cboot(syn); print(f"  {lens} synergy clamp-(install+remove): {m:+.3f} [{lo:+.3f},{hi:+.3f}]; install+remove mean {np.mean([ii[i]+rr[i] for i in ci]):+.3f}")
    d = paired(per_item(sel(sup, lens=lens, arm="clamp_plus_ablate")), per_item(sel(sup, lens=lens, arm="clamp"))); print(f"  {lens} clamp_plus_ablate - clamp:", fmt(d))
# readout ranks by arm (reader R on J clamp)
for arm in ("install", "remove", "clamp", "clamp_plus_ablate", "clamp_plus_random"):
    r = sel(supr, arm=arm)
    bi = [min(x["best_intermediate"]) for x in r]; bs = [min(x["best_swap_to"]) for x in r]
    print(f"  readout {arm:18s}: median best-rank intermediate {np.median(bi):.0f}, swap_to {np.median(bs):.0f} (n={len(r)})")
# threshold analysis: required = clean logp(answer) - logp(swap_answer)  (block02 clean rows)
rec = load_jsonl("block02/block02_records.jsonl")
clean = {r["index"]: r for r in sel(rec, condition="clean")}
req = {i: clean[i]["logp_answer"] - clean[i]["logp_swap_answer"] for i in ELIGIBLE}
print(f"  required threshold (logp ans - logp swap at clean): median {np.median(list(req.values())):.3f} mean {np.mean(list(req.values())):.3f}")
# alt: threshold vs top1 logp (same since clean-correct)
req2 = {i: clean[i]["topk"][0][1] - clean[i]["logp_swap_answer"] for i in ELIGIBLE}
print(f"  required vs top1 logp: median {np.median(list(req2.values())):.3f}; all clean top1==answer? {all(clean[i]['top1_is_answer'] for i in ELIGIBLE)}")
for arm in ("install", "remove", "clamp"):
    dm = per_item(sel(sup, lens="J", arm=arm))
    frac = np.mean([dm[i] >= req[i] for i in ELIGIBLE]); frac2 = np.mean([dm[i] > req[i] for i in ELIGIBLE])
    print(f"  J {arm:8s}: fraction ΔM >= threshold {frac:.3f} (strict > {frac2:.3f}); actual flip {rate(sel(sup, lens='J', arm=arm))[0]:.3f}")
flips = {r["index"]: bool(r["top1_is_swap"]) for r in sel(sup, lens="J", arm="clamp")}
dm = per_item(sel(sup, lens="J", arm="clamp"))
print(f"  clamp J: median ΔM flipped {np.median([dm[i] for i in ELIGIBLE if flips[i]]):.2f} vs not {np.median([dm[i] for i in ELIGIBLE if not flips[i]]):.2f}; median required flipped {np.median([req[i] for i in ELIGIBLE if flips[i]]):.2f} vs not {np.median([req[i] for i in ELIGIBLE if not flips[i]]):.2f}")

# ---------------- Claim 9 ----------------
print("\n== Claim 9 ==")
geo = load_jsonl("block05_answer_swap/geometry.jsonl"); print("geometry rows", len(geo), check_items(geo, "geometry"))
cos = {r["index"]: r["cos_entitydiff_answerdiff"] for r in geo}
for src, dmap, label in ((sup, per_item(sel(sup, lens="J", arm="clamp")), "block12 J clamp"), (c8, per_item(sel(c8, band=P, lens="J", arm="clamp")), "block08 J clamp")):
    xs = [cos[i] for i in ELIGIBLE]; ys = [dmap[i] for i in ELIGIBLE]
    print(f"  Spearman cos_entitydiff_answerdiff -> ΔM ({label}): {spearman(xs, ys):+.3f}; |cos| version {spearman([abs(x) for x in xs], ys):+.3f}")
    fl = [float(flips[i]) for i in ELIGIBLE] if label.startswith("block12") else [float(bool(r['top1_is_swap'])) for r in sorted(sel(c8, band=P, lens='J', arm='clamp'), key=lambda r: r['index'])]
    print(f"  Spearman ΔM -> flip ({label}): {spearman(ys, fl):+.3f}")
    print(f"  Spearman cos -> flip ({label}): {spearman(xs, fl):+.3f}")
print(f"  Spearman required threshold -> flip: {spearman([req[i] for i in ELIGIBLE], [float(flips[i]) for i in ELIGIBLE]):+.3f}")
print(f"  cos_entitydiff_answerdiff median flipped {np.median([cos[i] for i in ELIGIBLE if flips[i]]):.3f} vs not {np.median([cos[i] for i in ELIGIBLE if not flips[i]]):.3f}")
cr = load_jsonl("block04/cross_readout.jsonl")
for reader in "JR":
    rows = sel(cr, arm="clean", reader=reader)
    rk = {r["index"]: r["best_rank_intermediate"] for r in rows}
    for lens in "JR":
        fl = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens=lens, arm="clamp")}
        print(f"  Spearman clean readout rank (reader {reader}) -> flip ({lens} clamp): {spearman([rk[i] for i in ELIGIBLE], [fl[i] for i in ELIGIBLE]):+.3f}  | log10 rank: {spearman([np.log10(rk[i]) for i in ELIGIBLE], [fl[i] for i in ELIGIBLE]):+.3f}")
    print(f"    median clean best rank intermediate reader {reader}: {np.median([rk[i] for i in ELIGIBLE]):.0f}")
