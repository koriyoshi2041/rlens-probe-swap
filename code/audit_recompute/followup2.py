import sys, json, collections, itertools
sys.path.insert(0, ".")
from audit_lib import *
import numpy as np
P = "primary_L8_20"
c8 = load_jsonl("block08_clamp/clamp.jsonl"); sup = load_jsonl("block12_suppression/suppression.jsonl")
flJ = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens="J", arm="clamp")}
flR = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens="R", arm="clamp")}
dmJ = per_item(sel(sup, lens="J", arm="clamp")); dmR = per_item(sel(sup, lens="R", arm="clamp"))

def kendall(x, y):
    x = np.asarray(x); y = np.asarray(y); c = d = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        sx = np.sign(x[i]-x[j]); sy = np.sign(y[i]-y[j])
        if sx == 0 and sy == 0: continue
        if sx == 0: tx += 1
        elif sy == 0: ty += 1
        elif sx == sy: c += 1
        else: d += 1
    return (c - d) / np.sqrt((c+d+tx)*(c+d+ty))
print("Kendall tau-b (J clamp ΔM, flip):", round(kendall([dmJ[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]),3))
reach = json.loads((RES/"block02"/"knowledge_reachable_indices.json").read_text())
print("Spearman on reachable n=44:", round(spearman([dmJ[i] for i in reach],[flJ[i] for i in reach]),3))
print("Spearman(R ΔM, J flip):", round(spearman([dmR[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]),3), " (J ΔM, R flip):", round(spearman([dmJ[i] for i in ELIGIBLE],[flR[i] for i in ELIGIBLE]),3))
# point-biserial on log?
print("Spearman(log1p(max(ΔM,0)), flip):", round(spearman([np.log1p(max(dmJ[i],0)) for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]),3))
# delta_margin -> flip using ladder scale 0.5 low dose ΔM?
lad = load_jsonl("block10_ladder/ladder.jsonl")
for sc in (0.5, 1.5, 2.0, 3.0):
    d = per_item(sel(lad, lens="J", arm="clamp", scale=sc)); f = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(lad, lens="J", arm="clamp", scale=sc)}
    print(f"  ladder scale {sc}: Spearman(ΔM_s, flip_1) {spearman([d[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}; Spearman(ΔM_s, flip_s) {spearman([d[i] for i in ELIGIBLE],[f[i] for i in ELIGIBLE]):+.3f}")
# cos -> flip, threshold -> flip already; suppression readout rank -> flip?
supr = load_jsonl("block12_suppression/suppression_readout.jsonl")
for arm in ("clamp",):
    bi = {r["index"]: min(r["best_intermediate"]) for r in sel(supr, arm=arm)}; bs = {r["index"]: min(r["best_swap_to"]) for r in sel(supr, arm=arm)}
    print(f"  post-clamp readout: Spearman(intermediate rank, flip) {spearman([bi[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}; Spearman(swap_to rank, flip) {spearman([bs[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}; Spearman(log10 interm rank, ΔM) {spearman([np.log10(bi[i]) for i in ELIGIBLE],[dmJ[i] for i in ELIGIBLE]):+.3f}")
    print(f"  median post-clamp intermediate rank flipped {np.median([bi[i] for i in ELIGIBLE if flJ[i]]):.0f} vs not {np.median([bi[i] for i in ELIGIBLE if not flJ[i]]):.0f}; swap_to {np.median([bs[i] for i in ELIGIBLE if flJ[i]]):.0f} vs {np.median([bs[i] for i in ELIGIBLE if not flJ[i]]):.0f}; top10 frac {np.mean([bs[i]<=10 for i in ELIGIBLE if flJ[i]]):.2f} vs {np.mean([bs[i]<=10 for i in ELIGIBLE if not flJ[i]]):.2f}")
    geo = load_jsonl("block05_answer_swap/geometry.jsonl"); cos = {r["index"]: r["cos_entitydiff_answerdiff"] for r in geo}
    print(f"  Spearman(log10 post-clamp interm rank, cos) {spearman([np.log10(bi[i]) for i in ELIGIBLE],[cos[i] for i in ELIGIBLE]):+.3f}")

print("\n-- readout sources for clean intermediate rank --")
pers = load_jsonl("block06_persistence/persistence.jsonl")
print("persistence sample:", {k: (v if not isinstance(v, list) else v[:6]) for k, v in pers[0].items()})
for reader in "JR":
    rows = sel(pers, arm="clean", reader=reader)
    for agg, fn in (("min", min), ("median", np.median)):
        rk = {r["index"]: fn(r["best_intermediate"]) for r in rows}
        print(f"  persistence reader {reader} clean interm rank ({agg}): -> J flip {spearman([rk[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f} | R flip {spearman([rk[i] for i in ELIGIBLE],[flR[i] for i in ELIGIBLE]):+.3f}; layers {rows[0]['layers'][:3]}..{rows[0]['layers'][-1]}")
    # band-restricted (8..20)
    rows = sel(pers, arm="clean", reader=reader)
    L = rows[0]["layers"]; band_idx = [k for k, l in enumerate(L) if 8 <= l <= 20]
    rk = {r["index"]: min(r["best_intermediate"][k] for k in band_idx) for r in rows}
    hit = {r["index"]: float(np.mean([r["best_intermediate"][k] <= 10 for k in band_idx])) for r in rows}
    print(f"  persistence reader {reader} band-min rank -> J flip {spearman([rk[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f} | R flip {spearman([rk[i] for i in ELIGIBLE],[flR[i] for i in ELIGIBLE]):+.3f}; top10 hit-rate -> J flip {spearman([hit[i] for i in ELIGIBLE],[flJ[i] for i in ELIGIBLE]):+.3f}; mean hit flipped {np.mean([hit[i] for i in ELIGIBLE if flJ[i]]):.2f} vs not {np.mean([hit[i] for i in ELIGIBLE if not flJ[i]]):.2f}")
npz = np.load(RES/"block01"/"readout_ranks.npz", allow_pickle=True)
print("readout_ranks.npz keys:", list(npz.keys()), {k: npz[k].shape for k in npz.keys()})
rs = json.load(open(RES/"block01"/"readout_summary.json")); print("readout_summary keys:", list(rs.keys())[:10])
cr = load_jsonl("block08_clamp/clamp_readout.jsonl")
print("clamp_readout sample:", {k: (v if not isinstance(v, list) else v[:6]) for k, v in cr[0].items()})
