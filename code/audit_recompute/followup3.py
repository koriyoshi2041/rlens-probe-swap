import sys, json, collections
sys.path.insert(0, ".")
from audit_lib import *
import numpy as np
P = "primary_L8_20"
c8 = load_jsonl("block08_clamp/clamp.jsonl")
flJ = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens="J", arm="clamp")}
flR = {r["index"]: float(bool(r["top1_is_swap"])) for r in sel(c8, band=P, lens="R", arm="clamp")}
pers = load_jsonl("block06_persistence/persistence.jsonl")
print("persistence arm/reader counts:", collections.Counter((r["arm"], r["reader"]) for r in pers))
clean = sel(pers, arm="clean")
L = clean[0]["layers"]; print("layers", L[0], "..", L[-1], len(L))
band_idx = [k for k, l in enumerate(L) if 8 <= l <= 20]
for reader in "JR":
    rows = sel(clean, reader=reader); idx = [r["index"] for r in rows]
    fl = flJ if reader == "J" else flR
    for agg, fn in (("min all layers", lambda r: min(r["best_intermediate"])), ("min band 8-20", lambda r: min(r["best_intermediate"][k] for k in band_idx)), ("median band", lambda r: np.median([r["best_intermediate"][k] for k in band_idx])), ("hit-rate top10 band", lambda r: np.mean([r["best_intermediate"][k] <= 10 for k in band_idx]))):
        rk = {r["index"]: fn(r) for r in rows}
        print(f"  persistence clean reader {reader} n={len(rows)} {agg:22s}: -> {reader} flip {spearman([rk[i] for i in idx],[fl[i] for i in idx]):+.3f} | J flip {spearman([rk[i] for i in idx],[flJ[i] for i in idx]):+.3f} | R flip {spearman([rk[i] for i in idx],[flR[i] for i in idx]):+.3f}")
npz = np.load(RES/"block01"/"readout_ranks.npz", allow_pickle=True)
print("readout_ranks.npz:", {k: npz[k].shape for k in npz.keys()})
for k in npz.keys():
    if npz[k].dtype.kind in "OU" or npz[k].size < 100: print("  ", k, npz[k][:80] if npz[k].ndim == 1 else npz[k].shape)
rs = json.load(open(RES/"block01"/"readout_summary.json")); print("readout_summary keys:", list(rs.keys())[:10]); print(json.dumps(rs, ensure_ascii=False)[:600])
cr = load_jsonl("block08_clamp/clamp_readout.jsonl"); print("clamp_readout sample:", {k: (v if not isinstance(v, list) else v[:6]) for k, v in cr[0].items()})
m2 = load_jsonl("block11_mechanism/m2_early_state.jsonl"); print("m2 sample:", {k: (v if not isinstance(v, list) else v[:6]) for k, v in m2[0].items()})
