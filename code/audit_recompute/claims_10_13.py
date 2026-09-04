import sys, json
sys.path.insert(0, __import__('os').path.dirname(__file__))
from audit_lib import *
import numpy as np

# ---------------- Claim 10 ----------------
print("== Claim 10 ==")
lad = load_jsonl("block10_ladder/ladder.jsonl")
flips1 = {r["index"]: bool(r["top1_is_swap"]) for r in sel(lad, lens="J", arm="clamp", scale=1.0)}
print("E1 dose curve (J clamp):")
for sc in (0.5, 1.0, 1.5, 2.0, 3.0):
    r = sel(lad, lens="J", arm="clamp", scale=sc)
    dm = per_item(r)
    fl_group = [dm[i] for i in ELIGIBLE if flips1[i]]; nf_group = [dm[i] for i in ELIGIBLE if not flips1[i]]
    print(f"  scale {sc}: meanΔM {mean(r,'delta_margin'):+.3f} medΔM {med(r,'delta_margin'):+.3f} flip {rate(r)[0]:.3f} | mean flippable {np.mean(fl_group):+.3f} zero-flip {np.mean(nf_group):+.3f} | med energy {med(r,'total_dh2'):.4g}")
base_f = np.mean([per_item(sel(lad, lens='J', arm='clamp', scale=0.5))[i] for i in ELIGIBLE if flips1[i]])
base_n = np.mean([per_item(sel(lad, lens='J', arm='clamp', scale=0.5))[i] for i in ELIGIBLE if not flips1[i]])
for sc in (0.5, 1.0, 1.5, 2.0, 3.0):
    dm = per_item(sel(lad, lens="J", arm="clamp", scale=sc))
    print(f"  normalised to scale 0.5: flippable {np.mean([dm[i] for i in ELIGIBLE if flips1[i]])/base_f:.2f}, zero-flip {np.mean([dm[i] for i in ELIGIBLE if not flips1[i]])/base_n:.2f}")
r = sel(lad, lens="logit", arm="clamp", scale=3.0); print(f"  logit clamp scale 3: flip {rate(r)[0]:.3f} med energy {med(r,'total_dh2'):.4g}; J scale1 med energy {med(sel(lad, lens='J', arm='clamp', scale=1.0),'total_dh2'):.4g}; J scale 3 med energy {med(sel(lad, lens='J', arm='clamp', scale=3.0),'total_dh2'):.4g}")
e2 = load_jsonl("block13_deep/e2_positions.jsonl"); print("E2 rows", len(e2), check_items(e2, "e2"))
print("E2 per-position J clamp, median ΔM / flip by offset_from_end:")
for off in (-5, -4, -3, -2, -1, 0):
    r = sel(e2, lens="J", offset_from_end=off); print(f"  offset {off:3d}: n={len(r)} medΔM {med(r,'delta_margin'):+.3f} meanΔM {mean(r,'delta_margin'):+.3f} flip {rate(r)[0]:.3f}")
r = [x for x in e2 if x["lens"] == "J" and x["offset_from_end"] <= -6]; print(f"  offset <= -6: n={len(r)} medΔM {med(r,'delta_margin'):+.3f} flip {rate(r)[0]:.3f}")
sums = {}
for x in e2:
    if x["lens"] == "J":
        sums[x["index"]] = sums.get(x["index"], 0.0) + x["delta_margin"]
print(f"  sum of single-position ΔM per item: median {np.median(list(sums.values())):.3f} mean {np.mean(list(sums.values())):.3f}")
sup = load_jsonl("block12_suppression/suppression.jsonl")
print(f"  full clamp J (block12): meanΔM {mean(sel(sup, lens='J', arm='clamp'),'delta_margin'):+.3f} medΔM {med(sel(sup, lens='J', arm='clamp'),'delta_margin'):+.3f}")
e5 = load_jsonl("block13_deep/e5_mix.jsonl"); print("E5 rows", len(e5), check_items(e5, "e5"))
for lens in "JR":
    line = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = sel(e5, lens=lens, mix=t); line.append(f"mix {t}: flip {rate(r)[0]:.3f} meanΔM {mean(r,'delta_margin'):+.2f} E {med(r,'total_dh2'):.3g}")
    print(f"  E5 {lens}: " + " | ".join(line))
e4 = json.load(open(RES / "block13_deep" / "e4_gaps.json")); print("E4 rows", len(e4), "items", len({r['index'] for r in e4}))
print("E4 coordinate gap: median over items of median_abs_gap, J / R / ratio(R/J of medians) / median of per-item ratio:")
for L in (3, 4, 5, 6, 7, 8, 10, 12, 16, 20):
    gj = {r["index"]: r["median_abs_gap"] for r in e4 if r["lens"] == "J" and r["layer"] == L}
    gr = {r["index"]: r["median_abs_gap"] for r in e4 if r["lens"] == "R" and r["layer"] == L}
    mj, mr = np.median(list(gj.values())), np.median(list(gr.values()))
    pr = np.median([gr[i] / gj[i] for i in gj])
    mnj, mnr = np.mean(list(gj.values())), np.mean(list(gr.values()))
    print(f"  L{L:2d}: J {mj:.3f} R {mr:.3f} ratio {mr/mj:.2f} | median per-item ratio {pr:.2f} | means J {mnj:.3f} R {mnr:.3f} ratio {mnr/mnj:.2f}")

# ---------------- Claim 11 ----------------
print("\n== Claim 11 ==")
le = load_jsonl("block19_ladder_eval/ladder_eval.jsonl"); print("ladder_eval rows", len(le), check_items(le, "ladder_eval"))
geo = json.load(open(RES / "block19_ladder_eval" / "geometry.json"))
for lens in ("ours_n25", "ours_n50", "ours_n100", "ours_n200", "ours_n400", "published_J", "published_R"):
    c = sel(le, lens=lens, arm="clamp"); v = sel(le, lens=lens, arm="involution")
    m, lo, hi, n, nc = cboot(per_item(c))
    g = [x for x in geo if x["rung"] == lens]
    cosm = np.mean([x["cos_to_published"] for x in g]) if g else float("nan")
    cost = np.mean([x["cos_to_top_rung"] for x in g]) if g else float("nan")
    nr = [x["norm_ratio_to_published"] for x in g]
    print(f"  {lens:12s}: clamp ΔM {m:+.2f} [{lo:+.2f},{hi:+.2f}] flip {rate(c)[0]:.3f} | involution flip {rate(v)[0]:.3f} ΔM {mean(v,'delta_margin'):+.2f} | cos_pub {cosm:.3f} cos_top {cost:.3f} norm ratio {min(nr) if nr else float('nan'):.2f}-{max(nr) if nr else float('nan'):.2f} | med energy clamp {med(c,'total_dh2'):.3g}")
d = paired(per_item(sel(le, lens="ours_n400", arm="clamp")), per_item(sel(le, lens="ours_n25", arm="clamp"))); print("  n400 - n25 clamp:", fmt(d))
d = paired(per_item(sel(le, lens="ours_n25", arm="clamp")), per_item(sel(le, lens="published_J", arm="clamp"))); print("  ours_n25 - published_J clamp:", fmt(d))
d = paired(per_item(sel(le, lens="ours_n400", arm="clamp")), per_item(sel(le, lens="published_J", arm="clamp"))); print("  ours_n400 - published_J clamp:", fmt(d))
d = paired(per_item(sel(le, lens="published_J", arm="clamp")), per_item(sel(load_jsonl("block08_clamp/clamp.jsonl"), band="primary_L8_20", lens="J", arm="clamp"))); print("  block19 published_J clamp - block08 J clamp:", fmt(d))

# ---------------- Claim 12 ----------------
print("\n== Claim 12 ==")
dmp = load_jsonl("block16_damping/damping.jsonl"); print("damping rows", len(dmp), check_items(dmp, "damping"))
for lens in "JR":
    for sc in (0.01, 0.03, 0.1, 0.25, 0.5, 1.0):
        r = sel(dmp, lens=lens, scale=sc)
        ratios = [x["ratio"] for x in r if x["ratio"] is not None]
        ra = np.sum([x["actual"] for x in r]) / np.sum([x["predicted_linear"] for x in r])
        print(f"  {lens} scale {sc}: median ratio {np.median(ratios):.3f} mean ratio {np.mean(ratios):.3f} sum(actual)/sum(pred) {ra:.3f} | med actual {med(r,'actual'):+.3f} med pred {med(r,'predicted_linear'):+.3f} | med max_rel {med(r,'max_rel_perturbation'):.4f} | flip {rate(r)[0]:.3f} | n ratio None {sum(x['ratio'] is None for x in r)}")
dc = load_jsonl("block24_direction_control/direction_control.jsonl"); print("direction_control rows", len(dc), "items", len({r['index'] for r in dc}))
for direction in ("clamp", "contrast", "gradient"):
    for tgt in (0.5, 1.0, 2.0, 5.0, 10.0):
        r = sel(dc, direction=direction, predicted=tgt)
        print(f"  {direction:9s} pred {tgt:4}: n={len(r)} median ratio_one_sided {med(r,'ratio_one_sided'):.3f} mean {mean(r,'ratio_one_sided'):.3f} | median ratio_central {med(r,'ratio_central'):.3f} mean {mean(r,'ratio_central'):.3f} | med rel pert {med(r,'median_rel_perturbation'):.4g}")

# ---------------- Claim 13 ----------------
print("\n== Claim 13 ==")
pr = load_jsonl("block18_rederivation/position_repair.jsonl"); print("position_repair rows", len(pr), check_items(pr, "pos_repair"))
full = sel(pr, spared=None); print(f"  full (spared=None): n={len(full)} medΔM {med(full,'delta_margin'):+.3f} meanΔM {mean(full,'delta_margin'):+.3f} flip {rate(full)[0]:.3f}")
fm = per_item(full)
for off in (-5, -4, -3, -2, -1, 0):
    r = sel(pr, spared_offset=off); dm = per_item(r)
    loss = np.median([fm[i] - dm[i] for i in dm])
    print(f"  spared offset {off:3d}: n={len(r)} medΔM {med(r,'delta_margin'):+.3f} meanΔM {mean(r,'delta_margin'):+.3f} flip {rate(r)[0]:.3f} | median loss vs full {loss:+.3f} | med(full)-med(spared) {np.median(list(fm.values()))-med(r,'delta_margin'):+.3f}")
dr = load_jsonl("block18_rederivation/depth_recovery.jsonl"); print("depth_recovery rows", len(dr), check_items(dr, "depth"))
for lens in "JR":
    for arm in ("partial_8_12", "full_8_20"):
        rows = sel(dr, lens=lens, arm=arm)
        prof = {}
        for x in rows:
            for p in x["profile"]:
                prof.setdefault(p["layer"], []).append(p["swapped_fraction"])
        print(f"  {lens} {arm:12s} median swapped_fraction: " + " ".join(f"L{L}:{np.median(v):.2f}" for L, v in sorted(prof.items())))
        print(f"       flip {rate(rows)[0]:.3f} meanΔM {mean(rows,'delta_margin'):+.3f}  | mean swapped_fraction: " + " ".join(f"L{L}:{np.mean(v):.2f}" for L, v in sorted(prof.items()) if L in (13,15,17,19,21,24)))
    # by flip group
    rows = sel(dr, lens=lens, arm="partial_8_12")
    fl = {r["index"]: bool(r["top1_is_swap"]) for r in sel(dr, lens=lens, arm="full_8_20")}
    for g, name in ((True, "flippable"), (False, "zero-flip")):
        prof = {}
        for x in rows:
            if fl[x["index"]] == g:
                for p in x["profile"]:
                    prof.setdefault(p["layer"], []).append(p["swapped_fraction"])
        print(f"       {name:10s} L20 median {np.median(prof[20]):.2f} L19 {np.median(prof[19]):.2f} (n={len(prof[20])})")
s3 = load_jsonl("block20_specificity/s3_repair.jsonl"); print("s3 rows", len(s3), check_items(s3, "s3"))
for lens in "JR":
    rows = sel(s3, lens=lens)
    prof = {}
    for x in rows:
        for p in x["profile"]:
            prof.setdefault(p["layer"], []).append((p["attn_push_on_source"], p["mlp_push_on_source"]))
    print(f"  {lens} median attn/mlp push on source: " + " | ".join(f"L{L}: attn {np.median([a for a,_ in v]):+.3f} mlp {np.median([m for _,m in v]):+.3f}" for L, v in sorted(prof.items())))
    print(f"  {lens} mean   attn/mlp push on source: " + " | ".join(f"L{L}: attn {np.mean([a for a,_ in v]):+.3f} mlp {np.mean([m for _,m in v]):+.3f}" for L, v in sorted(prof.items())))
