import sys, json, glob, math, collections
sys.path.insert(0, __import__('os').path.dirname(__file__))
from audit_lib import *
import numpy as np

# ---------------- Claim 14 ----------------
print("== Claim 14 ==")
s1 = load_jsonl("block20_specificity/s1_scale_matched.jsonl"); print("s1 rows", len(s1), "items", sorted({r['index'] for r in s1}))
for arm in ("true_patch", "other_item_patch_matched", "random_matched"):
    r = sel(s1, arm=arm)
    per = " ".join(f"L{L}:{rate(sel(s1, arm=arm, layer=L))[0]:.2f}/{len(sel(s1, arm=arm, layer=L))}" for L in (8, 12, 16))
    neither = 1 - rate(r)[0] - rate(r, "top1_is_answer")[0]
    print(f"  {arm:26s}: n={len(r)} flip {rate(r)[0]:.3f} still-answer {rate(r,'top1_is_answer')[0]:.3f} neither {neither:.3f} medΔM {med(r,'delta_margin'):+.3f} medKL {med(r,'kl'):.3f} | per layer {per}")
# ---------------- Claim 15 ----------------
print("\n== Claim 15 ==")
mc = json.load(open(RES / "block23_semantic" / "manual_classification.json"))
print("  keys:", list(mc.keys()))
for k, v in mc.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} entries")
    elif isinstance(v, dict) and k == "entity_probe_dissociation":
        for kk, vv in v.items():
            if isinstance(vv, list): print(f"  {k}.{kk}: {len(vv)}")
sem = json.load(open(RES / "block23_semantic" / "semantic.json")); print("  semantic.json items:", len(sem), check_items(sem, "semantic"))
sup = load_jsonl("block12_suppression/suppression.jsonl")
strict = {r["name"]: bool(r["top1_is_swap"]) for r in sel(sup, lens="J", arm="clamp")}
print(f"  strict flips (block12 J clamp): {sum(strict.values())}/59")
sem_names = {e["name"] for e in mc["rewrite_semantic_only"]}
dam_names = {e["name"] for e in mc["damaged"]}
print(f"  semantic-only names strict-flip status: {[(n, strict.get(n)) for n in sem_names]}")
print(f"  damaged names strict-flip status: {[(n, strict.get(n)) for n in dam_names]}")
unchanged = 59 - sum(strict.values()) - len(sem_names) - len(dam_names)
print(f"  implied unchanged = 59 - 24 - 3 - 5 = {unchanged}")
# check the semantic.json first-token consistency with strict flips
mism = []
for rec in sem:
    cont = rec["arms"]["J"]["continuation"]
    sa = rec["swap_answer"]
    starts = cont.strip().startswith(sa)
    if starts != strict[rec["name"]]:
        mism.append((rec["name"], sa, cont[:40], strict[rec["name"]]))
print(f"  items where J continuation startswith(swap_answer) disagrees with strict flip: {len(mism)}")
for m in mism: print("    ", m)
# ---------------- Claim 16 ----------------
print("\n== Claim 16 ==")
cr = json.load(open(RES / "block01" / "clean_rows.json"))
by = {(r["index"], r["mode"]): r for r in cr}
elig, modes = [], collections.Counter()
for i in range(90):
    s = by[(i, "rstrip")]
    row = s if (s["answers_single"] and s["answers_distinct"]) else by[(i, "as_is")]
    if row["clean_correct"] and row["concept_feasible"] and row["answers_single"] and row["answers_distinct"]:
        elig.append(i); modes[row["mode"]] += 1
print(f"  recomputed eligible: n={len(elig)} equal to file: {elig == ELIGIBLE}; modes {dict(modes)}")
cats = collections.Counter(by[(i, "rstrip")]["category"] for i in elig)
print(f"  categories: {len(cats)}; multihop {cats.get('multihop')}")
print(f"  as_is clean-correct {sum(by[(i,'as_is')]['clean_correct'] for i in range(90))}, rstrip {sum(by[(i,'rstrip')]['clean_correct'] for i in range(90))}")
pairs_in = [p for p in REVERSE_PAIRS if p[0] in elig and p[1] in elig]; print(f"  reverse pairs inside eligible: {pairs_in} -> clusters {len(elig)-len(pairs_in)}")
kn = json.load(open(RES / "block02" / "knowledge.json"))
for side in ("intermediate", "swap_to"):
    r = [x for x in kn if x["side"] == side]
    print(f"  knowledge {side}: {sum(x['correct'] for x in r)}/{len(r)} correct; rank<=5 {sum(x['rank_expected']<=5 for x in r)}; items {len({x['index'] for x in r})}")
kr = json.loads((RES / "block02" / "knowledge_reachable_indices.json").read_text())
sw_ok = {x["index"] for x in kn if x["side"] == "swap_to" and x["correct"]}
print(f"  reachable indices n={len(kr)} equals swap_to-correct set: {set(kr)==sw_ok}")
kn4 = json.load(open(RES / "block04" / "knowledge.json"))
print(f"  block04/knowledge.json identical to block02/knowledge.json: {kn4==kn}")
elig_median = np.median([by[(i, by[(i,'rstrip')]['mode'] if False else ('rstrip' if (by[(i,'rstrip')]['answers_single'] and by[(i,'rstrip')]['answers_distinct']) else 'as_is'))]['p_answer'] for i in elig])
print(f"  eligible clean median p(answer): {elig_median:.3f}")

# ---------------- Oddities ----------------
print("\n== Oddities scan ==")
files = sorted(glob.glob(str(RES / "*" / "*.jsonl")))
for f in files:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rel = f.split("results/")[1]
    # duplicated full rows
    seen = collections.Counter(json.dumps(r, sort_keys=True) for r in rows)
    dups = sum(c - 1 for c in seen.values())
    # duplicate keys (all non-numeric fields)
    keyf = [k for k in rows[0] if k not in ("delta_margin", "delta_logp_answer", "delta_logp_swap_answer", "kl", "total_dh2", "dh_norm", "energy", "topk", "wall_ms", "logp_answer", "logp_swap_answer", "neither_mass", "kl_clean_to_hooked", "top1_id", "top1_str", "top1_is_swap", "top1_is_answer", "profile", "best_intermediate", "best_swap_to", "best_rank_intermediate", "best_rank_swap_to", "best_rank_answer", "best_rank_swap_answer", "final_intermediate", "final_swap_to", "final_answer", "final_swap_answer", "predicted_linear", "actual", "ratio", "grad_norm", "max_rel_perturbation", "median_cos_direction_contrast", "cos_direction_contrast", "cos_lens_diffmean", "frac_var_explained_by_lens_dir", "one_sided", "central", "curvature", "ratio_one_sided", "ratio_central", "median_rel_perturbation", "central_difference", "curvature_term", "achieved_rel", "predicted", "token", "layers", "clamped_layers", "cos_intermediate_answer", "cos_swapto_swapanswer", "cos_intermediate_swapto", "cos_answer_swapanswer", "cos_entitydiff_answerdiff")]
    keyc = collections.Counter(json.dumps({k: r.get(k) for k in keyf}, sort_keys=True) for r in rows)
    keydups = sum(c - 1 for c in keyc.values())
    # NaN / inf
    def walk(v):
        if isinstance(v, float):
            yield v
        elif isinstance(v, dict):
            for x in v.values(): yield from walk(x)
        elif isinstance(v, list):
            for x in v: yield from walk(x)
    nan = sum(1 for r in rows for v in walk(r) if isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    nones = collections.Counter(k for r in rows for k, v in r.items() if v is None)
    # energy blow-ups
    big = 0; bigmax = 0
    for r in rows:
        e = r.get("total_dh2", r.get("dh_norm", 0.0) ** 2 if r.get("dh_norm") is not None else 0.0)
        if e is not None and e > 1e6:
            big += 1; bigmax = max(bigmax, e)
    # bounds
    bad = 0
    for r in rows:
        for k in ("kl", "kl_clean_to_hooked"):
            if k in r and r[k] < -1e-6: bad += 1
        if "neither_mass" in r and not (-1e-6 <= r["neither_mass"] <= 1 + 1e-6): bad += 1
        if "top1_is_swap" in r and "top1_is_answer" in r and r["top1_is_swap"] and r["top1_is_answer"]: bad += 1
    nitems = len({r.get("index") for r in rows})
    print(f"  {rel:50s} rows={len(rows):5d} items={nitems:2d} exactdup={dups} keydup={keydups} nan/inf={nan} big_energy(>1e6)={big} (max {bigmax:.3g}) impossible={bad} None-fields={dict(nones) if nones else '-'}")
