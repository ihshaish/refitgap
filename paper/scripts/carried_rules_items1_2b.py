"""Carried rules, items 1 and 2b (plans/carried-rules.txt): for every DistilBERT comparison, the lead under
the carried rules a to d and f against the lead after both refit (e, and e500 for rule d), on the same scoring
halves, with each arm's levels; and, for replay minus sequential training on ASRS and MultiEURLEX, each label's
part of G_U, G_V and the change, the macro values with each label left out, and error counts for the two labels
with the largest part of the change. Backs Figure 3 and supplement Tables C.7 (item 1) and C.8 (item 2b).

    python scripts/carried_rules_items1_2b.py SCORES_DIR [--out DIR] [--workers N]

Writes item1.json and item2b.json, and the raw counts (raw/item1_<n>.pkl, for carried_rules_csvs.py).
Stored outputs: results/carried_rules/. About 10 minutes on 6 workers.
"""
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carried_rules_core as C                       # noqa: E402

CARRIED = ["theta1", "cut0.5", "own_val", "prev_period", "matched_rate"]


def summarise_item1(results):
    lv = {r: np.array([[C.level_halves(x, m, r) for m in (0, 1)] for x in results]) for r in C.RULES}
    lw = {r: np.array([[C.level_whole(x, m, r) for m in (0, 1)] for x in results]) for r in C.WHOLE_RULES}
    gv = lv["refit"][:, 0] - lv["refit"][:, 1]
    gv5 = lv["refit500"][:, 0] - lv["refit500"][:, 1]
    out = dict(seeds=[x["seed"] for x in results], arms=results[0]["arms"], n_pairs=len(results),
               G_V=C.t_summary(gv), G_V500=C.t_summary(gv5),
               levels_refit={results[0]["arms"][m]: float(lv["refit"][:, m].mean()) for m in (0, 1)},
               levels_refit500={results[0]["arms"][m]: float(lv["refit500"][:, m].mean()) for m in (0, 1)},
               levels_refit_per_pair={results[0]["arms"][m]: lv["refit"][:, m].tolist() for m in (0, 1)},
               rules={})
    for r in CARRIED:
        g = lv[r][:, 0] - lv[r][:, 1]
        e = dict(letter=C.RULE_LETTER[r], G_same_folds=C.t_summary(g), G_minus_G_V=C.t_summary(g - gv),
                 levels_same_folds={results[0]["arms"][m]: float(lv[r][:, m].mean()) for m in (0, 1)},
                 levels_same_folds_per_pair={results[0]["arms"][m]: lv[r][:, m].tolist() for m in (0, 1)})
        if r == "prev_period":
            e["G_minus_G_V500"] = C.t_summary(g - gv5)
        if r in C.WHOLE_RULES:
            gw = lw[r][:, 0] - lw[r][:, 1]
            e["G_whole_period"] = C.t_summary(gw)
            e["levels_whole_period"] = {results[0]["arms"][m]: float(lw[r][:, m].mean()) for m in (0, 1)}
        out["rules"][r] = e
    out["checks"] = dict(
        matched_rate_equal_counts_all_pairs=all(x["checks"]["matched_equal_counts"] for x in results),
        partition_hash_equal_across_pairs_of_a_seed="per comparison; see hashes",
        partition_hashes={x["seed"]: x["checks"]["partition_hash"] for x in results},
        prev_draw_hashes={x["seed"]: x["checks"]["prev_draw_hash"] for x in results},
        limited_draw_hashes={x["seed"]: x["checks"]["limited_draw_hash"] for x in results},
        short_halves=sorted({tuple(s) for x in results for s in x["checks"]["short_halves"]}),
        theta1_labels_differing_between_arms={x["seed"]: x["theta1_differs_between_arms"] for x in results},
        own_val_labels_differing_from_theta1={x["seed"]: x["own_differs_from_theta1"] for x in results},
        score_range=[min(x["score_range"][0] for x in results), max(x["score_range"][1] for x in results)],
        n_val=results[0]["n_val"])
    return out


def summarise_item2b(results, labels):
    K = results[0]["K"]
    ua = np.array([C.f1_whole(x, 0, "theta1") for x in results])          # (pairs, periods, K)
    ub = np.array([C.f1_whole(x, 1, "theta1") for x in results])
    va = np.array([C.f1_halves(x, 0, "refit").mean(axis=(1, 2)) for x in results])   # (pairs, periods, K)
    vb = np.array([C.f1_halves(x, 1, "refit").mean(axis=(1, 2)) for x in results])
    du, dv = (ua - ub).mean(1), (va - vb).mean(1)                        # (pairs, K)
    pu, pv = du / K, dv / K
    pc = pv - pu
    gu = np.array([C.level_whole(x, 0, "theta1") - C.level_whole(x, 1, "theta1") for x in results])
    gv = np.array([C.level_halves(x, 0, "refit") - C.level_halves(x, 1, "refit") for x in results])
    sum_dev = float(max(np.abs(pu.sum(1) - gu).max(), np.abs(pv.sum(1) - gv).max(),
                        np.abs(pc.sum(1) - (gv - gu)).max()))
    if sum_dev > 1e-12:
        raise SystemExit(f"label parts do not sum to the macro values ({sum_dev:.1e})")
    rows = []
    for k in range(K):
        keep = [i for i in range(K) if i != k]
        lo_u, lo_v = du[:, keep].mean(1), dv[:, keep].mean(1)
        rows.append(dict(label=labels[k], positives_per_period=results[0]["WPOS"][:, k].tolist(),
                         records_per_period=results[0]["WN"].tolist(),
                         part_G_U=C.t_summary(pu[:, k]), part_G_V=C.t_summary(pv[:, k]), part_change=C.t_summary(pc[:, k]),
                         left_out_G_U=C.t_summary(lo_u), left_out_G_V=C.t_summary(lo_v),
                         left_out_change=C.t_summary(lo_v - lo_u)))
    order = np.argsort(-np.abs(pc.mean(0)), kind="stable")
    top = [int(order[0]), int(order[1])]
    errs = {}
    for k in top:
        e = {}
        for m in (0, 1):
            for rule in ("theta1", "refit"):
                r = C.RULES.index(rule)
                fp = np.array([(x["FP"][m, r][..., k] * 1000.0 / x["NS"]).mean(axis=(1, 2)).mean() for x in results])
                fn = np.array([(x["FN"][m, r][..., k] * 1000.0 / x["NS"]).mean(axis=(1, 2)).mean() for x in results])
                tp = np.array([(x["TP"][m, r][..., k] * 1000.0 / x["NS"]).mean(axis=(1, 2)).mean() for x in results])
                e[f"{results[0]['arms'][m]}|{rule}"] = dict(fp_per_1000=float(fp.mean()), fn_per_1000=float(fn.mean()),
                                                           tp_per_1000=float(tp.mean()), fp_per_pair=fp.tolist(),
                                                           fn_per_pair=fn.tolist())
        pos = np.array([(x["POS"][..., k] * 1000.0 / x["NS"]).mean() for x in results])
        e["positives_per_1000_scoring"] = float(pos.mean())
        errs[labels[k]] = e
    return dict(seeds=[x["seed"] for x in results], arms=results[0]["arms"], K=K, labels=rows,
                macro=dict(G_U=C.t_summary(gu), G_V=C.t_summary(gv), change=C.t_summary(gv - gu)),
                sum_of_parts_largest_difference=sum_dev, top_two_by_abs_part_of_change=[labels[k] for k in top],
                errors_top_two=errs)


def main():
    a = C.arguments(__doc__)
    C.need(C.comparison_files())
    t0 = time.time()
    raw = a.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tasks, keys = [], []
    for n, (st, x, y) in enumerate(C.COMPARISONS):
        for s in C.STREAMS[st]["seeds"]:
            tasks.append((str(C.SCORES), st, x, y, s, C.SPLITS))
            keys.append(n)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(C.one_comparison, tasks))
    by = {}
    for n, r in zip(keys, res):
        by.setdefault(n, []).append(r)
    item1, item2b = {}, {}
    for n, (st, x, y) in enumerate(C.COMPARISONS):
        name = C.comp_name(st, x, y)
        with open(raw / f"item1_{n}.pkl", "wb") as fh:
            pickle.dump(dict(name=name, stream=st, arms=[x, y], results=by[n]), fh, protocol=4)
        item1[name] = summarise_item1(by[n])
        if y == "sequential training" and st in ("ASRS", "MultiEURLEX"):
            z = np.load(C.arm_file(st, x, 0))
            labels = [str(v) for v in z["labels"]]
            item2b[name] = summarise_item2b(by[n], labels)
    meta = dict(plan_sha256=C.sha256(C.PLAN), seconds=time.time() - t0, workers=a.workers, splits=C.SPLITS,
                periods=[j + 1 for j in C.PIDS])
    (a.out / "item1.json").write_text(json.dumps(dict(meta=meta, comparisons=item1), indent=1, default=float))
    (a.out / "item2b.json").write_text(json.dumps(dict(meta=meta, comparisons=item2b), indent=1, default=float))
    print(f"items 1 and 2b done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
