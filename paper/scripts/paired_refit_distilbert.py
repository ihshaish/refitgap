"""Paired refit, DistilBERT (seeds 0-5, periods 2 to 7), through refitgap.compare() and method_contrast():
replay against sequential training on ASRS and MultiEURLEX, and against extra current-data updates on MultiEURLEX.
python scripts/paired_refit_distilbert.py RUNS_DIR STORED_DIR OUT_JSON [workers]; exit 0 if every level and contrast
equals the stored paired_refit.json and levels.csv (STORED_DIR/refit_<name>/) to 1e-12. Backs Table C.1 and Figure 2.
Needs the 30 DistilBERT score files {primary,eurlex}_distilbert_{replay,naive,replaycur}_s<k>_scores.npz
(not shipped)."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import refitgap as rg
from refitgap.runs import load_run

SEEDS, PERIODS, SPLITS, TOL = list(range(6)), list(range(1, 7)), 200, 1e-12
CONTRASTS = {  # name: (stream, arm a, arm b, names)
    "primary": ("primary", "replay", "naive", ("replay", "sequential")),
    "eurlex": ("eurlex", "replay", "naive", ("replay", "sequential")),
    "eurlex_replay_vs_replaycur": ("eurlex", "replay", "replaycur", ("replay", "replaycur")),
}


def one(task):
    runs, name, s = task
    stream, arm_a, arm_b, _ = CONTRASTS[name]
    f = lambda arm: Path(runs) / f"{stream}_distilbert_{arm}_s{s}_scores.npz"      # noqa: E731
    a, b = load_run(f(arm_a), periods=PERIODS), load_run(f(arm_b), periods=PERIODS)
    kw = dict(seed=s, splits=SPLITS, period_ids=PERIODS, labels_final=b["Ys"], rule_name="inherited")
    # the two arms carry different theta_1 on a few labels and compare() takes one rule, so each arm
    # is evaluated under its own rule; the refitted levels do not depend on it
    at_a = rg.compare(a["Ps"], b["Ps"], a["Ys"], a["thresholds"], **kw)
    same = bool(np.array_equal(a["thresholds"], b["thresholds"]))
    at_b = at_a if same else rg.compare(a["Ps"], b["Ps"], a["Ys"], b["thresholds"], **kw)
    ev_a, ev_b = at_a["evaluations"]["ref"], at_b["evaluations"]["final"]
    per = {j: dict(U_a=at_a["per_period"][j]["carried"]["ref"], U_b=at_b["per_period"][j]["carried"]["final"],
                   V_a=at_a["per_period"][j]["refitted"]["ref"], V_b=at_b["per_period"][j]["refitted"]["final"])
           for j in PERIODS}
    return name, s, ev_a, ev_b, per, at_a["carried"]["ref"] - at_a["carried"]["final"], \
        int(np.sum(a["thresholds"] != b["thresholds"]))


def stored_levels(path):
    import csv
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["model"] == "final" and r["rule"] == "inherited" and r["period"].isdigit():
                out[(r["method"], int(r["seed"]), int(r["period"]))] = (float(r["U_full"]), float(r["V_sep"]))
    return out


def main(runs, stored_dir, out, workers=4):
    tasks = [(runs, n, s) for n in CONTRASTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, tasks))
    worst, report = 0.0, {}
    for name, (_, _, _, names) in CONTRASTS.items():
        rows = [r for r in res if r[0] == name]
        st = json.loads((Path(stored_dir) / f"refit_{name}" / "paired_refit.json").read_text())
        lv = stored_levels(Path(stored_dir) / f"refit_{name}" / "levels.csv")
        lw = 0.0
        for _, s, _, _, per, _, _ in rows:
            for j in PERIODS:
                ua, va = lv[(names[0], s, j)]
                ub, vb = lv[(names[1], s, j)]
                lw = max(lw, abs(per[j]["U_a"] - ua), abs(per[j]["U_b"] - ub),
                         abs(per[j]["V_a"] - va), abs(per[j]["V_b"] - vb))
        mc = rg.method_contrast({r[1]: r[2] for r in rows}, {r[1]: r[3] for r in rows}, "inherited", names=names,
                                same_rule=False)
        terms = {}
        pw = 0.0
        for ours, theirs in (("lead_at_carried_rule", "G_U"), ("lead_after_refit", "G_V"), ("placement", "Delta")):
            t = mc["terms"][ours]
            per_seed = [t["per_seed"][s] for s in SEEDS]
            pw = max(pw, max(abs(x - y) for x, y in zip(per_seed, st[theirs]["per_pair"])),
                     abs(t["mean"] - st[theirs]["mean"]))
            if "t95" in st[theirs]:
                pw = max(pw, abs(t["ci95"][0] - st[theirs]["t95"][0]), abs(t["ci95"][1] - st[theirs]["t95"][1]))
            terms[ours] = dict(per_pair=per_seed, mean=t["mean"], ci95_t_over_seeds=t["ci95"],
                               stored=theirs, stored_mean=st[theirs]["mean"], stored_t95=st[theirs].get("t95"),
                               stored_class_word=st[theirs].get("class_word"))
        shared = [r[5] for r in sorted(rows, key=lambda r: r[1])]
        theta_diff = [r[6] for r in sorted(rows, key=lambda r: r[1])]
        worst = max(worst, lw, pw)
        report[name] = dict(direction=mc["direction"], terms=terms,
                            placement_share=mc["placement_share"],
                            labels_with_different_theta1=theta_diff,
                            stored_labels_with_different_theta1=st["labels_with_different_theta1"],
                            lead_at_one_shared_rule_a=dict(per_pair=shared, mean=float(np.mean(shared))),
                            max_abs_difference_levels=lw, max_abs_difference_contrasts=pw)
        print(f"\n{name}: {mc['direction']}")
        for ours, t in terms.items():
            cw = f"  stored class {t['stored_class_word']}" if t["stored_class_word"] else ""
            print(f"  {ours:22s} {t['mean']:+.4f} t95 [{t['ci95_t_over_seeds'][0]:+.4f}, "
                  f"{t['ci95_t_over_seeds'][1]:+.4f}]   stored {t['stored']} {t['stored_mean']:+.4f}{cw}")
        print(f"  lead at one shared rule (a's theta_1 for both): {np.mean(shared):+.4f} "
              f"(labels with different theta_1 per seed {theta_diff})")
        print(f"  largest difference: levels {lw:.2e}, per-pair contrasts, means and t95 {pw:.2e}")
    print(f"\nlargest difference from the stored values: {worst:.2e}")
    Path(out).write_text(json.dumps(dict(block="DistilBERT encoder block (runs_cuda), seeds 0-5, period ids 1-6",
                                         splits=SPLITS, contrasts=report, max_abs_difference=worst, tolerance=TOL),
                                    indent=1, default=float))
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4))
