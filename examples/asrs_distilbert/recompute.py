"""Recompute the ASRS DistilBERT paired refit (replay against sequential training) from scores and labels.

    python examples/asrs_distilbert/recompute.py [--workers N]

Input: scores/primary_distilbert_{replay,naive}_s<k>_scores.npz, k = 0..5 (naive = sequential training).
The computation is that of paper/scripts/paired_refit_distilbert.py: refitgap.compare per pair, each arm at its
own inherited thresholds, then refitgap.method_contrast over the six pairs; periods 2 to 7 of the paper (period
ids 1-6), 200 random half partitions per period, partition b of period id j, seed s the b-th permutation of
default_rng(7000 + 100*s + j). It also recomputes the decomposition B = A + dR of the six sequential runs
(supplement Table C.4) from the first-learned scores P_j_j. Exit status 0 when every per-pair level matches
../figure2/distilbert_levels.csv and every lead, interval and decomposition term matches the stored values to TOL.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import refitgap as rg
from refitgap.runs import load_run

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEEDS, PERIODS, SPLITS, TOL = list(range(6)), [1, 2, 3, 4, 5, 6], 200, 1e-10
BOOT, BOOT_SEED = 20000, 20260920
# Table C.4, paired refit's sequential runs: full-precision values of the original analysis (bootstrap over seeds)
DECOMP_STORED = dict(B=(0.016778717383476454, [0.014379477823539488, 0.01886415063316136]),
                     dR=(0.0187495575420209, [0.016240435476064925, 0.020814011024650914]),
                     A=(-0.0019708401585444477, [-0.002710050522562568, -0.001241098167838517]))


def path(arm, s):
    return HERE / "scores" / f"primary_distilbert_{arm}_s{s}_scores.npz"


def pair(s):
    """One initialisation: replay (a) against sequential (b), each at its own inherited thresholds."""
    a, b = load_run(path("replay", s), periods=PERIODS), load_run(path("naive", s), periods=PERIODS)
    kw = dict(seed=s, splits=SPLITS, period_ids=PERIODS, labels_final=b["Ys"], rule_name="inherited")
    at_a = rg.compare(a["Ps"], b["Ps"], a["Ys"], a["thresholds"], **kw)
    same = bool(np.array_equal(a["thresholds"], b["thresholds"]))
    at_b = at_a if same else rg.compare(a["Ps"], b["Ps"], a["Ys"], b["thresholds"], **kw)
    per = {j: dict(replay=(at_a["per_period"][j]["carried"]["ref"], at_a["per_period"][j]["refitted"]["ref"]),
                   sequential=(at_b["per_period"][j]["carried"]["final"], at_b["per_period"][j]["refitted"]["final"]))
           for j in PERIODS}
    # backward decomposition of the sequential run: model j on period j against the final model
    seq_final = load_run(path("naive", s), periods=PERIODS)
    seq_first = load_run(path("naive", s), model="diagonal", periods=PERIODS)
    rule = {"inherited": seq_final["thresholds"]}
    ev_f = rg.evaluate(seq_final["Ps"], seq_final["Ys"], s, PERIODS, rules=rule, splits=SPLITS, single_model=False)
    ev_r = rg.evaluate(seq_first["Ps"], seq_first["Ys"], s, PERIODS, rules=rule, splits=SPLITS, single_model=False)
    d = rg.decompose(ev_r, ev_f, "inherited", periods=PERIODS)
    return s, at_a["evaluations"]["ref"], at_b["evaluations"]["final"], per, (d["D"], d["R"], d["A"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = sorted(ex.map(pair, SEEDS), key=lambda r: r[0])

    # per-pair levels against the stored levels
    stored = {}
    with open(REPO / "examples" / "figure2" / "distilbert_levels.csv") as fh:
        for r in csv.DictReader(fh):
            if r["stream"] == "asrs":
                stored[(r["arm"], int(r["seed"]), int(r["period"]))] = (float(r["U_inherited"]), float(r["V_refit"]))
    lv = max(abs(x - y) for s, _, _, per, _ in res for j in PERIODS for arm in ("replay", "sequential")
             for x, y in zip(per[j][arm], stored[(arm, s, j)]))

    # the three leads, paired t intervals over the six initialisations
    mc = rg.method_contrast({r[0]: r[1] for r in res}, {r[0]: r[2] for r in res}, "inherited",
                            names=("replay", "sequential"), same_rule=False)
    exp = json.loads((REPO / "examples" / "figure2" / "expected.json").read_text())["distilbert_asrs_sequential"]
    print("replay minus sequential training, ASRS, DistilBERT, 6 pairs, periods 2-7, macro-F1, 95% paired t")
    worst = 0.0
    for ours, theirs, text in (("lead_at_carried_rule", "inherited", "at the inherited thresholds"),
                               ("lead_after_refit", "refitted", "after both refits"),
                               ("change", "change", "change (refitted - inherited)")):
        t = mc["terms"][ours]
        per_seed = [t["per_seed"][s] for s in SEEDS]
        worst = max(worst, abs(t["mean"] - exp[theirs]["mean"]),
                    *(abs(x - y) for x, y in zip(t["ci95"], exp[theirs]["ci"])),
                    *(abs(x - y) for x, y in zip(per_seed, exp[theirs]["per_pair"])))
        print(f"  {text:30s} {t['mean']:+.4f}  [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]")

    # decomposition of the sequential runs, percentile bootstrap over seeds as in the original analysis
    per = np.array([r[4] for r in res])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(per), (BOOT, len(per)))
    boot = per[idx].mean(axis=1)
    print("sequential runs, B = A + dR, periods 2-7, 95% percentile bootstrap over seeds")
    dw = 0.0
    for c, name in enumerate(("B", "dR", "A")):
        m, ci = float(per[:, c].mean()), [float(np.percentile(boot[:, c], q)) for q in (2.5, 97.5)]
        dw = max(dw, abs(m - DECOMP_STORED[name][0]), *(abs(x - y) for x, y in zip(ci, DECOMP_STORED[name][1])))
        print(f"  {name:2s} {m:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    print(f"largest difference: per-pair levels {lv:.1e}, leads and intervals {worst:.1e}, "
          f"decomposition {dw:.1e} (tolerance {TOL:g})")
    ok = max(lv, worst, dw) <= TOL
    print("match" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
