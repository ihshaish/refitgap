"""Reproduce Figure 2: replay's lead in the five paired comparisons (supplement Table C.1).

    python examples/figure2/reproduce_figure2.py [--from-levels] [--workers N] [--out FILE]

Inputs
  BiGRU, 10 pairs per stream (seeds 0-9, both arms trained under the same seed): the stored
    score files in paper/results/asrs_bigru_paired/ and paper/results/multieurlex_bigru/.
    Recomputed from the raw scores with refitgap.compare: U at the inherited thresholds on each
    period's whole test set, V by split-half cross-fitting with 200 partitions. About 12 min of
    CPU in all, 3 to 4 min on 4 workers (--workers).
    --from-levels reads the stored per-pair levels (paper/results/paired_refit/) instead: seconds.
  DistilBERT, 6 pairs per comparison (seeds 0-5): stored per-pair, per-period levels in
    distilbert_levels.csv (U at each arm's own inherited thresholds, V cross-fitted as above).
    Its 30 score files (about 60 MB) are not in the repository, so the levels are the input
    here; paper/scripts/paired_refit_distilbert.py recomputes them from the score files.
  Periods 2 to 7 of the paper throughout (period ids 1-6).

Output
  Per comparison: the lead at the inherited thresholds (G_U), the lead after both models
  refit (G_V) and the paired change G_V - G_U, each with its 95% interval: paired t over the
  six DistilBERT pairs; percentile bootstrap over the ten BiGRU pairs (20000 resamples, numpy
  default_rng(20260920), one resample matrix shared by all terms). Printed to 4 decimals;
  --out writes full precision as JSON. Exit status 0 when every per-pair lead, mean and
  interval bound is within TOL of expected.json, 1 otherwise.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import refitgap as rg
from refitgap.inference import paired_contrast
from refitgap.runs import load_run

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "paper" / "results"
PERIODS = [1, 2, 3, 4, 5, 6]
SPLITS, BOOT, BOOT_SEED = 200, 20000, 20260920
TOL = 1e-10          # stated tolerance; the recomputation agrees to the last bit with numpy 1.26 / scipy 1.13
BIGRU = {"bigru_asrs_sequential": ("asrs_bigru_paired", "primary", "paired_replay.json"),
         "bigru_multieurlex_sequential": ("multieurlex_bigru", "eurlex", "paired_replay_eurlex.json")}
DISTILBERT = {"distilbert_asrs_sequential": ("asrs", "sequential"),
              "distilbert_multieurlex_sequential": ("multieurlex", "sequential"),
              "distilbert_multieurlex_extra_updates": ("multieurlex", "extra_updates")}


def bigru_pair(task):
    """(name, seed) -> (name, seed, per-period levels) recomputed from the two score files."""
    name, seed = task
    folder, stream, _ = BIGRU[name]
    f = lambda arm: RESULTS / folder / f"{stream}_C2_{arm}_lam1_s{seed}_scores.npz"   # noqa: E731
    rep, seq = load_run(f("replay"), periods=PERIODS), load_run(f("naive"), periods=PERIODS)
    if not np.array_equal(rep["thresholds"], seq["thresholds"]):
        raise ValueError(f"{name} seed {seed}: the two arms carry different inherited thresholds")
    r = rg.compare(rep["Ps"], seq["Ps"], rep["Ys"], rep["thresholds"], seed=seed, splits=SPLITS,
                   period_ids=PERIODS, labels_final=seq["Ys"], rule_name="inherited")
    per = {j: dict(U_rep=r["per_period"][j]["carried"]["ref"], U_seq=r["per_period"][j]["carried"]["final"],
                   V_rep=r["per_period"][j]["refitted"]["ref"], V_seq=r["per_period"][j]["refitted"]["final"])
           for j in PERIODS}
    return name, seed, per


def stored_bigru_levels(name):
    """Per-pair, per-period levels as the paper's analysis stored them."""
    st = json.loads((RESULTS / "paired_refit" / BIGRU[name][2]).read_text())
    return {int(s): {j: st["pairs"][s]["per_period"][str(j)] for j in PERIODS} for s in st["pairs"]}


def leads(levels):
    """{seed: {period: U_rep, U_seq, V_rep, V_seq}} -> {seed: (G_U, G_V)}, means over periods."""
    out = {}
    for s, per in sorted(levels.items()):
        m = {k: float(np.mean([per[j][k] for j in PERIODS])) for k in ("U_rep", "U_seq", "V_rep", "V_seq")}
        out[s] = (m["U_rep"] - m["U_seq"], m["V_rep"] - m["V_seq"])
    return out


def bootstrap_summary(g):
    """Percentile bootstrap over pairs, as the paper's analysis drew it."""
    seeds = sorted(g)
    x = {"inherited": np.array([g[s][0] for s in seeds]), "refitted": np.array([g[s][1] for s in seeds])}
    x["change"] = x["refitted"] - x["inherited"]
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(seeds), (BOOT, len(seeds)))
    out = {}
    for k, v in x.items():
        b = v[idx].mean(1)
        out[k] = dict(mean=float(v.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                      per_pair=v.tolist())
    return out


def distilbert_levels():
    """distilbert_levels.csv -> {(stream, arm): {seed: {period: (U, V)}}}."""
    out = {}
    with open(HERE / "distilbert_levels.csv") as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["stream"], r["arm"]), {}).setdefault(int(r["seed"]), {})[int(r["period"])] = (
                float(r["U_inherited"]), float(r["V_refit"]))
    return out


def t_summary(lv, stream, comparator):
    """Paired t intervals over the six DistilBERT pairs (refitgap.inference.paired_contrast)."""
    mean_over = lambda arm, i: {s: float(np.mean([lv[(stream, arm)][s][j][i] for j in PERIODS]))  # noqa: E731
                                for s in lv[(stream, arm)]}
    u_a, u_b, v_a, v_b = (
        mean_over("replay", 0),
        mean_over(comparator, 0),
        mean_over("replay", 1),
        mean_over(comparator, 1),
    )
    if set(u_a) != set(u_b):
        raise ValueError(f"{stream}: replay and {comparator} hold different seeds")
    g_u = {s: u_a[s] - u_b[s] for s in u_a}
    g_v = {s: v_a[s] - v_b[s] for s in v_a}
    out = {}
    for k, (a, b) in (("inherited", (u_a, u_b)), ("refitted", (v_a, v_b)), ("change", (g_v, g_u))):
        c = paired_contrast(a, b)
        out[k] = dict(mean=c["mean"], ci=list(c["ci95"]), per_pair=[a[s] - b[s] for s in sorted(a)])
    return out


def worst_difference(got, exp):
    return max(max(abs(got[t]["mean"] - exp[t]["mean"]), *(abs(x - y) for x, y in zip(got[t]["ci"], exp[t]["ci"])),
                   *(abs(x - y) for x, y in zip(got[t]["per_pair"], exp[t]["per_pair"])))
               for t in ("inherited", "refitted", "change"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-levels", action="store_true", help="BiGRU from stored per-pair levels, not raw scores")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="write the full-precision values as JSON")
    args = ap.parse_args(argv)
    expected = json.loads((HERE / "expected.json").read_text())

    got, source = {}, {}
    lv = distilbert_levels()
    for name, (stream, comparator) in DISTILBERT.items():
        got[name], source[name] = t_summary(lv, stream, comparator), "stored per-pair levels"
    if args.from_levels:
        levels = {name: stored_bigru_levels(name) for name in BIGRU}
    else:
        tasks = [(name, s) for name in BIGRU for s in range(10)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            levels = {name: {} for name in BIGRU}
            for name, s, per in ex.map(bigru_pair, tasks):
                levels[name][s] = per
    for name in BIGRU:
        got[name] = bootstrap_summary(leads(levels[name]))
        source[name] = "stored per-pair levels" if args.from_levels else "raw scores"

    print(f"{'replay against':38s} {'inherited':>26s} {'both refitted':>26s} {'change':>26s}   input")
    worst = 0.0
    for name, exp in expected.items():
        g = got[name]
        worst = max(worst, worst_difference(g, exp))
        cells = [
            f"{g[t]['mean']:+.4f} [{g[t]['ci'][0]:+.4f}, {g[t]['ci'][1]:+.4f}]"
            for t in ("inherited", "refitted", "change")
        ]
        print(f"{exp['label']:38s} {cells[0]:>26s} {cells[1]:>26s} {cells[2]:>26s}   {source[name]}")
    ok = worst <= TOL
    print(f"largest difference from expected.json: {worst:.1e} (tolerance {TOL:g}): {'match!' if ok else 'MISMATCH'}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(tolerance=TOL, largest_difference=worst, source=source,
                                                  comparisons=got), indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
