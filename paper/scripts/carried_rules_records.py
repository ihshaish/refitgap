"""Carried rules, item 2a (plans/carried-rules.txt): records and initialisations resampled together, for
DistilBERT replay minus sequential training on ASRS (and replay minus extra updates on current data on
MultiEURLEX). Per replicate the pairs are drawn with replacement, and for each period 2 to 7 and each of 10
partitions records are drawn with replacement within each half, one draw shared by both arms and all pairs;
thresholds are refitted within the replicate. Percentile 95% intervals of G_U (same folds), G_V and the change.
Seed default_rng(20261118). Backs the record-resampling sentence of supplement Appendix C.5.

    python scripts/carried_rules_records.py SCORES_DIR --stream ASRS --comparator "sequential training" \
        [--replicates 1000] [--workers N] [--out DIR] [--time-only]

Writes item2a_<stream>.json. About 1.4 million threshold fits per stream: hours on a few cores
(--replicates 5 --time-only projects the full run). Stored outputs: results/carried_rules/.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from refitgap.fitters import fit_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carried_rules_core as C                       # noqa: E402

SEED = 20261118
PIDS = list(range(1, 7))
NPART = 10
FINAL = 7
TAG = {"sequential training": "naive", "replay": "replay", "extra updates on current data": "replaycur"}
PREFIX = {"ASRS": "primary_distilbert", "MultiEURLEX": "eurlex_distilbert"}
SEEDS = {"ASRS": list(range(6)), "MultiEURLEX": list(range(6))}
SCORES = None

_D = {}


def arm_path(stream, arm, s):
    return Path(SCORES) / f"{PREFIX[stream]}_{TAG[arm]}_s{s}_scores.npz"


def load(stream, a, b):
    seeds = SEEDS[stream]
    out = dict(seeds=seeds, P={}, theta={}, Y=None)
    Y0 = None
    for s in seeds:
        for m, arm in enumerate((a, b)):
            z = np.load(arm_path(stream, arm, s))
            out["P"][(s, m)] = {j: z[f"P_{FINAL}_{j}"].astype(np.float64) for j in PIDS}
            out["theta"][(s, m)] = z["thresholds"].astype(np.float64)
            Y = {j: z[f"Y_{j}"].astype(int) for j in PIDS}
            if Y0 is None:
                Y0 = Y
            elif not all(np.array_equal(Y0[j], Y[j]) for j in PIDS):
                raise SystemExit("labels differ across arms or initialisations: one record draw cannot be shared")
    out["Y"] = Y0
    out["n"] = {j: len(Y0[j]) for j in PIDS}
    return out


def draw_replicate(rng, n_pairs, ns):
    """One replicate's draws, in this order: pair indices, then per period and partition the permutation and
    the two within-half resamples. Returns (pairs, {j: [(fit-side half 0 records, half 1 records), ...]})."""
    pairs = rng.integers(0, n_pairs, n_pairs)
    halves = {}
    for j in PIDS:
        n = ns[j]
        row = []
        for _ in range(NPART):
            perm = rng.permutation(n)
            h0, h1 = perm[: n // 2], perm[n // 2:]
            i0 = rng.integers(0, len(h0), len(h0))
            i1 = rng.integers(0, len(h1), len(h1))
            row.append((h0[i0], h1[i1]))
        halves[j] = row
    return pairs, halves


def macro_f1_counts(pred, Y):
    Yb = Y.astype(bool)
    tp = (pred & Yb).sum(0)
    den = pred.sum(0) + Yb.sum(0)
    return float(np.where(den > 0, 2.0 * tp / np.maximum(den, 1), 0.0).mean())


def _init(scores, stream, a, b):
    global SCORES
    SCORES = scores
    _D.update(load(stream, a, b))


def replicate_values(task):
    """(r, pairs, halves) -> per drawn pair U and V of both arms (same folds), and the replicate's means."""
    r, pairs, halves = task
    seeds = _D["seeds"]
    Y = _D["Y"]
    uniq = sorted(set(int(p) for p in pairs))
    per = {}
    for pi in uniq:
        s = seeds[pi]
        U = np.zeros((2, len(PIDS), NPART, 2))
        V = np.zeros((2, len(PIDS), NPART, 2))
        for c, j in enumerate(PIDS):
            for p, (h0, h1) in enumerate(halves[j]):
                for d in (0, 1):
                    F, S = (h0, h1) if d == 0 else (h1, h0)
                    YF, YS = Y[j][F], Y[j][S]
                    for m in (0, 1):
                        P = _D["P"][(s, m)][j]
                        PS = P[S]
                        U[m, c, p, d] = macro_f1_counts(PS >= _D["theta"][(s, m)], YS)
                        V[m, c, p, d] = macro_f1_counts(PS >= fit_exact(P[F], YF), YS)
        per[pi] = dict(U=U.mean(axis=(1, 2, 3)), V=V.mean(axis=(1, 2, 3)))
    w = np.array([per[int(p)]["U"] for p in pairs]), np.array([per[int(p)]["V"] for p in pairs])
    Um, Vm = w[0].mean(0), w[1].mean(0)
    return dict(replicate=r, pairs=[int(p) for p in pairs], U_a=float(Um[0]), U_b=float(Um[1]), V_a=float(Vm[0]),
                V_b=float(Vm[1]), G_U=float(Um[0] - Um[1]), G_V=float(Vm[0] - Vm[1]),
                change=float((Vm[0] - Vm[1]) - (Um[0] - Um[1])))


def run(stream, a, b, replicates, workers):
    data = load(stream, a, b)
    rng = np.random.default_rng(SEED)
    npairs = len(data["seeds"])
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(SCORES), stream, a, b)) as ex:
        futs = []
        for r in range(replicates):
            pairs, halves = draw_replicate(rng, npairs, data["n"])
            futs.append(ex.submit(replicate_values, (r, pairs, halves)))
            if len(futs) - len(out) > 4 * workers:                  # bounded queue
                out.append(futs[len(out)].result())
        for f in futs[len(out):]:
            out.append(f.result())
    return out, time.time() - t0


def summarise(rows):
    res = {}
    for k in ("G_U", "G_V", "change", "U_a", "U_b", "V_a", "V_b"):
        x = np.array([r[k] for r in rows])
        res[k] = dict(mean_of_replicates=float(x.mean()), median=float(np.median(x)),
                      percentile95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))])
    return res


def main():
    global SCORES
    ap = argparse.ArgumentParser(description="records and initialisations resampled together")
    ap.add_argument("scores", help="directory holding the score files (not in the repository)")
    ap.add_argument("--stream", required=True, choices=sorted(PREFIX))
    ap.add_argument("--arm", default="replay")
    ap.add_argument("--comparator", required=True, choices=sorted(TAG))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--out", default=str(C.DEFAULT_OUT))
    a = ap.parse_args()
    SCORES = a.scores
    C.need([arm_path(a.stream, arm, s) for arm in (a.arm, a.comparator) for s in SEEDS[a.stream]])
    rows, secs = run(a.stream, a.arm, a.comparator, a.replicates, a.workers)
    fits = a.replicates * len(PIDS) * NPART * 2 * len(SEEDS[a.stream]) * 2
    proj = secs / a.replicates * 1000
    print(f"{a.stream} {a.arm} minus {a.comparator}: {a.replicates} replicates in {secs:.1f} s on {a.workers} workers; "
          f"projected 1000 replicates {proj / 60:.1f} min (planned fits {fits}, fewer computed when pairs repeat)")
    res = dict(stream=a.stream, comparison=f"{a.arm} minus {a.comparator}", seed=SEED, replicates=a.replicates,
               partitions_per_period=NPART, periods=[j + 1 for j in PIDS], workers=a.workers, seconds=secs,
               projected_seconds_1000=proj, rows=rows)
    if not a.time_only:
        res["summary"] = summarise(rows)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"item2a_{a.stream}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
