#!/usr/bin/env python3
"""Training-regime factorial (epochs per period x embedding learning-rate multiplier): attainable, mismatch and total
backward transfer per run (exact split-half), the corner interaction and main effects with a seed-paired bootstrap,
and within-epochs trends by blocked permutation; see plans/training-regimes.txt.
Reads results/training_regime/primary_C4_naive_lam*_scores.npz; writes ladder_components.csv and ladder_outcome.json
there.
Backs Table E.2 and the training-regime rows of Figure 4. Needs 51 of the 52 score files (one is shipped)."""
import glob
import itertools
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from exact_fitters import exact_fit_matrix

SPLITS = 200
LDIR = "results/training_regime"


def macro_f1(P, Y, th):
    pred = P >= th
    tp = (pred & (Y == 1)).sum(0)
    return float((2 * tp / (pred.sum(0) + Y.sum(0) + 1e-12)).mean())


def split_half_exact(P, Y, rng):
    vals = []
    for _ in range(SPLITS):
        q = rng.permutation(len(Y)); h = len(Y) // 2
        A, B = q[:h], q[h:]
        vals.append(0.5 * (macro_f1(P[B], Y[B], exact_fit_matrix(P[A], Y[A])) +
                           macro_f1(P[A], Y[A], exact_fit_matrix(P[B], Y[B]))))
    return float(np.mean(vals))


def per_run(path):
    m = re.search(r"lam([\d.]+)(?:_ep(\d+))?_s(\d+)_scores", path)
    lam = float(m.group(1))
    ep = int(m.group(2)) if m.group(2) else 3
    seed = int(m.group(3))
    z = np.load(path)
    th = z["thresholds"].astype(np.float64)
    N = 1 + max(int(re.match(r"P_(\d+)_(\d+)", k).group(1))
                for k in z.files if k.startswith("P_"))
    att, tot = [], []
    for j in range(1, N - 1):
        Y = z[f"Y_{j}"].astype(np.int64)
        Pf = z[f"P_{N-1}_{j}"].astype(np.float64)
        Pc = z[f"P_{j}_{j}"].astype(np.float64)
        rng = np.random.default_rng(7000 + 100 * seed + j)
        vf = split_half_exact(Pf, Y, rng)
        vc = split_half_exact(Pc, Y, np.random.default_rng(7000 + 100 * seed + j))
        att.append(vf - vc)
        tot.append(macro_f1(Pf, Y, th) - macro_f1(Pc, Y, th))
    r = dict(lam=lam, epochs=ep, seed=seed,
             bwt_att=float(np.mean(att)), bwt_total=float(np.mean(tot)))
    r["bwt_op"] = r["bwt_total"] - r["bwt_att"]
    return r


def blocked_perm(t, col, level_col="lam"):
    """Within-seed blocked permutation, over the seeds that hold every level (reported as complete_blocks)."""
    levels = sorted(t[level_col].unique())
    counts = t.groupby("seed")[level_col].nunique()
    seeds = sorted(counts[counts == len(levels)].index)
    vals = {s: t[t.seed == s].set_index(level_col)[col].reindex(levels).to_numpy()
            for s in seeds}
    obs_s = abs(stats.spearmanr(t[level_col], t[col]).statistic)
    obs_p = abs(stats.pearsonr(t[level_col].to_numpy(),
                               t[col].to_numpy()).statistic)
    perms = list(itertools.permutations(range(len(levels))))
    rep = np.concatenate([np.array(levels)] * len(seeds))
    cs = cp = tot = 0
    for combo in itertools.product(perms, repeat=len(seeds)):
        v = np.concatenate([vals[s][list(p)] for s, p in zip(seeds, combo)])
        cs += (abs(stats.spearmanr(rep, v).statistic) >= obs_s - 1e-12)
        cp += (abs(stats.pearsonr(rep, v).statistic) >= obs_p - 1e-12)
        tot += 1
    return dict(spearman=float(stats.spearmanr(t[level_col], t[col]).statistic),
                p_spearman=cs / tot, p_pearson=cp / tot, arrangements=tot,
                complete_blocks=len(seeds))


def main():
    files = sorted(glob.glob(f"{LDIR}/primary_C4_naive_lam*_scores.npz"))
    if not files:
        raise SystemExit(f"no score files under {LDIR}; nothing to analyse")
    t = pd.DataFrame([per_run(f) for f in files])
    t.to_csv(f"{LDIR}/ladder_components.csv", index=False)
    print(t.groupby(["epochs", "lam"])[["bwt_att", "bwt_op", "bwt_total"]]
          .agg(["mean", "std", "count"]).round(4).to_string())

    out = {"cells": json.loads(
        t.groupby(["epochs", "lam"])[["bwt_att", "bwt_op", "bwt_total"]]
        .mean().reset_index().to_json(orient="records"))}

    # corner interaction, paired by seed
    corner = t[t.lam.isin([0.0, 1.0])]
    wide = corner.pivot_table(index="seed", columns=["epochs", "lam"],
                              values="bwt_att")
    complete = wide.dropna()
    print(f"\ncorner design: {len(complete)} complete seeds of {len(wide)}")
    if {(3, 0.0), (3, 1.0), (10, 0.0), (10, 1.0)} <= set(complete.columns):
        d_int = ((complete[(10, 1.0)] - complete[(10, 0.0)])
                 - (complete[(3, 1.0)] - complete[(3, 0.0)]))
        rng = np.random.default_rng(20260829)
        bs = [d_int.values[rng.integers(0, len(d_int), len(d_int))].mean()
              for _ in range(20000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["interaction"] = dict(
            point=float(d_int.mean()), ci=[float(lo), float(hi)],
            n_seeds=int(len(d_int)), bootstrap_seed=20260829)
        for name, contrast in [
                ("lam_effect_at_e3", complete[(3, 1.0)] - complete[(3, 0.0)]),
                ("lam_effect_at_e10", complete[(10, 1.0)] - complete[(10, 0.0)]),
                ("ep_effect_at_lam0", complete[(10, 0.0)] - complete[(3, 0.0)]),
                ("ep_effect_at_lam1", complete[(10, 1.0)] - complete[(3, 1.0)])]:
            bs = [contrast.values[rng.integers(0, len(contrast),
                                               len(contrast))].mean()
                  for _ in range(20000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[name] = dict(point=float(contrast.mean()),
                             ci=[float(lo), float(hi)])
        print(json.dumps({k: v for k, v in out.items() if k != "cells"},
                         indent=1))

    # secondary: within-E trend over all four lambda levels
    out["trend"] = {}
    for ep in sorted(t.epochs.unique()):
        sub = t[t.epochs == ep]
        if sub.lam.nunique() >= 3:
            out["trend"][f"epochs_{ep}"] = {
                col: blocked_perm(sub, col) for col in
                ("bwt_att", "bwt_op", "bwt_total")}

    json.dump(out, open(f"{LDIR}/ladder_outcome.json", "w"), indent=1)
    print(f"\nwrote {LDIR}/ladder_components.csv and ladder_outcome.json")


if __name__ == "__main__":
    main()
