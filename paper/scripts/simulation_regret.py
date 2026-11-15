#!/usr/bin/env python3
"""Simulation of threshold-selection regret in the estimated gap: a reference scorer against scorers with the same
population optimum but harder selection ("curved", "narrow") and one with a known gap ("worse5"), for two prevalence
profiles; 200 studies of ten paired test sets at n = 250, 500, 1000, how often the interval lies above 0 and 0.005.
Writes results/paired_refit/simulation_regret.json (--quick: a small run to _quick.json).
Backs Appendix A, "Finite labelled budgets" (second simulation). Needs nothing beyond the repository; hours of CPU."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as Beta
from sklearn.metrics import f1_score

from constructed_drifts import fit_exact

OUT = Path(
    "results/paired_refit/simulation_regret_quick.json"
    if "--quick" in sys.argv
    else "results/paired_refit/simulation_regret.json"
)
PREV = {"asrs": np.array([0.05, 0.07, 0.08, 0.14, 0.16, 0.24, 0.25, 0.48, 0.51]),
        "rare": np.array([0.02, 0.07, 0.08, 0.14, 0.16, 0.24, 0.25, 0.48, 0.51])}
SEP = np.array([1.2, 1.0, 1.4, 1.1, 1.3, 0.9, 1.0, 1.2, 1.1])
K = 9
NPOP, NEVAL = 400_000, 100_000
STUDIES, BLOCKS, M, BOOT = 200, 10, 50, 2000
BUDGETS = (250, 500, 1000)
W_EASY, HARD = 0.6, 0.25
C_NARROW = 4.0
SEEDS = {("asrs", "curved"): 20261001, ("asrs", "narrow"): 20261002, ("asrs", "worse5"): 20261003,
         ("rare", "curved"): 20261011, ("rare", "narrow"): 20261012, ("rare", "worse5"): 20261013}
QUICK = "--quick" in sys.argv
if QUICK:
    STUDIES, BLOCKS, M, NPOP, NEVAL = 4, 4, 5, 100_000, 50_000


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def f1_label(p, y, th):
    pred = p >= th
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


# scorers: latent uniforms (u, c), labels y, label index k -> scores for label k
def s1(u, c, y, k, sep=SEP):
    return np.where(y == 1, Beta.ppf(u, 2 + sep[k], 2), Beta.ppf(u, 2, 2 + sep[k]))


def s_curved(u, c, y, k, g):
    easy = c < W_EASY
    pos = np.where(easy, Beta.ppf(u, 2 + g[k] * SEP[k], 2), Beta.ppf(u, 2 + HARD * SEP[k], 2))
    return np.where(y == 1, pos, Beta.ppf(u, 2, 2 + SEP[k]))


def s_narrow(u, c, y, k, sepn):
    a = C_NARROW
    return np.where(y == 1, Beta.ppf(u, a * (2 + sepn[k]), 2 * a), Beta.ppf(u, 2 * a, a * (2 + sepn[k])))


def pop_f1_label(scorer, k, prev, rng_seed=1):
    """Population optimum of one label: fit on NPOP, evaluate on NEVAL, both drawn here."""
    r = np.random.default_rng(rng_seed)
    y = (r.random(NPOP) < prev[k]).astype(int); u = r.random(NPOP); c = r.random(NPOP)
    p = scorer(u, c, y, k)
    th = fit_exact(p[:, None], y[:, None])[0]
    r2 = np.random.default_rng(rng_seed + 777)
    y2 = (r2.random(NEVAL) < prev[k]).astype(int); u2 = r2.random(NEVAL); c2 = r2.random(NEVAL)
    p2 = scorer(u2, c2, y2, k)
    return f1_label(p2, y2, th), th, (p2, y2)


def width_near_max(p, y, f1max, tol=0.005):
    grid = np.linspace(0.01, 0.99, 197)
    vals = np.array([f1_label(p, y, t) for t in grid])
    ok = grid[vals >= f1max - tol]
    return float(ok.max() - ok.min()) if len(ok) else 0.0


def solve(param_scorer, k, prev, target, lo, hi, iters=28):
    """Bisection on a scalar parameter so that the label's population F1 optimum equals target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f, _, _ = pop_f1_label(lambda u, c, y, kk: param_scorer(u, c, y, kk, mid), k, prev)
        if f < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate(prev):
    """Per-label parameters of the comparison scorers for one prevalence profile."""
    base, widths = [], []
    for k in range(K):
        f, th, (p, y) = pop_f1_label(s1, k, prev)
        base.append(f); widths.append(width_near_max(p, y, f))
    base = np.array(base)
    g = np.zeros(K); sepn = np.zeros(K)
    for k in range(K):
        g[k] = solve(lambda u, c, y, kk, val: s_curved(u, c, y, kk, np.full(K, val)), k, prev, base[k], 0.5, 6.0)
        sepn[k] = solve(lambda u, c, y, kk, val: s_narrow(u, c, y, kk, np.full(K, val)), k, prev, base[k], 0.05, 6.0)
    return base, widths, g, sepn


def describe(scorer, prev):
    f1s, widths, ths = [], [], []
    for k in range(K):
        f, th, (p, y) = pop_f1_label(scorer, k, prev)
        f1s.append(f); ths.append(th); widths.append(width_near_max(p, y, f))
    return dict(macro_f1=float(np.mean(f1s)), per_label_f1=[float(x) for x in f1s],
                thresholds=[float(x) for x in ths], width_within_0005=[float(x) for x in widths])


def draw_pair(n, prev, scorerA, scorerB, rng):
    Y = (rng.random((n, K)) < prev).astype(int)
    U = rng.random((n, K)); C = rng.random((n, K))
    PA = np.empty((n, K)); PB = np.empty((n, K))
    for k in range(K):
        PA[:, k] = scorerA(U[:, k], C[:, k], Y[:, k], k)
        PB[:, k] = scorerB(U[:, k], C[:, k], Y[:, k], k)
    return PA, PB, Y


def split_half_gap(PA, PB, Y, rng):
    n = len(Y); h = n // 2; vals = []
    for _ in range(M):
        perm = rng.permutation(n); a, b = perm[:h], perm[h:]
        vA = 0.5 * (mf1(PA[b], Y[b], fit_exact(PA[a], Y[a])) + mf1(PA[a], Y[a], fit_exact(PA[b], Y[b])))
        vB = 0.5 * (mf1(PB[b], Y[b], fit_exact(PB[a], Y[a])) + mf1(PB[a], Y[a], fit_exact(PB[b], Y[b])))
        vals.append(vA - vB)
    return float(np.mean(vals))


def run_config(name, prev, scorerB, A_pop, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for n in BUDGETS:
        est, lo, hi = [], [], []
        for s in range(STUDIES):
            blocks = []
            for b in range(BLOCKS):
                PA, PB, Y = draw_pair(2 * n, prev, s1, scorerB, rng)
                blocks.append(split_half_gap(PA, PB, Y, rng))
            blocks = np.array(blocks)
            boot = np.array([blocks[rng.integers(0, BLOCKS, BLOCKS)].mean() for _ in range(BOOT)])
            est.append(blocks.mean()); lo.append(np.quantile(boot, .025)); hi.append(np.quantile(boot, .975))
        est, lo, hi = map(np.array, (est, lo, hi))
        out[str(n)] = dict(
            A_hat_mean=float(est.mean()),
            bias=float(est.mean() - A_pop),
            sd_studies=float(est.std()),
            frac_ci_above_zero=float(np.mean(lo > 0)),
            frac_ci_below_zero=float(np.mean(hi < 0)),
            frac_ci_above_0005=float(np.mean(lo > 0.005)),
            frac_point_above_0005=float(np.mean(est > 0.005)),
        )
        print(f"  {name:8s} n={n:5d}  A_pop {A_pop:+.4f}  A_hat {est.mean():+.4f} (sd {est.std():.4f})  "
              f"CI>0 {np.mean(lo > 0):.3f}  CI<0 {np.mean(hi < 0):.3f}  CI>0.005 {np.mean(lo > 0.005):.3f}", flush=True)
    return out


results = dict(settings=dict(studies=STUDIES, blocks=BLOCKS, splits=M, boot=BOOT, budgets=list(BUDGETS),
                             w_easy=W_EASY, hard=HARD, c_narrow=C_NARROW, quick=QUICK), profiles={})
for pname, prev in PREV.items():
    print(f"profile {pname}: calibrating comparison scorers", flush=True)
    base, widths, g, sepn = calibrate(prev)
    scorers = {"curved": lambda u, c, y, k: s_curved(u, c, y, k, g),
               "narrow": lambda u, c, y, k: s_narrow(u, c, y, k, sepn),
               "worse5": lambda u, c, y, k: s1(u, c, y, k, SEP * 0.95)}
    descA = describe(s1, prev)
    prof = dict(reference=descA, comparison={}, g=g.tolist(), sep_narrow=sepn.tolist())
    for sname, sc in scorers.items():
        d = describe(sc, prev)
        A_pop = descA["macro_f1"] - d["macro_f1"]
        print(
            f"  {sname}: population macro-F1 {d['macro_f1']:.4f} vs reference {descA['macro_f1']:.4f} (A_pop "
            f"{A_pop:+.4f}); "
            f"width within 0.005: ref {np.mean(descA['width_within_0005']):.3f} vs "
            f"{np.mean(d['width_within_0005']):.3f}",
            flush=True,
        )
        d["A_pop"] = float(A_pop)
        d["budgets"] = run_config(sname, prev, sc, A_pop, seed=SEEDS[(pname, sname)])
        prof["comparison"][sname] = d
        results["profiles"][pname] = prof
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(OUT, "w"), indent=1)
print("done", flush=True)
