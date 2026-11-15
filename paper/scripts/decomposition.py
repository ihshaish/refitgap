#!/usr/bin/env python3
"""The decomposition B = A + dR from two score matrices: the deficit B at the inherited thresholds splits into an
attainable gap A (split-half cross-fitted, same partitions for both models) and a mismatch change dR.
Reads one run's npz (P_<i>_<j>, Y_<j>, thresholds), by default
results/multieurlex_bigru/eurlex_C2_naive_lam1_s0_scores.npz; prints per period, writes nothing. Used by the
method comparison; tests/test_crossfit.py checks the package against it.
Needs numpy and scikit-learn only; the demo uses 20 partitions, the paper 200."""
import sys

import numpy as np
from sklearn.metrics import f1_score

GRID = np.linspace(0.05, 0.95, 19)


def macro_f1(P, Y, theta):
    return f1_score(Y, (P >= theta).astype(int), average="macro", zero_division=0)


def fit_exact(P, Y):
    """Per label, the cutpoint over all score-induced cutpoints that maximises F1.
    Positive rule is score >= theta; ties go to the highest maximiser. Same code
    as constructed_drifts.fit_exact, copied so this file stands alone."""
    P = np.asarray(P, dtype=np.float64)
    out = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        s, y = P[:, k], Y[:, k].astype(bool)
        o = np.argsort(-s, kind="mergesort")
        ss, ys = s[o], y[o]
        npos = int(ys.sum())
        if npos == 0:
            out[k] = float(ss[0]) + 1e-9
            continue
        f1 = 2 * np.cumsum(ys) / (np.arange(1, len(ss) + 1) + npos)
        ok = np.ones(len(ss), bool)
        ok[:-1] = ss[:-1] > ss[1:]
        out[k] = float(ss[int(np.argmax(np.where(ok, f1, -1.0)))])
    return out


def fit_grid(P, Y):
    """The coarser fitter a deployment would use on a few hundred records."""
    th = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        f = [f1_score(Y[:, k], (P[:, k] >= g).astype(int), zero_division=0) for g in GRID]
        th[k] = GRID[int(np.argmax(f))]
    return th


def attainable_pair(P_ref, P_final, Y, splits=200, rng=None):
    """Split-half cross-fitted attainable macro-F1 for both models, same partitions.
    Returns (V_ref, V_final)."""
    rng = np.random.default_rng(0) if rng is None else rng
    v_ref, v_fin = [], []
    for _ in range(splits):
        perm = rng.permutation(len(Y))
        h = len(Y) // 2
        a, b = perm[:h], perm[h:]
        v_ref.append(0.5 * (macro_f1(P_ref[b], Y[b], fit_exact(P_ref[a], Y[a]))
                            + macro_f1(P_ref[a], Y[a], fit_exact(P_ref[b], Y[b]))))
        v_fin.append(
            0.5
            * (
                macro_f1(P_final[b], Y[b], fit_exact(P_final[a], Y[a]))
                + macro_f1(P_final[a], Y[a], fit_exact(P_final[b], Y[b]))
            )
        )
    return float(np.mean(v_ref)), float(np.mean(v_fin))


def decompose(P_ref, P_final, Y, theta, splits=200, rng=None):
    """One period. U is performance at the inherited rule, V the attainable level.
    B = U_ref - U_final, A = V_ref - V_final, R = V - U for each model, dR = R_final - R_ref."""
    Y = np.asarray(Y).astype(int)
    U_ref, U_fin = macro_f1(P_ref, Y, theta), macro_f1(P_final, Y, theta)
    V_ref, V_fin = attainable_pair(P_ref, P_final, Y, splits, rng)
    out = dict(U_ref=U_ref, U_final=U_fin, V_ref=V_ref, V_final=V_fin,
               B=U_ref - U_fin, A=V_ref - V_fin, R_ref=V_ref - U_ref, R_final=V_fin - U_fin)
    out["dR"] = out["R_final"] - out["R_ref"]
    assert abs(out["B"] - (out["A"] + out["dR"])) < 1e-9
    return out


def threshold_update(P_final, Y, theta, n=500, draws=20, rng=None):
    """What a refit on n labelled records of the period buys, scored on the rest.
    Mean gain over draws, in macro-F1, against the inherited rule on the same rest."""
    rng = np.random.default_rng(0) if rng is None else rng
    Y = np.asarray(Y).astype(int)
    gains = []
    for _ in range(draws):
        sel = rng.choice(len(Y), min(n, len(Y) // 2), replace=False)
        rest = np.setdiff1d(np.arange(len(Y)), sel)
        gains.append(
            macro_f1(P_final[rest], Y[rest], fit_grid(P_final[sel], Y[sel])) - macro_f1(P_final[rest], Y[rest], theta)
        )
    return float(np.mean(gains))


def from_npz(path, splits=200, seed=0, periods=None):
    """Run the decomposition on every historical period of one stored run."""
    z = np.load(path)
    theta = z["thresholds"].astype(np.float64)
    N = max(int(k.split("_")[1]) for k in z.files if k.startswith("P_")) + 1
    periods = range(1, N - 1) if periods is None else periods
    rows = []
    for j in periods:
        r = decompose(
            z[f"P_{j}_{j}"], z[f"P_{N-1}_{j}"], z[f"Y_{j}"], theta, splits, np.random.default_rng(seed * 100 + j)
        )
        r["period"] = j + 1
        rows.append(r)
    return rows


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/multieurlex_bigru/eurlex_C2_naive_lam1_s0_scores.npz"
    rows = from_npz(path, splits=20)
    print(f"{path}\n period      U_ref  U_final    V_ref  V_final        B        A       dR")
    for r in rows:
        print(f" {r['period']:6d}  {r['U_ref']:9.4f}{r['U_final']:9.4f}{r['V_ref']:9.4f}{r['V_final']:9.4f}"
              f"  {r['B']:+8.4f} {r['A']:+8.4f} {r['dR']:+8.4f}")
    m = {k: float(np.mean([r[k] for r in rows])) for k in ("B", "A", "dR")}
    print(f" mean over periods 2 to {rows[-1]['period']}:  B {m['B']:+.4f} = A {m['A']:+.4f} + dR {m['dR']:+.4f}")
    print(" (20 split-half partitions here; the paper uses 200, and ten runs, so expect the third decimal to move)")
