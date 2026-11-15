#!/usr/bin/env python3
"""Exact F1-optimal threshold fitters shared by the analyses (bwt_statistics.py, prior_movement_*.py,
training_regime_factorial.py). Reads and writes nothing; the matrix fitter is constructed_drifts.fit_exact,
and exact_th_label adds optional record weights with the same tie rule. A self-test runs at import.
Needs nothing beyond the repository."""
import numpy as np

from constructed_drifts import fit_exact


def exact_th_label(s, y, w=None):
    """Exact F1-optimal threshold for one label, optional record weights.
    Same tie semantics as fit_exact: candidates are distinct observed scores,
    positive rule s >= theta, highest maximising cutpoint wins."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    o = np.argsort(-s, kind="mergesort")
    ss, ys = s[o], y[o]
    wts = np.ones(len(ss)) if w is None else np.asarray(w, dtype=np.float64)[o]
    pos = float((wts * ys).sum())
    if pos <= 0:
        return float(ss[0]) + 1e-9
    f1 = 2.0 * np.cumsum(wts * ys) / (np.cumsum(wts) + pos)
    ok = np.ones(len(ss), bool)
    ok[:-1] = ss[:-1] > ss[1:]
    return float(ss[int(np.argmax(np.where(ok, f1, -1.0)))])


def exact_fit_matrix(P, Y):
    """Per-label exact thresholds for a score matrix."""
    return fit_exact(P, Y)


def _self_test():
    rng = np.random.default_rng(20260829)
    GRID = np.linspace(0.05, 0.95, 19)
    for _ in range(50):
        n, K = int(rng.integers(40, 300)), int(rng.integers(2, 8))
        P = rng.random((n, K))
        Y = (rng.random((n, K)) < rng.uniform(0.05, 0.5)).astype(np.int64)
        thm = fit_exact(P, Y)
        ths = np.array([exact_th_label(P[:, k], Y[:, k]) for k in range(K)])
        assert np.allclose(thm, ths, atol=0), \
            "scalar exact fitter diverged from fit_exact"
        # exact must dominate the 19-point grid in in-sample per-label F1
        for k in range(K):
            y, p = Y[:, k], P[:, k]
            if y.sum() == 0:
                continue
            def f1_at(th):
                pred = p >= th
                tp = (pred & (y == 1)).sum()
                return 2 * tp / (pred.sum() + y.sum() + 1e-12)
            ge = max(f1_at(g) for g in GRID)
            assert f1_at(thm[k]) >= ge - 1e-12, "exact fitter beaten by grid"
        # weighted fitter reduces to unweighted at w = 1
        k0 = 0
        assert exact_th_label(P[:, k0], Y[:, k0], np.ones(n)) == ths[k0]


_self_test()
