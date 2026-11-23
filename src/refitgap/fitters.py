"""Threshold fitters and thresholded F1.

f1_vec, macro_f1 (per-label F1 and its mean), fit_exact (best cutpoint per label over every
observed score), fit_shared_exact (one vector for several periods), fit_grid (19 points
0.05..0.95, how theta_1 was fitted). Positive means score >= theta; F1 is 0 when nothing is
predicted and nothing is positive. Fitters refuse non-finite scores and non-0/1 labels."""
import numpy as np

GRID = np.linspace(0.05, 0.95, 19)


def _check_PY(P, Y, what):
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{what}: non-finite scores")
    Y = np.asarray(Y)
    if Y.dtype != bool and not np.isin(Y, (0, 1)).all():
        raise ValueError(f"{what}: labels must be 0/1")


def _above(top):
    """A cutpoint above every score: top + 1e-9, or the next float above top where adding
    1e-9 does not change it (|top| above about 1.7e7)."""
    top = np.asarray(top, np.float64)
    c = top + 1e-9
    return np.where(c > top, c, np.nextafter(top, np.inf))


def f1_vec(P, Y, theta):
    """Per-label F1 at thresholds theta, zero_division = 0 (as sklearn's f1_score with zero_division=0)."""
    pred = P >= theta
    Yb = Y.astype(bool)
    tp = (pred & Yb).sum(0)
    den = pred.sum(0) + Yb.sum(0)
    return np.where(den > 0, 2.0 * tp / np.maximum(den, 1), 0.0)


def macro_f1(P, Y, theta):
    return float(f1_vec(P, Y, theta).mean())


def fit_exact(P, Y):
    """Per label, the cutpoint over all score-induced cutpoints that maximises F1 on (P, Y).
    Ties go to the highest maximiser (exactly: each F1 value is one correctly rounded division
    of integers, so equal F1 values are equal floats). A label with no positive gets a cutpoint
    above every score (predicts nothing; see _above)."""
    P = np.asarray(P, dtype=np.float64)
    _check_PY(P, Y, "fit_exact")
    Yb = np.asarray(Y).astype(bool)
    n, K = P.shape
    o = np.argsort(-P, axis=0, kind="mergesort")          # stable descending sort per column
    ss = np.take_along_axis(P, o, 0)
    ys = np.take_along_axis(Yb, o, 0)
    npos = ys.sum(0)
    f1 = 2 * np.cumsum(ys, 0) / (np.arange(1, n + 1)[:, None] + npos[None, :])
    ok = np.ones((n, K), bool)
    ok[:-1] = ss[:-1] > ss[1:]                             # only cut between distinct scores
    idx = np.argmax(np.where(ok, f1, -1.0), 0)
    out = ss[idx, np.arange(K)].astype(np.float64)
    none = npos == 0
    out[none] = _above(ss[0, none])
    return out


def fit_shared_exact(Ps, Ys, w=None):
    """Per-period scores and labels (and weights w) -> one (K,) threshold vector maximising
    sum_j w_j F1_jk(t) per label over all periods' cutpoints. Ties go to the higher cutpoint, up
    to float rounding of the sum; a label with no positive anywhere gets a cutpoint above every
    score. One period gives fit_exact."""
    J = len(Ps)
    K = Ys[0].shape[1]
    w = np.full(J, 1.0 / J) if w is None else np.asarray(w, float)
    for j in range(J):
        _check_PY(np.asarray(Ps[j], np.float64), Ys[j], f"fit_shared_exact, period {j}")
    out = np.empty(K)
    for k in range(K):
        cols = [np.asarray(P[:, k], np.float64) for P in Ps]
        ys = [np.asarray(Y[:, k]).astype(bool) for Y in Ys]
        cand = np.unique(np.concatenate(cols))[::-1]          # distinct cutpoints, descending
        npos = [int(y.sum()) for y in ys]
        if sum(npos) == 0:
            out[k] = float(_above(cand[0]))
            continue
        obj = np.zeros(len(cand))
        for j in range(J):
            if npos[j] == 0:
                continue
            a = np.sort(cols[j])
            p = np.sort(cols[j][ys[j]])
            cnt = len(a) - np.searchsorted(a, cand, "left")   # predicted positives at score >= t
            tp = npos[j] - np.searchsorted(p, cand, "left")
            obj += w[j] * (2 * tp / (cnt + npos[j]))
        out[k] = float(cand[int(np.argmax(obj))])
    return out


def shared_objective(Ps, Ys, theta, w=None):
    """sum_j w_j macroF1_j(theta), the quantity fit_shared_exact maximises label by label."""
    J = len(Ps)
    w = np.full(J, 1.0 / J) if w is None else np.asarray(w, float)
    return float(sum(w[j] * macro_f1(np.asarray(Ps[j], np.float64), np.asarray(Ys[j]), theta) for j in range(J)))


def fit_grid(P, Y, grid=GRID):
    """Best of the 19-point grid per label (first best on ties). The paper fitted theta_1 and
    the validation rule this way."""
    P = np.asarray(P, np.float64)
    _check_PY(P, Y, "fit_grid")
    Y = np.asarray(Y).astype(int)
    th = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        f = [macro_f1(P[:, [k]], Y[:, [k]], np.array([g])) for g in grid]
        th[k] = grid[int(np.argmax(f))]
    return th
