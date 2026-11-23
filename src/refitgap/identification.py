"""The limits of rank summaries and archived counts, in exact fractions.

identification_example(): fourteen records, three scorers with equal AUROC (6/11) and average
precision (5/18), equal deficits (1/26), one all mismatch and one all attainable gap.
f1_bounds_from_counts(): the range of best single-threshold F1 compatible with one confusion
table. Both are in-sample statements, with no cross-fitting and no sampling error."""
from fractions import Fraction

import numpy as np

# ------------------------------------------------------------------ rank arithmetic ---


def f1_top(pos_ranks, k):
    """F1 with the top k ranks predicted positive; 0 when k and the positives are both 0."""
    tp = sum(1 for r in pos_ranks if r <= k)
    den = k + len(pos_ranks)
    return Fraction(2 * tp, den) if den else Fraction(0)


def best_f1(pos_ranks, n):
    """The best F1 over every single threshold (every top-k, k = 0..n) on these records."""
    return max(f1_top(pos_ranks, k) for k in range(n + 1))


def auroc(pos_ranks, n):
    """AUROC (no ties; rank 1 is the top)."""
    P = sorted(pos_ranks)
    neg = n - len(P)
    above = sum(sum(1 for q in range(r + 1, n + 1) if q not in P) for r in P)
    return Fraction(above, len(P) * neg)


def average_precision(pos_ranks):
    """Average precision (no ties)."""
    P = sorted(pos_ranks)
    return sum(Fraction(i + 1, r) for i, r in enumerate(P)) / len(P)


def rank_scores(n, selected, threshold=0.5):
    """Scores 1..n ranks apart such that exactly the top `selected` ranks reach `threshold`
    (rank r gets threshold + (selected + 1/2 - r) / (2(n + 1)); all in (0, 1) for threshold 1/2)."""
    r = np.arange(1, n + 1, dtype=np.float64)
    return threshold + (selected + 0.5 - r) / (2.0 * (n + 1))


# ------------------------------------------------------------------ the example ---
N_RECORDS = 14
SCORERS = {                       # positive ranks (1 = top), records the carried rule selects
    "reference": dict(positives=(4, 8, 9), selected=9),
    "final A": dict(positives=(4, 8, 9), selected=10),     # same ranking, rule takes one more record
    "final B": dict(positives=(5, 6, 10), selected=10),    # different ranking
}


def identification_example():
    """() -> dict of exact Fractions: per scorer the carried and best F1, AUROC and average
    precision; per final scorer the deficit, attainable gap and change in mismatch; plus float
    scores and a threshold of 1/2 that realise the example."""
    out = dict(n_records=N_RECORDS, n_labels=1, threshold=0.5, scorers={}, decomposition={},
               note=("in-sample: the best F1 is fitted and scored on the same fourteen records; no cross-fitting, "
                     "no sampling uncertainty"))
    for name, s in SCORERS.items():
        P = s["positives"]
        y = np.zeros(N_RECORDS, int)
        y[[r - 1 for r in P]] = 1
        out["scorers"][name] = dict(positives=P, selected=s["selected"], carried_f1=f1_top(P, s["selected"]),
                                    best_f1=best_f1(P, N_RECORDS), auroc=auroc(P, N_RECORDS),
                                    average_precision=average_precision(P),
                                    scores=rank_scores(N_RECORDS, s["selected"]), labels=y)
    ref = out["scorers"]["reference"]
    for name in ("final A", "final B"):
        f = out["scorers"][name]
        D = ref["carried_f1"] - f["carried_f1"]
        A = ref["best_f1"] - f["best_f1"]
        out["decomposition"][name] = dict(deficit=D, attainable_gap=A, mismatch_change=D - A)
    return out


# ------------------------------------------------------------------ counts ---
def _f1(tp, pred, pos):
    den = pred + pos
    return Fraction(2 * tp, den) if den else Fraction(0)


def f1_bounds_from_counts(tp, fp, fn, tn, exact=False):
    """(tp, fp, fn, tn) at one threshold -> (lower, upper) sharp bounds on the best F1 over all
    thresholds and all rankings compatible with the table (Fractions with exact=True):
        lower = max(2tp / (2tp + fp + fn), 2P / (2P + N)),  upper = max(2tp / (2tp + fn), 2P / (2P + fp))
    with P = tp + fn, N = fp + tn. A bound on what the counts identify, not a confidence interval."""
    vals = [int(v) for v in (tp, fp, fn, tn)]
    if any(v < 0 for v in vals) or any(int(v) != v for v in (tp, fp, fn, tn)):
        raise ValueError("counts must be non-negative integers")
    tp, fp, fn, tn = vals
    P, N = tp + fn, fp + tn
    if P == 0:
        lo = hi = Fraction(0)
    else:
        lo = max(_f1(tp, tp + fp, P), _f1(P, P + N, P))
        hi = max(_f1(tp, tp, P), _f1(P, P + fp, P))
    return (lo, hi) if exact else (float(lo), float(hi))
