"""Known-answer fixtures: inputs whose split B = A + dR is known exactly by hand, run through the
public API. Self-contained (refitgap, numpy, fractions, pytest; no conftest, no score files), so the
file can be copied out of the repository and run against an installed wheel.

1. A hand-built fixed calibration/evaluation split with two labels: label 0 loses only at the
   carried rule (all change in mismatch), label 1 loses its ranking (all attainable gap).
2. The paper's cross-fitted estimator (200 split-half partitions) on a final scorer that is a
   strictly increasing map of the reference: the refitted levels are equal partition by partition,
   so A = 0 exactly and the whole deficit is mismatch, B = dR, with B known by hand.
3. The fourteen-record non-identification example (identification.py), recomputed through the
   package's own fitters and F1: deficit 1/26 for both finals, all mismatch for A, all attainable
   gap for B, with equal AUROC and AP.
4. The eight-record tiny example against its hand values (API.txt works each one).
"""
from fractions import Fraction as F

import numpy as np
import pytest

import refitgap as rg
from refitgap import fitters, tiny


def _close(got, want, tol=1e-12):
    return abs(float(got) - float(want)) <= tol


# ------------------------------------------------------------------ 1. fixed split ---
# 8 calibration records then 10 evaluation records; label 0 and label 1 share the labels.
Y_CAL = [1, 1, 1, 0, 0, 0, 0, 0]
Y_EV = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
REF_CAL = [0.9, 0.8, 0.7, 0.2, 0.15, 0.1, 0.05, 0.01]          # perfectly separated: refit cut 0.7
REF_EV = [0.9, 0.8, 0.75, 0.1, 0.95, 0.15, 0.1, 0.05, 0.02, 0.01]
# label 0 of the final scorer: the reference cubed (same ranking, lower scale)
# label 1 of the final scorer: same calibration, two evaluation positives fall to the bottom
FIN1_EV = [0.9, 0.8, 0.05, 0.1, 0.95, 0.15, 0.1, 0.05, 0.02, 0.01]
RULE = np.array([0.5, 0.5])


def _fixture_fixed():
    y = np.array(Y_CAL + Y_EV)
    Y = np.column_stack([y, y])
    ref = np.array(REF_CAL + REF_EV)
    P_ref = np.column_stack([ref, ref])
    P_fin = np.column_stack([ref ** 3, np.array(REF_CAL + FIN1_EV)])
    cal, ev = np.arange(8), np.arange(8, 18)
    return P_ref, P_fin, Y, cal, ev


def test_fixed_split_known_answer():
    """By hand, on the 10 evaluation records (4 positives):
    reference, both labels: refit cut 0.7 and carried 0.5 select the same four records
        (0.9, 0.8, 0.75 positive; 0.95 negative): TP 3, FP 1, FN 1, F1 = 3/4 at both rules.
    final label 0 (cubed): refit cut 0.343 selects the same four (F1 3/4); carried 0.5 selects
        0.729, 0.512 and 0.857: TP 2, FP 1, FN 2, F1 = 4/7.
    final label 1: refit cut 0.7 and carried 0.5 both select 0.9, 0.8, 0.95: F1 = 4/7.
    Macro: U_ref = V_ref = 3/4, U_fin = 4/7, V_fin = 37/56;
        B = 5/28, A = 5/56, dR = 5/56; per label (5/28, 0, 5/28) and (5/28, 5/28, 0)."""
    P_ref, P_fin, Y, cal, ev = _fixture_fixed()
    r = rg.compare(P_ref, P_fin, Y, RULE, fixed_split={"calibration": cal, "evaluation": ev})
    assert _close(r["carried"]["ref"], F(3, 4)) and _close(r["carried"]["final"], F(4, 7))
    assert _close(r["refitted"]["ref"], F(3, 4)) and _close(r["refitted"]["final"], F(37, 56))
    c = r["components"]
    assert _close(c["deficit"], F(5, 28))
    assert _close(c["attainable_gap"], F(5, 56))
    assert _close(c["mismatch_change"], F(5, 56))
    assert _close(c["deficit"], c["attainable_gap"] + c["mismatch_change"], 1e-15)
    # each label alone
    for k, (B, A, dR) in enumerate(((F(5, 28), 0, F(5, 28)), (F(5, 28), F(5, 28), 0))):
        rk = rg.compare(P_ref[:, [k]], P_fin[:, [k]], Y[:, [k]], RULE[[k]],
                        fixed_split={"calibration": cal, "evaluation": ev})["components"]
        assert _close(rk["deficit"], B) and _close(rk["attainable_gap"], A) and _close(rk["mismatch_change"], dR)
    # the refitted cuts are the ones stated above
    assert np.allclose(fitters.fit_exact(P_ref[cal], Y[cal]), [0.7, 0.7])
    assert np.allclose(fitters.fit_exact(P_fin[cal], Y[cal]), [0.7 ** 3, 0.7])


# ------------------------------------------------------- 2. cross-fitted, A = 0 exactly ---
def _fixture_monotone(n=60, K=3, seed=11):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, K)) < 0.3).astype(int)
    Y[:4] = 1                                       # every label has positives
    P_ref = np.clip(0.5 * Y + rng.normal(0.25, 0.2, (n, K)), 0.001, 0.999)
    P_fin = P_ref ** 2                              # strictly increasing on (0, 1): same ranking
    return P_ref, P_fin, Y


def test_crossfit_monotone_map_known_answer():
    """The refit cut is a score of the calibration half (fitters.fit_exact), and s >= t iff
    s**2 >= t**2, so each partition's refitted F1 is identical for the two scorers: A = 0
    exactly. U (all records, carried 0.5) is computed here by hand in fractions, so
    B = U_ref - U_fin and dR = B are known without the package's estimator."""
    P_ref, P_fin, Y = _fixture_monotone()

    def f1(P, theta):
        vals = []
        for k in range(Y.shape[1]):
            pred = P[:, k] >= theta
            tp = int(np.sum(pred & (Y[:, k] == 1)))
            den = int(pred.sum() + Y[:, k].sum())
            vals.append(F(2 * tp, den) if den else F(0))
        return sum(vals) / len(vals)

    B = f1(P_ref, 0.5) - f1(P_fin, 0.5)
    assert B > 0                                    # the squared scores fall below the carried 0.5
    r = rg.compare(P_ref, P_fin, Y, np.full(3, 0.5), seed=3, splits=200)
    c = r["components"]
    assert c["attainable_gap"] == 0.0
    assert _close(r["refitted"]["ref"], r["refitted"]["final"], 0.0)
    assert _close(c["deficit"], B) and _close(c["mismatch_change"], B)


# ------------------------------------------------ 3. the 14-record non-identification ---
def _f1_frac(scores, y, theta):
    pred = scores >= theta
    tp, den = int(np.sum(pred & (y == 1))), int(pred.sum() + y.sum())
    return F(2 * tp, den) if den else F(0)


def test_identification_example_known_answer():
    e = rg.identification_example()
    s = e["scorers"]
    assert e["n_records"] == 14 and e["n_labels"] == 1
    want = {"reference": (F(1, 2), F(1, 2)), "final A": (F(6, 13), F(1, 2)), "final B": (F(6, 13), F(6, 13))}
    for name, (carried, best) in want.items():
        assert s[name]["carried_f1"] == carried and s[name]["best_f1"] == best
        assert s[name]["auroc"] == F(6, 11) and s[name]["average_precision"] == F(5, 18)
        # recomputed through the package: the carried rule 1/2 and the in-sample exact refit
        P, y = s[name]["scores"], s[name]["labels"]
        assert _f1_frac(P, y, e["threshold"]) == carried
        th = fitters.fit_exact(P[:, None], y[:, None])[0]
        assert _f1_frac(P, y, th) == best
        assert _close(fitters.macro_f1(P[:, None], y[:, None], np.array([e["threshold"]])), carried)
    dA, dB = e["decomposition"]["final A"], e["decomposition"]["final B"]
    assert (dA["deficit"], dA["attainable_gap"], dA["mismatch_change"]) == (F(1, 26), 0, F(1, 26))
    assert (dB["deficit"], dB["attainable_gap"], dB["mismatch_change"]) == (F(1, 26), F(1, 26), 0)
    for d in (dA, dB):
        assert d["deficit"] == d["attainable_gap"] + d["mismatch_change"]


def test_count_bounds_known_answer():
    assert rg.f1_bounds_from_counts(40, 10, 10, 40, exact=True) == (F(4, 5), F(10, 11))
    assert rg.f1_bounds_from_counts(0, 5, 5, 0, exact=True)[0] == F(2, 3)


# --------------------------------------------------------------- 4. the tiny example ---
def test_tiny_example_hand_values():
    res = tiny.tiny_result()
    seen = 0
    for r in res["decomposition"]:
        p = "mean" if str(r["period"]).startswith("mean") else r["period"]
        if p not in tiny.HAND["decomposition"]:
            continue
        for k in ("B", "A", "dR"):
            assert r[k] == pytest.approx(tiny.HAND["decomposition"][p][k], abs=1e-12), (p, k)
        seen += 1
    assert seen == 3
