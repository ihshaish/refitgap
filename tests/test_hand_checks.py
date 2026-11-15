"""Known answers worked by hand, an independent
brute-force evaluator, record alignment, ties, missing labels, labels with no positive,
score-only identification, and the absence of any retraining recommendation in outputs.

The hand examples do not call the package to obtain their expected values: every expected
number below is written out from the arithmetic in the comments. The brute-force evaluator
uses sklearn.metrics.f1_score and a plain search over all cutpoints, not the package fitters.
Seconds to run."""
from pathlib import Path

import numpy as np
import pytest

import refitgap as rg
from refitgap import crossfit, fitters

EXACT = 1e-12


# ------------------------------------------------------------- hand examples ---
def test_hand_f1_and_exact_fit():
    # scores 0.9 0.8 0.3 0.1, labels 1 0 1 0
    P = np.array([[0.9], [0.8], [0.3], [0.1]])
    Y = np.array([[1], [0], [1], [0]])
    # at 0.5: predicted {0.9, 0.8}, TP 1, 2*1/(2+2) = 0.5
    assert fitters.macro_f1(P, Y, np.array([0.5])) == 0.5
    # cutpoints 0.9: 2/(1+2) = 2/3; 0.8: 2/(2+2) = 1/2; 0.3: 4/(3+2) = 4/5; 0.1: 4/(4+2) = 2/3
    assert fitters.fit_exact(P, Y)[0] == 0.3
    assert abs(fitters.macro_f1(P, Y, np.array([0.3])) - 0.8) < EXACT


def test_hand_ties():
    # a tie AT the threshold is predicted positive (score >= theta)
    P = np.array([[0.5], [0.5], [0.2]])
    Y = np.array([[1], [0], [0]])
    # at 0.5: predicted both tied records, TP 1, 2/(2+1) = 2/3
    assert abs(fitters.macro_f1(P, Y, np.array([0.5])) - 2 / 3) < EXACT
    # tied scores are never split: cutpoints 0.5 -> 2/3, 0.2 -> 2/(3+1) = 1/2; so 0.5 and F1 2/3, not 1
    assert fitters.fit_exact(P, Y)[0] == 0.5
    # equal F1 at two cutpoints goes to the higher one:
    # scores 0.9 0.8 0.7 0.6, labels 1 0 0 1: 0.9 -> 2/3, 0.8 -> 1/2, 0.7 -> 2/5, 0.6 -> 4/6 = 2/3
    P = np.array([[0.9], [0.8], [0.7], [0.6]])
    Y = np.array([[1], [0], [0], [1]])
    assert fitters.fit_exact(P, Y)[0] == 0.9
    assert fitters.fit_shared_exact([P], [Y])[0] == 0.9


def test_hand_zero_positive_label():
    # label 1 has no positive. F1 is 0 whatever is predicted (0 when nothing is predicted, by
    # the zero_division = 0 convention), so it still counts in the macro mean.
    P = np.array([[0.9, 0.2], [0.8, 0.6], [0.3, 0.1], [0.1, 0.4]])
    Y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    th = fitters.fit_exact(P, Y)
    assert th[0] == 0.3
    assert th[1] == 0.6 + 1e-9                     # above every score: predicts nothing
    # label 0 at 0.3 is 4/5, label 1 is 0: macro (0.8 + 0) / 2 = 0.4
    assert abs(fitters.macro_f1(P, Y, th) - 0.4) < EXACT
    assert fitters.f1_vec(P, Y, th)[1] == 0.0


def test_hand_shared_rule_and_weights():
    # period a: 0.9 (1), 0.1 (0); period b: 0.6 (0), 0.4 (1). Mean-of-periods F1 per cutpoint:
    #   0.9: a 1, b 0 (nothing predicted)                  -> 1/2
    #   0.6: a 1, b 0 (0.6 predicted, TP 0)                -> 1/2
    #   0.4: a 1, b 2/(2+1) = 2/3                          -> 5/6
    #   0.1: a 2/(2+1) = 2/3, b 2/3                        -> 2/3
    Pa, Ya = np.array([[0.9], [0.1]]), np.array([[1], [0]])
    Pb, Yb = np.array([[0.6], [0.4]]), np.array([[0], [1]])
    assert fitters.fit_shared_exact([Pa, Pb], [Ya, Yb])[0] == 0.4
    assert abs(fitters.shared_objective([Pa, Pb], [Ya, Yb], np.array([0.4])) - 5 / 6) < EXACT
    # weight 0 on b: a alone scores 1 at 0.9, 0.6 and 0.4; the highest maximiser is 0.9
    assert fitters.fit_shared_exact([Pa, Pb], [Ya, Yb], w=[1.0, 0.0])[0] == 0.9


def _hand_stream():
    """Two periods of four records, one label, seed 0, one partition. The partition rule gives
    (default_rng(7000 + 100*0 + j).permutation(4)) period 0: [1, 2, 0, 3], period 1: [3, 0, 1, 2],
    so the halves are {1, 2} | {0, 3} and {3, 0} | {1, 2}. Checked below against numpy."""
    final = [np.array([[0.9], [0.8], [0.3], [0.1]]), np.array([[0.6], [0.4], [0.7], [0.2]])]
    ref = [np.array([[0.9], [0.1], [0.8], [0.2]]), np.array([[0.9], [0.8], [0.1], [0.2]])]
    Ys = [np.array([[1], [0], [1], [0]]), np.array([[1], [1], [0], [0]])]
    return final, ref, Ys


def test_hand_crossfit_levels_and_bridge():
    final, ref, Ys = _hand_stream()
    assert list(np.random.default_rng(7000).permutation(4)) == [1, 2, 0, 3]
    assert list(np.random.default_rng(7001).permutation(4)) == [3, 0, 1, 2]
    rules = {"inherited": np.array([0.5])}
    ev = crossfit.evaluate(final, Ys, 0, rules=rules, splits=1)
    pp = ev.per_period()
    # period 0, own-period rule.
    #   fit {1,2} = 0.8 (0), 0.3 (1): 0.8 -> 0, 0.3 -> 2/3; theta 0.3; score {0,3} = 0.9 (1), 0.1 (0): F1 1
    #   fit {0,3}: 0.9 -> 1; theta 0.9; score {1,2}: nothing predicted, one positive: F1 0
    #   V_sep = (1 + 0) / 2 = 0.5
    # period 1, own-period rule.
    #   fit {3,0} = 0.2 (0), 0.6 (1): 0.6 -> 1; theta 0.6; score {1,2} = 0.4 (1), 0.7 (0): 0.7 predicted, TP 0: 0
    #   fit {1,2}: 0.7 -> 0, 0.4 -> 2/3; theta 0.4; score {3,0}: 0.6 predicted, TP 1: 1
    #   V_sep = 0.5
    # shared rule (both periods, uniform), direction 0 (fit halves {1,2} and {3,0}):
    #   cutpoints 0.8: 0, 0 -> 0; 0.6: 0, 1 -> 1/2; 0.3: 2/3, 1 -> 5/6; 0.2: 2/3, 2/3 -> 2/3; theta 0.3
    #   scored: period 0 {0,3}: 1; period 1 {1,2}: both predicted, TP 1: 2/3
    # direction 1 (fit halves {0,3} and {1,2}):
    #   cutpoints 0.9: 1, 0 -> 1/2; 0.7: 1, 0 -> 1/2; 0.4: 1, 2/3 -> 5/6; 0.1: 2/3, 2/3 -> 2/3; theta 0.4
    #   scored: period 0 {1,2}: 0.8 predicted, TP 0: 0; period 1 {3,0}: 0.6 predicted, TP 1: 1
    #   V_shared: period 0 (1 + 0)/2 = 1/2; period 1 (2/3 + 1)/2 = 5/6
    # inherited rule 0.5:
    #   period 0 halves: {0,3}: 1; {1,2}: 0.8 predicted, TP 0: 0 -> 1/2; full: 0.9, 0.8 predicted, TP 1: 1/2
    #   period 1 halves: {1,2}: 0.7 predicted, TP 0: 0; {3,0}: 0.6, TP 1: 1 -> 1/2; full: 0.6, 0.7, TP 1: 2/4
    exp = {0: (0.5, 0.5, 0.5, 0.5), 1: (0.5, 5 / 6, 0.5, 0.5)}
    for j, (vs, vsh, uh, uf) in exp.items():
        assert abs(pp[j]["V_sep"] - vs) < EXACT and abs(pp[j]["V_shared"] - vsh) < EXACT
        assert abs(pp[j]["U_half"]["inherited"] - uh) < EXACT and abs(pp[j]["U_full"]["inherited"] - uf) < EXACT
    # both periods: V_sep 1/2, V_shared 2/3, C = -1/6 (the shared fit sees twice the records, so
    # on halves this small C can be negative), U_half 1/2, R = 0, M = 2/3 - 1/2 = 1/6
    s = ev.summary()
    r = s["rules"]["inherited"]
    for got, want in ((s["V_sep"], 0.5), (s["V_shared"], 2 / 3), (s["C"], -1 / 6), (r["R"], 0.0), (r["M"], 1 / 6)):
        assert abs(got - want) < EXACT
    # reference models (a different perfect scorer per period).
    #   period 0: fit {1,2} = 0.1 (0), 0.8 (1): theta 0.8; {0,3} = 0.9 (1), 0.2 (0): 1
    #             fit {0,3}: theta 0.9; {1,2}: 0.8 < 0.9, nothing predicted: 0          -> V_sep 1/2
    #   period 1: fit {3,0} = 0.2 (0), 0.9 (1): theta 0.9; {1,2} = 0.8 (1), 0.1 (0): 0
    #             fit {1,2}: theta 0.8; {3,0}: 0.9 predicted, TP 1: 1                     -> V_sep 1/2
    #   at 0.5 on the full periods both are perfect: U_ref = 1
    # D = U_ref - U_final = 1 - 1/2 = 1/2; A = 1/2 - 1/2 = 0; R = (1/2 - 1/2) - (1/2 - 1) = 1/2
    ev_ref = crossfit.evaluate(ref, Ys, 0, rules=rules, splits=1, single_model=False)
    d = crossfit.decompose(ev_ref, ev, "inherited")
    for k, want in (("D", 0.5), ("A", 0.0), ("R", 0.5), ("U_ref", 1.0), ("U_final", 0.5)):
        assert abs(d[k] - want) < EXACT, k


# ------------------------------------------------- independent brute force ---
def _bf_fit(P, Y):
    """Highest cutpoint maximising sklearn F1, searched over every distinct score and one
    value above them all."""
    from sklearn.metrics import f1_score
    th = []
    for k in range(Y.shape[1]):
        cand = sorted(set(P[:, k].tolist()) | {float(P[:, k].max()) + 1.0}, reverse=True)
        f = [f1_score(Y[:, k], (P[:, k] >= t).astype(int), zero_division=0) for t in cand]
        th.append(cand[int(np.argmax(f))])
    return np.array(th)


def _bf_shared(Ps, Ys):
    from sklearn.metrics import f1_score
    th = []
    for k in range(Ys[0].shape[1]):
        allv = np.concatenate([P[:, k] for P in Ps])
        cand = sorted(set(allv.tolist()) | {float(allv.max()) + 1.0}, reverse=True)
        f = [np.mean([f1_score(Y[:, k], (P[:, k] >= t).astype(int), zero_division=0) for P, Y in zip(Ps, Ys)])
             for t in cand]
        th.append(cand[int(np.argmax(f))])
    return np.array(th)


def _bf_macro(P, Y, th):
    from sklearn.metrics import f1_score
    return float(np.mean([f1_score(Y[:, k], (P[:, k] >= th[k]).astype(int), zero_division=0)
                          for k in range(Y.shape[1])]))


def _bf_levels(Ps, Ys, seed, splits, theta):
    """V_sep, V_shared, U_half per period, written from the documented rule only: partition b of
    period j is the b-th permutation of default_rng(7000 + 100*seed + j); half 0 is perm[:n//2];
    fit on one half, score on the other, both directions; average."""
    N = len(Ps)
    gens = [np.random.default_rng(7000 + 100 * seed + j) for j in range(N)]
    acc = {"sep": np.zeros(N), "sh": np.zeros(N), "u": np.zeros(N)}
    for _ in range(splits):
        perm = [g.permutation(len(Y)) for g, Y in zip(gens, Ys)]
        halves = [(p[:len(p) // 2], p[len(p) // 2:]) for p in perm]
        for d in (0, 1):
            fit = [h[d] for h in halves]
            ev = [h[1 - d] for h in halves]
            ths = _bf_shared([P[f] for P, f in zip(Ps, fit)], [Y[f] for Y, f in zip(Ys, fit)])
            for j in range(N):
                Pe, Ye = Ps[j][ev[j]], Ys[j][ev[j]]
                acc["sep"][j] += _bf_macro(Pe, Ye, _bf_fit(Ps[j][fit[j]], Ys[j][fit[j]])) / (2 * splits)
                acc["sh"][j] += _bf_macro(Pe, Ye, ths) / (2 * splits)
                acc["u"][j] += _bf_macro(Pe, Ye, theta) / (2 * splits)
    return acc


def test_independent_bruteforce_agrees():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(11)
    Ps, Ys = [], []
    for j, n in enumerate((23, 30, 17)):
        Y = (rng.random((n, 3)) < [0.3, 0.1, 0.5]).astype(int)
        Y[:, 1] = 0 if j == 1 else Y[:, 1]                       # a period with no positive of label 1
        P = np.round(1 / (1 + np.exp(-(1.2 * Y + rng.normal(0, 1, Y.shape) + 0.2 * j))), 1)   # heavy ties
        Ps.append(P); Ys.append(Y)
    theta = np.array([0.5, 0.6, 0.4])
    ev = crossfit.evaluate(Ps, Ys, 3, rules={"t": theta}, splits=4)
    bf = _bf_levels(Ps, Ys, 3, 4, theta)
    pp = ev.per_period()
    worst = max(max(abs(pp[j]["V_sep"] - bf["sep"][j]), abs(pp[j]["V_shared"] - bf["sh"][j]),
                    abs(pp[j]["U_half"]["t"] - bf["u"][j])) for j in range(3))
    assert worst < EXACT, worst


# ---------------------------------------------------------------- alignment ---
def test_alignment_by_record_id():
    rng = np.random.default_rng(2)
    ids = [f"r{i}" for i in range(40)]
    Y = (rng.random((40, 2)) < 0.4).astype(int)
    P = rng.random((40, 2))
    order = rng.permutation(40)
    ids_other = [ids[i] for i in order]                          # another pipeline's row order
    P_other, Y_other = P[order], Y[order]
    P_back, Y_back = rg.align_to(ids, ids_other, P_other, Y_other)
    assert np.array_equal(P_back, P) and np.array_equal(Y_back, Y)
    # extra rows are dropped, missing or duplicated ids are refused
    assert rg.align_to(ids[:10], ids_other, P_other).shape == (10, 2)
    with pytest.raises(ValueError, match="no row"):
        rg.align_to(ids + ["new"], ids_other, P_other)
    with pytest.raises(ValueError, match="duplicates"):
        rg.align_to(ids, ids_other[:-1] + [ids_other[0]], P_other)
    # evaluations of two models on the same records in different row orders are not paired
    a = crossfit.evaluate([P, P], [Y, Y], 0, splits=2)
    b = crossfit.evaluate([P, P_other], [Y, Y_other], 0, splits=2)
    with pytest.raises(ValueError, match="record order"):
        crossfit.decompose(a, b, "x")
    # after alignment they pair
    c = crossfit.evaluate([P, P_back], [Y, Y_back], 0, splits=2)
    assert np.array_equal(a.sep, c.sep)


def test_scores_misaligned_with_their_own_labels_are_not_detectable():
    """Documented limit: the label hash pairs two evaluations' labels; a model whose score rows
    were permuted against its own labels is evaluated as given. Align by record id first."""
    rng = np.random.default_rng(3)
    Y = (rng.random((60, 2)) < 0.4).astype(int)
    P = 1 / (1 + np.exp(-(2 * Y + rng.normal(0, 1, Y.shape))))
    good = crossfit.evaluate([P], [Y], 0, splits=3)
    bad = crossfit.evaluate([P[rng.permutation(60)]], [Y], 0, splits=3)
    assert good.label_hash == bad.label_hash
    assert bad.summary()["V_sep"] < good.summary()["V_sep"]


# ----------------------------------------------------------- missing labels ---
def test_missing_labels_are_refused_and_can_be_dropped():
    rng = np.random.default_rng(4)
    Y = (rng.random((30, 2)) < 0.4).astype(float)
    P = rng.random((30, 2))
    Y[5, 1] = np.nan
    for bad in (Y, np.where(np.isnan(Y), -1, Y)):
        with pytest.raises(ValueError, match="0/1"):
            crossfit.evaluate([P], [bad], 0, splits=2)
    ids, Yk, Pk = rg.drop_unlabelled(list(range(30)), Y, P)
    assert len(ids) == 29 and 5 not in ids and Yk.dtype.kind == "i"
    crossfit.evaluate([Pk], [Yk], 0, splits=2)


def test_zero_positive_period_and_label_in_crossfit():
    rng = np.random.default_rng(5)
    Ys = [(rng.random((40, 3)) < 0.3).astype(int) for _ in range(3)]
    for Y in Ys:
        Y[:, 2] = 0                                              # label 2 never positive
    Ys[1][:] = 0                                                 # period 1: no positive at all
    Ps = [rng.random((40, 3)) for _ in range(3)]
    ev = crossfit.evaluate(Ps, Ys, 0, rules={"t": np.full(3, 0.5)}, splits=3)
    pp = ev.per_period()
    assert pp[1]["V_sep"] == 0.0 and pp[1]["V_shared"] == 0.0 and pp[1]["U_full"]["t"] == 0.0
    # label 2 contributes 0 to every period's macro-F1, so no level can exceed 2/3
    assert all(pp[j]["V_sep"] <= 2 / 3 + EXACT for j in range(3))
    assert ev.summary()["rules"]["t"]["identity_residual"] < EXACT


# ------------------------------------------------ score-only identification ---
def test_levels_depend_on_scores_only():
    """Two scorers with different internals (different features and weights) that emit the
    same score matrices give bitwise identical evaluations: the evaluator identifies only
    what is in the scores."""
    rng = np.random.default_rng(6)
    Y = (rng.random((80, 2)) < 0.4).astype(int)
    X1 = Y + rng.normal(0, 0.8, Y.shape)                         # scorer 1: two features, identity head
    s1 = 1 / (1 + np.exp(-X1))
    # scorer 2: seven features (two halved copies of each signal plus three noise columns) and a
    # head that adds the copies back; x/2 + x/2 == x exactly in binary floating point
    X2 = np.column_stack([X1 / 2, X1 / 2, rng.normal(0, 1, (80, 3))])
    W2 = np.zeros((7, 2))
    W2[0, 0] = W2[2, 0] = W2[1, 1] = W2[3, 1] = 1.0
    s2 = 1 / (1 + np.exp(-(X2 @ W2)))
    assert np.array_equal(s1, s2)
    a = crossfit.evaluate([s1, s1], [Y, Y], 0, splits=3)
    b = crossfit.evaluate([s2, s2], [Y, Y], 0, splits=3)
    assert np.array_equal(a.sep, b.sep) and np.array_equal(a.shared, b.shared)


def test_attainable_gap_does_not_identify_lost_representation():
    """Counterexample to reading A as destroyed representation information: the final model
    keeps every informative feature, but its output head no longer uses them for label 0.
    A from the stored scores is large; a linear probe on the retained features recovers the
    reference level. Stored scores alone cannot tell this apart from features that were lost."""
    rng = np.random.default_rng(7)
    n, K = 400, 2
    Y = (rng.random((n, K)) < 0.3).astype(int)
    feats = 2.0 * Y + rng.normal(0, 1, (n, K))                  # retained features, informative for both labels
    ref = 1 / (1 + np.exp(-feats))                              # reference head: uses them
    head = feats.copy()
    head[:, 0] = rng.normal(0, 1, n)                            # final head: ignores feature 0
    final = 1 / (1 + np.exp(-head))
    rules = {"t": np.full(K, 0.5)}
    ev_ref = crossfit.evaluate([ref], [Y], 0, rules=rules, splits=10, single_model=False)
    ev_fin = crossfit.evaluate([final], [Y], 0, rules=rules, splits=10)
    d = crossfit.decompose(ev_ref, ev_fin, "t")
    assert d["A"] > 0.1                                        # label 0 alone: about 0.29 of F1
    # linear probe on the retained features, fitted on an independent draw from the same process
    # (no evaluation record is used to fit it), then applied to the evaluation records
    Y_fit = (rng.random((n, K)) < 0.3).astype(int)
    feats_fit = 2.0 * Y_fit + rng.normal(0, 1, (n, K))
    W = np.linalg.lstsq(np.column_stack([feats_fit, np.ones(n)]), Y_fit, rcond=None)[0][:K]
    probe = feats @ W
    ev_probe = crossfit.evaluate([probe], [Y], 0, splits=10)
    assert abs(ev_probe.summary()["V_sep"] - ev_ref.summary()["V_sep"]) < 0.05


# ------------------------------------------------ no retraining recommendation ---
def test_no_retrain_recommendation_in_outputs(capsys):
    src = Path(rg.__file__).resolve().parent
    for p in src.glob("*.py"):
        assert "retrain" not in p.read_text().lower(), p.name
    from refitgap import example
    example.main()
    out = capsys.readouterr().out.lower()
    assert "retrain" not in out and "recommend" not in out
    final, ref, Ys = _hand_stream()
    rules = {"inherited": np.array([0.5])}
    d = crossfit.decompose(crossfit.evaluate(ref, Ys, 0, rules=rules, splits=1, single_model=False),
                           crossfit.evaluate(final, Ys, 0, rules=rules, splits=1), "inherited")
    assert not any("retrain" in str(k).lower() or "retrain" in str(v).lower() for k, v in d.items())
