"""The limited refit, the previous period's rule, rule_lead and per-label levels (seconds)."""
import numpy as np
import pytest

import refitgap as rg
from refitgap import crossfit


def synth(seed=0, N=4, K=4, n=(160, 141, 120, 133), sep=1.5, shift=0.0):
    rng = np.random.default_rng(seed)
    Ps, Ys = [], []
    for j in range(N):
        Y = (rng.random((n[j], K)) < rng.uniform(0.1, 0.4, K)).astype(int)
        z = sep * Y + rng.normal(0, 1, Y.shape) + 0.2 * j + shift
        Ps.append((1 / (1 + np.exp(-z))).astype(np.float16).astype(np.float64))
        Ys.append(Y)
    return Ps, Ys


def other_model(Ps, seed=1, noise=0.6):
    rng = np.random.default_rng(seed)
    return [np.clip(P + noise * rng.normal(0, 0.1, P.shape), 0, 1).astype(np.float16).astype(np.float64)
            for P in Ps]


def best_cut(p, y):
    """Direct: the highest score cutpoint maximising F1 on (p, y); +inf when y has no positive."""
    if y.sum() == 0:
        return np.inf
    best, arg = -1.0, None
    for t in np.unique(p):
        pred = p >= t
        f = 2 * (pred & (y == 1)).sum() / (pred.sum() + y.sum())
        if f >= best:
            best, arg = f, t
    return arg


def f1(p, y, t):
    pred = p >= t
    den = pred.sum() + y.sum()
    return 2 * (pred & (y == 1)).sum() / den if den else 0.0


def direct_level(P, Y, seed, j, splits, fit_rows):
    """Mean over partitions and directions of macro-F1 on the scoring half; fit_rows(b, d, F)
    returns (P_fit, Y_fit) for that partition, direction and fitting half."""
    n, K = Y.shape
    h = n // 2
    rng = np.random.default_rng(7000 + 100 * seed + j)
    vals = []
    for b in range(splits):
        perm = rng.permutation(n)
        for d in (0, 1):
            F, S = (perm[:h], perm[h:]) if d == 0 else (perm[h:], perm[:h])
            Pf, Yf = fit_rows(b, d, F)
            vals.append(np.mean([f1(P[S, k], Y[S, k], best_cut(Pf[:, k], Yf[:, k])) for k in range(K)]))
    return float(np.mean(vals))


SPLITS = 6


def test_limited_refit_equals_full_refit_when_n_covers_the_half():
    Ps, Ys = synth()
    big = max(len(Y) for Y in Ys)
    ev = rg.evaluate(Ps, Ys, 3, rules={"c": np.full(4, 0.5)}, splits=SPLITS, refit_n=big)
    assert np.array_equal(ev.limited, ev.sep)
    assert ev.summary()["V_limited"] == ev.summary()["V_sep"]
    # one record short of the larger half of period 1 (141 records: halves 70 and 71)
    d = rg.limited_draws(141, 3, 1, 71, SPLITS)
    assert all(len(r[0]) == 70 and np.array_equal(r[0], np.arange(70)) and len(r[1]) == 71 for r in d)


def test_limited_draws_are_reproducible_and_shared_by_models():
    a = rg.limited_draws(160, 2, 1, 30, SPLITS)
    b = rg.limited_draws(160, 2, 1, 30, SPLITS)
    assert all(np.array_equal(x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    rng = np.random.default_rng(9000 + 100 * 2 + 1)                 # one generator, b then d
    for row in a:
        for d in (0, 1):
            assert np.array_equal(row[d], rng.choice(80, 30, replace=False))
    assert len({tuple(x) for r in a for x in r}) == 2 * SPLITS
    p = rg.previous_draws(141, 2, 1, 50, SPLITS)
    rng = np.random.default_rng(11000 + 100 * 2 + 1)
    assert all(np.array_equal(x, rng.choice(141, 50, replace=False)) for x in p)
    with pytest.raises(ValueError):
        rg.previous_draws(40, 2, 1, 50, SPLITS)
    with pytest.raises(ValueError):
        rg.limited_draws(160, 2, 1, 0, SPLITS)


def test_limited_refit_and_previous_rule_match_a_direct_computation():
    Ps, Ys = synth(4)
    seed, n_lim, n_prev = 1, 40, 60
    ev = rg.evaluate(Ps[1:], Ys[1:], seed, period_ids=[1, 2, 3], splits=SPLITS, refit_n=n_lim,
                     previous=list(zip(Ps[:-1], Ys[:-1])), previous_n=n_prev)
    for c, j in enumerate([1, 2, 3]):
        P, Y = Ps[j], Ys[j]
        h = len(Y) // 2
        lim = np.random.default_rng(9000 + 100 * seed + j)
        draws = {(b, d): lim.choice(h if d == 0 else len(Y) - h, n_lim, replace=False)
                 for b in range(SPLITS) for d in (0, 1)}
        want = direct_level(P, Y, seed, j, SPLITS, lambda b, d, F: (P[F[draws[b, d]]], Y[F[draws[b, d]]]))
        assert abs(ev.per_period()[j]["V_limited"] - want) < 1e-12
        pr = np.random.default_rng(11000 + 100 * seed + j)
        sel = [pr.choice(len(Ys[j - 1]), n_prev, replace=False) for _ in range(SPLITS)]
        want = direct_level(P, Y, seed, j, SPLITS, lambda b, d, F: (Ps[j - 1][sel[b]], Ys[j - 1][sel[b]]))
        assert abs(ev.per_period()[j]["U_previous"] - want) < 1e-12
    assert ev.summary()["U_previous"] is not None


def test_new_options_leave_the_existing_levels_unchanged():
    Ps, Ys = synth(5)
    rules = {"c": np.full(4, 0.5)}
    e0 = rg.evaluate(Ps[1:], Ys[1:], 0, period_ids=[1, 2, 3], rules=rules, splits=SPLITS)
    e1 = rg.evaluate(Ps[1:], Ys[1:], 0, period_ids=[1, 2, 3], rules=rules, splits=SPLITS, refit_n=30,
                     previous=[None, (Ps[1], Ys[1]), (Ps[2], Ys[2])], previous_n=40, per_label=True)
    assert np.array_equal(e0.sep, e1.sep) and np.array_equal(e0.shared, e1.shared)
    assert np.array_equal(e0.U_half["c"], e1.U_half["c"]) and np.array_equal(e0.U_full["c"], e1.U_full["c"])
    assert e1.per_period()[1]["U_previous"] is None and e1.summary()["U_previous"] is None
    assert e1.summary([2, 3])["U_previous"] is not None


def _pair(seed, Ps, Ys, Qs, **kw):
    prevA = list(zip(Ps[:-1], Ys[:-1]))
    prevB = list(zip(Qs[:-1], Ys[:-1]))
    ids = list(range(1, len(Ps)))
    ea = rg.evaluate(Ps[1:], Ys[1:], seed, ids, rules={"c": np.full(4, 0.5)}, splits=SPLITS, previous=prevA, **kw)
    eb = rg.evaluate(Qs[1:], Ys[1:], seed, ids, rules={"c": np.full(4, 0.5)}, splits=SPLITS, previous=prevB, **kw)
    return ea, eb


def test_identical_models_lead_by_exactly_zero():
    Ps, Ys = synth(6)
    ea, eb = zip(*[_pair(s, Ps, Ys, Ps, refit_n=30, previous_n=40) for s in range(3)])
    r = rg.rule_lead(dict(enumerate(ea)), dict(enumerate(eb)), "previous")
    for t, v in r["terms"].items():
        assert all(x == 0.0 for x in v["per_seed"].values()), t
    assert set(r["terms"]) == {"lead_at_rule", "lead_after_refit", "rule_minus_refit", "lead_after_limited_refit",
                               "rule_minus_limited_refit"}


def test_rule_lead_matches_the_levels_and_refuses_unpaired_inputs():
    Ps, Ys = synth(7)
    Qs = other_model(Ps)
    A, B = {}, {}
    for s in range(3):
        A[s], B[s] = _pair(s, Ps, Ys, Qs, refit_n=30, previous_n=40)
    r = rg.rule_lead(A, B, "previous", names=("a", "b"))
    for s in range(3):
        sa, sb = A[s].summary(), B[s].summary()
        assert abs(r["terms"]["lead_at_rule"]["per_seed"][s] - (sa["U_previous"] - sb["U_previous"])) < 1e-12
        assert abs(r["terms"]["lead_after_refit"]["per_seed"][s] - (sa["V_sep"] - sb["V_sep"])) < 1e-12
        assert abs(r["terms"]["lead_after_limited_refit"]["per_seed"][s] - (sa["V_limited"] - sb["V_limited"])) < 1e-12
        assert r["terms"]["rule_minus_refit"]["per_seed"][s] == (r["terms"]["lead_at_rule"]["per_seed"][s]
                                                               - r["terms"]["lead_after_refit"]["per_seed"][s])
    # a fixed rule on the halves equals method_contrast(halves=True) at that rule
    fx = rg.rule_lead(A, B, "c")
    mc = rg.method_contrast(A, B, "c", halves=True)
    for s in range(3):
        assert abs(fx["terms"]["lead_at_rule"]["per_seed"][s] - mc["terms"]["lead_at_carried_rule"]["per_seed"][s]) < 1e-12
        assert abs(fx["terms"]["lead_after_refit"]["per_seed"][s] - mc["terms"]["lead_after_refit"]["per_seed"][s]) < 1e-12
    # the previous period's records must be the same for both methods
    Y0 = Ys[0].copy(); Y0[0] = 1 - Y0[0]
    bad = rg.evaluate(Qs[1:], Ys[1:], 0, [1, 2, 3], splits=SPLITS,
                      previous=[(Qs[0], Y0), (Qs[1], Ys[1]), (Qs[2], Ys[2])], previous_n=40)
    with pytest.raises(ValueError):
        rg.rule_lead({0: A[0]}, {0: bad}, "previous")
    with pytest.raises(ValueError):
        rg.rule_lead({0: A[0]}, {0: rg.evaluate(Qs[1:], Ys[1:], 0, [1, 2, 3], splits=SPLITS)}, "previous")


def test_per_label_levels_average_to_the_macro_levels_and_parts_sum_to_the_lead():
    Ps, Ys = synth(8)
    Qs = other_model(Ps, 2)
    A, B = {}, {}
    for s in range(2):
        A[s], B[s] = _pair(s, Ps, Ys, Qs, refit_n=30, previous_n=40, per_label=True)
    sm, ls = A[0].summary(), A[0].label_summary()
    assert abs(ls["V_sep"].mean() - sm["V_sep"]) < 1e-12
    assert abs(ls["V_limited"].mean() - sm["V_limited"]) < 1e-12
    assert abs(ls["U_previous"].mean() - sm["U_previous"]) < 1e-12
    assert abs(ls["U_half"]["c"].mean() - sm["rules"]["c"]["U_half"]) < 1e-12
    assert abs(ls["U_full"]["c"].mean() - sm["rules"]["c"]["U_full"]) < 1e-12
    w = [1.0, 2.0, 3.0]
    assert abs(A[0].label_summary(weights=w)["V_sep"].mean() - A[0].summary(weights=w)["V_sep"]) < 1e-12
    # per partition too: the label axis averages to the stored macro level exactly
    assert np.array_equal(A[0].label_levels["V_sep"].mean(-1), A[0].sep)
    r = rg.rule_lead(A, B, "previous", by_label=True)
    for s in range(2):
        for t in ("lead_at_rule", "lead_after_refit", "lead_after_limited_refit"):
            assert abs(sum(r["label_parts"][s][t]) - r["terms"][t]["per_seed"][s]) < 1e-12
    with pytest.raises(ValueError):
        rg.evaluate(Ps, Ys, 0, splits=2).label_summary()
