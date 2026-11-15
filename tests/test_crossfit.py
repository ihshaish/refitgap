"""Fitters, partitions, cross-fitted levels and the decompositions on synthetic scores (seconds)."""
import numpy as np
import pytest

from refitgap import crossfit, fitters, inference



# ------------------------------------------------------------------ fixtures ---
def synth(seed=0, N=4, K=5, n=(120, 90, 151, 64), f16=True, sep=1.5):
    """N periods of K labels; scores rounded to float16 (as the stored runs) so ties occur."""
    rng = np.random.default_rng(seed)
    Ps, Ys = [], []
    for j in range(N):
        Y = (rng.random((n[j], K)) < rng.uniform(0.05, 0.4, K)).astype(int)
        z = sep * Y + rng.normal(0, 1, Y.shape) + 0.3 * j
        P = 1 / (1 + np.exp(-z))
        Ps.append(P.astype(np.float16) if f16 else P)
        Ys.append(Y)
    return Ps, Ys


def brute_shared(Ps, Ys, w):
    """Per label, the weighted mean-of-periods F1 at every candidate cutpoint (all distinct
    scores, plus one above all), computed directly; returns (best objective, highest maximiser)."""
    K = Ys[0].shape[1]
    best, arg = np.empty(K), np.empty(K)
    for k in range(K):
        cand = np.unique(np.concatenate([np.asarray(P[:, k], np.float64) for P in Ps]))
        cand = np.append(cand, cand[-1] + 1.0)
        vals = []
        for t in cand:
            f = 0.0
            for j, (P, Y) in enumerate(zip(Ps, Ys)):
                pred = np.asarray(P[:, k], np.float64) >= t
                y = Y[:, k].astype(bool)
                tp, den = (pred & y).sum(), pred.sum() + y.sum()
                f += w[j] * (2 * tp / den if den else 0.0)
            vals.append(f)
        vals = np.array(vals)
        best[k] = vals.max()
        arg[k] = cand[np.flatnonzero(vals >= vals.max() - 1e-12).max()]
    return best, arg


# ------------------------------------------------------------------- fitters ---
def test_fit_exact_equals_decomposition(DEC):
    for s in range(6):
        Ps, Ys = synth(s)
        Ys[0][:, 2] = 0                                         # a label with no positive
        for P, Y in zip(Ps, Ys):
            a, b = fitters.fit_exact(P, Y), DEC.fit_exact(P, Y)
            assert np.array_equal(a, b)


def test_fit_grid_and_macro_f1_equal_decomposition(DEC):
    Ps, Ys = synth(3)
    for P, Y in zip(Ps, Ys):
        P = P.astype(np.float64)
        assert np.array_equal(fitters.fit_grid(P, Y), DEC.fit_grid(P, Y))
        th = fitters.fit_exact(P, Y)
        assert abs(fitters.macro_f1(P, Y, th) - DEC.macro_f1(P, Y, th)) < 1e-12


def test_shared_fitter_one_period_is_fit_exact():
    Ps, Ys = synth(1)
    for P, Y in zip(Ps, Ys):
        assert np.array_equal(fitters.fit_shared_exact([P], [Y]), fitters.fit_exact(P, Y))


@pytest.mark.parametrize("weights", [None, [0.1, 0.2, 0.3, 0.4], [0.7, 0.0, 0.2, 0.1], [5, 1, 1, 3]])
def test_shared_fitter_vs_brute_force(weights):
    Ps, Ys = synth(2, n=(40, 33, 51, 27))
    Ys[1][:, 0] = 0                                             # label 0 has no positive in period 1
    Ys[3][:, 0] = 0
    for Y in Ys:
        Y[:, 4] = 0                                             # label 4 has no positive anywhere
    w = np.full(4, 0.25) if weights is None else np.asarray(weights, float)
    th = fitters.fit_shared_exact(Ps, Ys, None if weights is None else w)
    best, arg = brute_shared(Ps, Ys, w)
    for k in range(Ys[0].shape[1]):
        got = sum(w[j] * fitters.f1_vec(np.asarray(Ps[j][:, [k]], np.float64), Ys[j][:, [k]], th[[k]])[0]
                  for j in range(4))
        assert abs(got - best[k]) < 1e-12, (k, got, best[k])
        if k == 4:
            assert th[k] > max(float(np.max(P[:, k])) for P in Ps)    # predicts nothing
            assert best[k] == 0
        else:
            assert th[k] == arg[k]                                   # highest maximiser


def test_weighted_fitter_maximises_its_own_objective():
    Ps, Ys = synth(4)
    w = np.array([0.55, 0.05, 0.3, 0.1])
    th_w = fitters.fit_shared_exact(Ps, Ys, w)
    th_u = fitters.fit_shared_exact(Ps, Ys)
    assert fitters.shared_objective(Ps, Ys, th_w, w) >= fitters.shared_objective(Ps, Ys, th_u, w) - 1e-15
    assert fitters.shared_objective(Ps, Ys, th_u) >= fitters.shared_objective(Ps, Ys, th_w) - 1e-15


# ------------------------------------------------------------- partitions ---
def test_partitions_are_the_paper_rule():
    for seed, j in [(0, 0), (3, 5), (100, 7)]:
        a = crossfit.partitions(97, seed, j, 7)
        rng = np.random.default_rng(7000 + 100 * seed + j)
        c = [rng.permutation(97) for _ in range(7)]
        assert all(np.array_equal(x, z) for x, z in zip(a, c))


def test_period_ids_key_the_partitions():
    Ps, Ys = synth(5)
    full = crossfit.evaluate(Ps, Ys, 3, splits=6)
    sub = crossfit.evaluate(Ps[1:3], Ys[1:3], 3, period_ids=[1, 2], splits=6)
    assert np.array_equal(full.sep[:, :, 1:3], sub.sep)


# --------------------------------------------------------------- identities ---
def _two_models(seed=0):
    Ps, Ys = synth(seed)
    rng = np.random.default_rng(99)
    ref = [np.clip(P.astype(np.float64) + rng.normal(0, 0.05, P.shape), 0, 1) for P in Ps]
    theta = np.full(Ys[0].shape[1], 0.5)
    rules = {"inherited": theta, "validation": np.linspace(0.3, 0.7, Ys[0].shape[1])}
    return Ps, Ys, ref, rules


@pytest.mark.parametrize("weights", [None, [0.1, 0.2, 0.3, 0.4]])
def test_identity_guards(weights):
    """Float-rounding guards only: (sep - shared) + (shared - U) == sep - U and D = A + R hold
    for ANY numbers, so these asserts cannot catch a wrong fitter or partition (the hand
    recomputation below does that). Weighted case: subsets of periods use their own weights."""
    Ps, Ys, ref, rules = _two_models()
    ev_f = crossfit.evaluate(Ps, Ys, 1, rules=rules, splits=12, shared_weights=weights)
    ev_r = crossfit.evaluate(ref, Ys, 1, rules=rules, splits=12, shared_weights=weights)
    for ev in (ev_f, ev_r):
        for U in ev.U_half.values():
            assert np.max(np.abs((ev.sep - ev.shared) + (ev.shared - U) - (ev.sep - U))) < 1e-15
    for periods in (None, [0, 1, 2], [3]):
        if weights is None:
            w = None
        else:
            w = np.asarray(weights)[[0, 1, 2, 3] if periods is None else periods]     # subset weights
        for ev in (ev_f, ev_r):
            s = ev.summary(periods, w)
            for r in s["rules"].values():
                assert r["identity_residual"] < 1e-12
        for rule in rules:
            for halves in (False, True):
                d = crossfit.decompose(ev_r, ev_f, rule, periods, w, halves=halves)
                assert d["shared_terms"] == "reported"
                assert abs(d["D"] - (d["A"] + d["R"])) < 1e-12
                assert abs(d["R"] - (d["dC"] + d["dM"])) < 1e-12
                assert abs(d["D"] - (d["A_shared"] + d["dM"])) < 1e-12


@pytest.mark.parametrize("shared_periods,weights", [(None, None), (None, [0.1, 0.2, 0.3, 0.4]),
                                                    ([0, 2, 3], [0.5, 0.2, 0.3]), ([1, 3], None)])
def test_hand_recompute_partitions(shared_periods, weights):
    """Non-tautological: rebuild every partition by hand from the paper's permutation rule and
    the fitters on explicit halves, and compare with the stored per-partition arrays and the summary."""
    Ps, Ys, _, rules = _two_models(11)
    Ps = [P.astype(np.float64) for P in Ps]
    seed, splits = 4, 9
    ev = crossfit.evaluate(Ps, Ys, seed, rules=rules, splits=splits, shared_periods=shared_periods,
                           shared_weights=weights)
    S = [0, 1, 2, 3] if shared_periods is None else shared_periods
    perms = []                                                # the paper's rule, written out (not partitions())
    for j, Y in enumerate(Ys):
        rng = np.random.default_rng(7000 + 100 * seed + j)
        perms.append([rng.permutation(len(Y)) for _ in range(splits)])
    hand_sep, hand_sh = np.full((splits, 2, 4), np.nan), np.full((splits, 2, 4), np.nan)
    hand_U = {k: np.full((splits, 2, 4), np.nan) for k in rules}
    for b in range(splits):
        for d in (0, 1):
            fit, ev_idx = [], []
            for j, Y in enumerate(Ys):
                h = len(Y) // 2
                first, second = perms[j][b][:h], perms[j][b][h:]
                fit.append(first if d == 0 else second); ev_idx.append(second if d == 0 else first)
            th = fitters.fit_shared_exact([Ps[j][fit[j]] for j in S], [Ys[j][fit[j]] for j in S],
                                          None if weights is None else np.asarray(weights, float))
            for j in range(4):
                Pe, Ye = Ps[j][ev_idx[j]], Ys[j][ev_idx[j]]
                hand_sep[b, d, j] = fitters.macro_f1(Pe, Ye, fitters.fit_exact(Ps[j][fit[j]], Ys[j][fit[j]]))
                if j in S:
                    hand_sh[b, d, j] = fitters.macro_f1(Pe, Ye, th)
                for k, t in rules.items():
                    hand_U[k][b, d, j] = fitters.macro_f1(Pe, Ye, t)
    assert np.array_equal(ev.sep, hand_sep)
    assert np.array_equal(np.isnan(ev.shared), np.isnan(hand_sh))
    assert np.array_equal(np.nan_to_num(ev.shared), np.nan_to_num(hand_sh))
    for k in rules:
        assert np.array_equal(ev.U_half[k], hand_U[k])
        full = [fitters.macro_f1(Ps[j], Ys[j], rules[k]) for j in range(4)]
        assert np.array_equal(ev.U_full[k], np.array(full))
    # the summary over the shared periods, with the shared weights, from the hand arrays
    w = np.full(len(S), 1 / len(S)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    lvl = lambda X: float(np.mean((0.5 * (X[:, 0, S] + X[:, 1, S])) @ w))
    s = ev.summary(S)
    assert abs(s["V_sep"] - lvl(hand_sep)) < 1e-12 and abs(s["V_shared"] - lvl(hand_sh)) < 1e-12
    assert abs(s["rules"]["inherited"]["U_half"] - lvl(hand_U["inherited"])) < 1e-12
    # one partition, one direction, one period, written out without the loops above
    j, b = S[-1], 2
    h = len(Ys[j]) // 2
    A, B = perms[j][b][:h], perms[j][b][h:]
    th_sep = fitters.fit_exact(Ps[j][A], Ys[j][A])
    assert ev.sep[b, 0, j] == fitters.macro_f1(Ps[j][B], Ys[j][B], th_sep)
    assert ev.U_half["validation"][b, 1, j] == fitters.macro_f1(Ps[j][A], Ys[j][A], rules["validation"])


def test_weighted_summary_is_weighted_mean_of_periods():
    Ps, Ys, _, rules = _two_models(2)
    w = np.array([0.5, 0.1, 0.1, 0.3])
    ev = crossfit.evaluate(Ps, Ys, 0, rules=rules, splits=10, shared_weights=w)
    pp, s = ev.per_period(), ev.summary(weights=w)
    assert abs(s["V_sep"] - sum(wi * pp[j]["V_sep"] for wi, j in zip(w, range(4)))) < 1e-12
    assert abs(s["V_shared"] - sum(wi * pp[j]["V_shared"] for wi, j in zip(w, range(4)))) < 1e-12
    u = s["rules"]["inherited"]
    assert abs(u["U_full"] - sum(wi * pp[j]["U_full"]["inherited"] for wi, j in zip(w, range(4)))) < 1e-12
    # the weights change the shared fit, not the own-period rules
    ev_u = crossfit.evaluate(Ps, Ys, 0, rules=rules, splits=10)
    assert np.array_equal(ev.sep, ev_u.sep)
    assert not np.array_equal(ev.shared, ev_u.shared)


def test_bridge_equals_decomposition_module(DEC):
    """At one period, with the paper's partitions handed to decomposition.decompose, the
    bridge here equals decomposition.py's B, A, dR."""
    Ps, Ys, ref, rules = _two_models(3)
    seed, j, splits = 2, 1, 15
    ev_f = crossfit.evaluate(Ps, Ys, seed, rules=rules, splits=splits)
    ev_r = crossfit.evaluate(ref, Ys, seed, rules=rules, splits=splits)
    d = crossfit.decompose(ev_r, ev_f, "inherited", periods=[j])
    o = DEC.decompose(ref[j], Ps[j].astype(np.float64), Ys[j], rules["inherited"], splits,
                      np.random.default_rng(7000 + 100 * seed + j))
    for a, b in (("D", "B"), ("A", "A"), ("R", "dR"), ("V_ref", "V_ref"), ("V_final", "V_final")):
        assert abs(d[a] - o[b]) < 1e-12, (a, d[a], o[b])


# --------------------------------------------------------------- invariance ---
def test_invariance_under_common_monotone_map():
    """A strictly increasing map per label, the same in every period, applied to the scores
    (and to the fixed rules) leaves every level unchanged: only within-label order matters."""
    Ps, Ys, _, rules = _two_models(6)
    Ps = [P.astype(np.float64) for P in Ps]
    K = Ys[0].shape[1]
    maps = [lambda x: np.log(x / (1 - x)), lambda x: x ** 3 + 2.0, lambda x: 7 * x - 1,
            lambda x: np.exp(3 * x), lambda x: np.sqrt(x) + x]
    g = lambda P: np.column_stack([maps[k](P[:, k]) for k in range(K)])
    allP = np.vstack(Ps + [r[None, :] for r in rules.values()])
    for k in range(K):                                     # the map is strictly increasing on these values
        u = np.unique(allP[:, k])
        assert np.all(np.diff(maps[k](u)) > 0)
    rules_g = {n: g(th[None, :])[0] for n, th in rules.items()}
    a = crossfit.evaluate(Ps, Ys, 4, rules=rules, splits=10)
    b = crossfit.evaluate([g(P) for P in Ps], Ys, 4, rules=rules_g, splits=10)
    assert np.array_equal(a.sep, b.sep)
    assert np.array_equal(a.shared, b.shared)
    for n in rules:
        assert np.array_equal(a.U_half[n], b.U_half[n])
        assert np.array_equal(a.U_full[n], b.U_full[n])


def test_period_specific_map_moves_shared_not_sep():
    """The negative control: a map that differs by period leaves the own-period levels alone
    but changes what one shared rule can reach (that is what C measures)."""
    Ps, Ys, _, _ = _two_models(7)
    Ps = [P.astype(np.float64) for P in Ps]
    Pq = [P ** (1 + j) for j, P in enumerate(Ps)]
    a = crossfit.evaluate(Ps, Ys, 0, splits=8)
    b = crossfit.evaluate(Pq, Ys, 0, splits=8)
    assert np.array_equal(a.sep, b.sep)
    assert not np.array_equal(a.shared, b.shared)


# --------------------------------------------------------------- edge cases ---
def test_no_positives():
    Ps, Ys = synth(8)
    for Y in Ys:
        Y[:, 1] = 0                                         # label 1 never positive
    Ys[2][:] = 0                                            # period 2 has no positive at all
    th = {"inherited": np.full(Ys[0].shape[1], 0.5)}
    ev = crossfit.evaluate(Ps, Ys, 0, rules=th, splits=6)
    pp = ev.per_period()
    assert pp[2]["V_sep"] == 0.0 and pp[2]["V_shared"] == 0.0     # F1 = 0 by zero_division
    s = ev.summary()
    assert s["rules"]["inherited"]["identity_residual"] < 1e-12
    # the fitted rules predict nothing for the never-positive label
    P0 = np.asarray(Ps[0], np.float64)
    assert fitters.fit_exact(P0, Ys[0])[1] > P0[:, 1].max()
    assert fitters.fit_shared_exact(Ps, Ys)[1] > max(float(P[:, 1].max()) for P in Ps)


def test_input_validation():
    Ps, Ys = synth(9)
    with pytest.raises(ValueError):
        crossfit.evaluate(Ps, Ys[:-1], 0, splits=2)
    with pytest.raises(ValueError):
        crossfit.evaluate([Ps[0][:, :3]] + Ps[1:], Ys, 0, splits=2)
    bad = [Y.copy() for Y in Ys]; bad[0][0, 0] = 2
    with pytest.raises(ValueError):
        crossfit.evaluate(Ps, bad, 0, splits=2)
    with pytest.raises(ValueError):
        crossfit.evaluate([P[:1] for P in Ps], [Y[:1] for Y in Ys], 0, splits=2)
    with pytest.raises(ValueError):
        crossfit.evaluate(Ps, Ys, 0, period_ids=[0, 0, 1, 2], splits=2)
    a = crossfit.evaluate(Ps, Ys, 0, splits=2)
    b = crossfit.evaluate(Ps, Ys, 1, splits=2)
    with pytest.raises(ValueError):
        crossfit.decompose(a, b, "inherited")                # different seeds: different partitions
    for w in ([-1, 0, 0, 0], [1, np.nan, 1, 1], [0, 0, 0, 0], [1, 1, 1]):
        with pytest.raises(ValueError):
            crossfit.evaluate(Ps, Ys, 0, splits=2, shared_weights=w)
    with pytest.raises(ValueError):
        crossfit.evaluate(Ps, Ys, 0, splits=2, rules={"inherited": np.array([0.5, np.nan, 0.5, 0.5, 0.5])})
    with pytest.raises(ValueError):
        crossfit.evaluate(Ps, Ys, 0, splits=2, shared_periods=[0, 9])
    with pytest.raises(ValueError):
        a.summary(weights=[1, -1, 1, 1])


def test_summary_uses_the_shared_weights():
    """Fitted under one weighting, reported under the same one unless told otherwise."""
    Ps, Ys, _, rules = _two_models(5)
    w = [0.6, 0.1, 0.2, 0.1]
    ev = crossfit.evaluate(Ps, Ys, 0, rules=rules, splits=6, shared_weights=w)
    assert ev.summary() == ev.summary(weights=w)
    assert ev.summary()["weights"] == pytest.approx(w)
    with pytest.raises(ValueError):
        ev.summary(periods=[0, 1])                            # a subset needs explicit weights
    s = ev.summary(periods=[0, 1], weights=[1, 1])
    assert s["V_shared"] is not None
    ev2 = crossfit.evaluate(Ps, Ys, 0, rules=rules, splits=6, shared_periods=[1, 2], shared_weights=[0.3, 0.7])
    assert ev2.summary([1, 2]) == ev2.summary([1, 2], [0.3, 0.7])
    with pytest.raises(ValueError):
        ev2.summary()                                         # all periods differ from the shared ones


def test_decompose_pairing_checks():
    Ps, Ys, ref, rules = _two_models(4)
    ev_f = crossfit.evaluate(Ps, Ys, 2, rules=rules, splits=6)
    ev_r = crossfit.evaluate(ref, Ys, 2, rules=rules, splits=6)
    assert crossfit.decompose(ev_r, ev_f, "inherited")["same_rule"]
    # the same labels in another record order: same counts, different records per half
    perm = np.random.default_rng(1).permutation(len(Ys[1]))
    Ps_p = [P.copy() for P in Ps]; Ys_p = [Y.copy() for Y in Ys]
    Ps_p[1], Ys_p[1] = Ps_p[1][perm], Ys_p[1][perm]
    ev_p = crossfit.evaluate(Ps_p, Ys_p, 2, rules=rules, splits=6)
    assert ev_p.label_signature == ev_r.label_signature        # the old check alone would pass
    with pytest.raises(ValueError, match="record order"):
        crossfit.decompose(ev_r, ev_p, "inherited")
    # a different vector under the same rule name
    other = dict(rules, inherited=np.full(Ys[0].shape[1], 0.2))
    ev_o = crossfit.evaluate(Ps, Ys, 2, rules=other, splits=6)
    with pytest.raises(ValueError, match="different threshold vector"):
        crossfit.decompose(ev_r, ev_o, "inherited")
    d = crossfit.decompose(ev_r, ev_o, "inherited", same_rule=False)   # model-specific rule, flagged
    assert d["same_rule"] is False
    with pytest.raises(KeyError):
        crossfit.decompose(ev_r, ev_f, "nonexistent")
    # a per-period reference: no shared-rule terms
    ev_d = crossfit.evaluate(ref, Ys, 2, rules=rules, splits=6, single_model=False)
    d = crossfit.decompose(ev_d, ev_f, "inherited")
    assert "dC" not in d and "A_shared" not in d and d["shared_terms"].startswith("omitted")
    assert abs(d["D"] - (d["A"] + d["R"])) < 1e-12
    # shared rules fitted under different weights are not compared
    ev_w = crossfit.evaluate(Ps, Ys, 2, rules=rules, splits=6, shared_weights=[0.4, 0.2, 0.2, 0.2])
    d = crossfit.decompose(ev_r, ev_w, "inherited", weights=[1, 1, 1, 1])
    assert "dC" not in d and "weights" in d["shared_terms"]


# ---------------------------------------------------------------- inference ---
def test_paired_vector_is_strict():
    a = {s: 0.01 * s for s in range(10)}
    b = {s: 0.0 for s in range(6)}                            # a failed run would leave a gap
    with pytest.raises(ValueError, match="different seeds"):
        inference.paired_vector(a, b)
    with pytest.raises(ValueError, match="planned seeds missing"):
        inference.paired_vector(a, b, seeds=range(10))
    seeds, x = inference.paired_vector(a, b, seeds=range(6))
    assert seeds == list(range(6)) and len(x) == 6
    seeds, x = inference.paired_vector(a, b, strict=False)
    assert seeds == list(range(6))
    st = inference.paired_contrast(a, b, strict=False)
    assert st["n"] == 6 and st["dropped_seeds"] == [6, 7, 8, 9]
    with pytest.raises(ValueError):
        inference.paired_contrast(a, b)


def test_contrast_classes():
    rng = np.random.default_rng(0)
    big = 0.02 + rng.normal(0, 0.002, 10)
    zero = rng.normal(0, 0.0005, 10)
    noisy = rng.normal(0.004, 0.02, 10)
    fam = inference.family({"big": big, "zero": zero, "noisy": noisy})
    assert fam["big"]["class"] == "material"
    assert fam["zero"]["class"] == "equivalent"
    assert fam["noisy"]["class"] == "unresolved"
    st = fam["big"]
    lo, hi = st["ci95"]
    assert lo < st["mean"] < hi and lo > 0
    c = inference.paired_contrast({s: big[s] for s in range(10)}, {s: 0.0 for s in range(10)})
    assert abs(c["mean"] - big.mean()) < 1e-15 and c["seeds"] == list(range(10))
    ni = inference.family_ni({"close": rng.normal(-0.001, 0.001, 10), "far": -0.03 + rng.normal(0, 0.002, 10)})
    assert ni["close"]["ni"] and not ni["close"]["inf"]
    assert ni["far"]["inf"] and not ni["far"]["ni"]


def test_example_runs(capsys):
    from refitgap import example
    example.main()
    out = capsys.readouterr().out
    assert "B = " in out and "class material" in out
