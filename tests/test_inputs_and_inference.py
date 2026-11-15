"""Margins, input checks, weightings, ties, partition keys, and reference values for the
inference functions computed by hand or with scipy's own one-sample t test."""
import json

import numpy as np
import pytest
from scipy import stats

import refitgap as rg
from refitgap import analysis, crossfit, fitters, inference
from refitgap.cli import main as cli_main


def _runs(seeds=range(6), N=2, K=3, n=(40, 46), shared_ref=True, shift=0.0):
    a, b = [], []
    for s in seeds:
        rng = np.random.default_rng(300 + s)
        Ys = [(rng.random((n[j], K)) < [0.3, 0.25, 0.4]).astype(int) for j in range(N)]
        sc = lambda Y, sep, off: np.round(1 / (1 + np.exp(-(sep * Y + rng.normal(0, 1, Y.shape) + off))), 2)
        ref = [sc(Y, 2.5, -1.0) for Y in Ys] if shared_ref else None
        fa = [sc(Y, 2.2, -0.8) for Y in Ys]
        fb = [sc(Y, 2.2 - shift, -0.8) for Y in Ys]
        th = np.array([0.5, 0.45, 0.55])
        a.append(analysis.make_run(fa, Ys, s, {"inherited": th}, ref, source=f"a{s}"))
        b.append(analysis.make_run(fb, Ys, s, {"inherited": th}, ref, source=f"b{s}"))
    return a, b


# ---------------------------------------------------------- margin plumbing ---
PROBE = np.array([0.010, 0.012, 0.009, 0.011, 0.010, 0.013, 0.008, 0.011, 0.012, 0.010])


def test_probe_case_reads_equivalent_at_margin_002():
    f = inference.family_ni({"t": PROBE}, margin=0.02)["t"]
    g = inference.family_at({"t": PROBE}, 0.02)["t"]
    h = inference.family({"t": PROBE}, margin=0.02)["t"]
    assert f["class"] == g["class"] == h["class"] == "equivalent"
    assert f["p_tost"] == g["p_tost"] == h["p_tost"] < 1e-6
    assert f["p_tost_holm"] == g["p_tost_holm"] and f["margin"] == 0.02
    # at the default margin the same vector is material (mean 0.0106 >= 0.005)
    assert inference.family_ni({"t": PROBE})["t"]["class"] == "material"


def test_rebinding_the_global_margin_changes_nothing():
    base = inference.family({"t": PROBE})["t"]
    old = inference.MARGIN
    try:
        inference.MARGIN = 0.02
        moved = inference.family({"t": PROBE})["t"]
        unadj = inference.paired_contrast({s: v for s, v in enumerate(PROBE)}, {s: 0.0 for s in range(10)})
    finally:
        inference.MARGIN = old
    assert moved["class"] == base["class"] and moved["p_tost"] == base["p_tost"]
    assert unadj["class"] == "material"
    assert inference.MARGIN == 0.005


def test_margin_argument_everywhere():
    st = inference.paired_stats(PROBE, 0.02)
    assert inference.classify_unadjusted(dict(st), 0.02)["class"] == "equivalent"
    c = inference.paired_contrast({s: v for s, v in enumerate(PROBE)}, {s: 0.0 for s in range(10)}, margin=0.02)
    assert c["class"] == "equivalent" and c["p_tost"] == st["p_tost"]
    # classify at a margin above |mean|: small non-zero instead of material
    st = dict(inference.paired_stats(PROBE), p_holm=1e-9, p_tost_holm=1.0)
    assert inference.classify(st, 0.005) == "material" and inference.classify(st, 0.02) == "small non-zero"
    with pytest.raises(ValueError, match="margin"):
        inference.family({"t": PROBE}, margin=-0.01)


@pytest.mark.parametrize("margin", [0.02, 0.5])
def test_analyse_family_ni_and_family_at_agree(margin):
    a, b = _runs(shift=0.3)
    res = analysis.analyse(a, b, splits=4, margin=margin, family="F")
    rows = [r for r in res["contrasts"] if r["comparison"] == "a minus b"]
    vecs = {r["term"]: np.array([float(v) for v in r["values"].split()]) for r in rows}
    ni, at = inference.family_ni(vecs, margin), inference.family_at(vecs, margin)
    for r in rows:
        t = r["term"]
        assert r["margin"] == margin
        assert r["class"] == ni[t]["class"] == at[t]["class"]
        assert ni[t]["p_tost"] == at[t]["p_tost"] == inference.paired_stats(vecs[t], margin)["p_tost"]
        assert r["p_t_holm"] == ni[t]["p_holm"]
    if margin == 0.5:                               # all inside +-0.5
        assert all(r["class"] == "equivalent" for r in rows)


def test_cli_margin_reaches_the_class(tmp_path):
    a, b = _runs(shift=0.3)
    pa, pb = [], []
    for side, runs, paths in (("a", a, pa), ("b", b, pb)):
        for r in runs:
            p = tmp_path / f"{side}{r['seed']}.npz"
            analysis.save_arrays(p, r)
            paths.append(str(p))
    out = tmp_path / "o"
    assert cli_main(["evaluate", "--arrays", *pa, "--arrays-b", *pb, "--splits", "4", "--margin", "0.5",
                     "--family", "F", "--out", str(out), "--quiet"]) == 0
    res = json.loads((out / "result.json").read_text())
    rows = [r for r in res["contrasts"] if r["comparison"] == "a minus b"]
    assert rows and all(r["class"] == "equivalent" and r["margin"] == 0.5 for r in rows)


# -------------------------------------------------------- non-finite values ---
def test_nan_and_inf_are_refused_in_inference():
    x = np.array([0.01, 0.02, np.nan, 0.015, 0.012, 0.011, 0.013, 0.02])
    with pytest.raises(ValueError, match="non-finite value nan at position 2"):
        inference.paired_stats(x)
    with pytest.raises(ValueError, match="family member 'a'"):
        inference.family({"a": x, "b": np.zeros(8)})
    with pytest.raises(ValueError, match="family member 'a'"):
        inference.family_ni({"a": np.where(np.isnan(x), np.inf, x)})
    with pytest.raises(ValueError, match="holm: p value nan at position 1"):
        inference.holm([0.01, np.nan, 0.04])
    with pytest.raises(ValueError, match="not a number in"):
        inference.holm([0.01, 1.5])
    with pytest.raises(ValueError, match="side a, seed 2"):
        inference.paired_vector({s: v for s, v in enumerate(x)}, {s: 0.0 for s in range(8)})
    with pytest.raises(ValueError, match="side b, seed 3"):
        inference.paired_contrast({s: 0.0 for s in range(8)}, {s: (np.inf if s == 3 else 0.0) for s in range(8)})


def test_nan_and_inf_are_refused_in_scores_and_labels():
    rng = np.random.default_rng(0)
    P, Y = rng.random((20, 2)), (rng.random((20, 2)) < 0.4).astype(int)
    Pn = P.copy(); Pn[4, 1] = np.nan
    Yn = Y.astype(float); Yn[3, 0] = np.nan
    for bad in (Pn, np.where(Pn != Pn, np.inf, Pn)):
        with pytest.raises(ValueError, match="non-finite scores"):
            fitters.fit_exact(bad, Y)
        with pytest.raises(ValueError, match="non-finite scores"):
            fitters.fit_shared_exact([P, bad], [Y, Y])
        with pytest.raises(ValueError, match="non-finite scores"):
            fitters.fit_grid(bad, Y)
        with pytest.raises(ValueError, match="period 1: non-finite scores"):
            crossfit.evaluate([P, bad], [Y, Y], 0, splits=2)
        with pytest.raises(ValueError, match="scores_final has non-finite"):
            rg.compare(P, bad, Y, [0.5, 0.5], seed=0, splits=2)
    with pytest.raises(ValueError, match="labels must be 0/1"):
        fitters.fit_exact(P, Yn)
    with pytest.raises(ValueError, match="labels must be 0/1"):
        crossfit.evaluate([P], [Yn], 0, splits=2)
    with pytest.raises(ValueError, match="labels must be 0/1"):
        rg.compare(P, P, Yn, [0.5, 0.5], seed=0, splits=2)


# -------------------------------------- two weightings in one decomposition ---
def test_decompose_refuses_two_default_weightings():
    a, _ = _runs(seeds=[0])
    r = a[0]
    ev_w = crossfit.evaluate(r["final"], r["labels"], 0, rules=r["rules"], shared_weights=[0.9, 0.1], splits=3)
    ev_u = crossfit.evaluate(r["reference"], r["labels"], 0, rules=r["rules"], splits=3)
    with pytest.raises(ValueError, match="different shared_weights"):
        crossfit.decompose(ev_u, ev_w, "inherited")
    d = crossfit.decompose(ev_u, ev_w, "inherited", weights=[1, 1])        # one explicit weighting
    assert d["U_final"] == ev_w.summary(weights=[1, 1])["rules"]["inherited"]["U_full"]
    with pytest.raises(ValueError, match="different shared_weights"):
        rg.method_contrast({0: ev_u}, {0: ev_w}, "inherited")
    # equal weights on both sides: unchanged behaviour
    ev_w2 = crossfit.evaluate(r["reference"], r["labels"], 0, rules=r["rules"], shared_weights=[0.9, 0.1], splits=3)
    assert crossfit.decompose(ev_w2, ev_w, "inherited")["shared_terms"] == "reported"


# ---------------------------------------------- analyse(shared_weights=...) ---
def test_analyse_with_shared_weights_and_references(tmp_path):
    a, b = _runs(seeds=[0, 1, 2])
    res = analysis.analyse(a, b, splits=3, shared_weights=[0.7, 0.3])
    assert res["decomposition"] and res["settings"]["shared_weights"] == [0.7, 0.3]
    # the per-period rows equal the unweighted ones (one period has no weighting)
    plain = analysis.analyse(a, b, splits=3)
    per = lambda R: {(r["method"], r["seed"], r["period"]): r for r in R["decomposition"] if r["period"] in (0, 1)}
    pw, pp = per(res), per(plain)
    for k in pp:
        for c in ("B", "A", "dR", "U_ref", "U_final"):
            assert pw[k][c] == pp[k][c]
    # the mean rows use the 0.7 / 0.3 weighting on both reference and final
    ev_f, ev_r = res["evaluations"][("a", 0, "final")], res["evaluations"][("a", 0, "reference")]
    m = next(r for r in res["decomposition"] if r["method"] == "a" and r["seed"] == 0 and r["period"] != 0
             and r["period"] != 1)
    d = crossfit.decompose(ev_r, ev_f, "inherited", weights=[0.7, 0.3])
    assert m["B"] == d["B"] and m["A"] == d["A"]
    # the command line
    paths = []
    for r in a:
        p = tmp_path / f"a{r['seed']}.npz"
        analysis.save_arrays(p, r)
        paths.append(str(p))
    assert cli_main(["evaluate", "--arrays", *paths, "--splits", "3", "--shared-weights", "0.7", "0.3",
                     "--out", str(tmp_path / "o"), "--quiet"]) == 0


# -------------------------------------- fixed_split with integer record ids ---
def test_fixed_split_integer_ids():
    n = 10
    P = np.linspace(0.95, 0.05, n)[:, None]
    Y = np.array([1] * 5 + [0] * 5)[:, None]
    ids = list(range(9, -1, -1))                    # rows 0..4 hold ids 9..5, the five positives
    fs = {"calibration": [9, 8, 7, 6, 5], "evaluation": [4, 3, 2, 1, 0]}
    with pytest.raises(ValueError, match="fixed_split_by"):
        rg.compare(P, P, Y, [0.5], record_ids=ids, fixed_split=fs)
    r = rg.compare(P, P, Y, [0.5], record_ids=ids, fixed_split=fs, fixed_split_by="ids")
    assert r["metadata"]["label_budget"][0]["calibration_positives_per_label"] == [5]
    r = rg.compare(P, P, Y, [0.5], record_ids=ids, fixed_split=fs, fixed_split_by="positions")
    assert r["metadata"]["label_budget"][0]["calibration_positives_per_label"] == [0]
    # unambiguous integer ids are read as ids
    ids2 = list(range(100, 110))
    r = rg.compare(P, P, Y, [0.5], record_ids=ids2,
                   fixed_split={"calibration": [100, 101, 102, 105], "evaluation": [103, 104, 106]})
    assert r["metadata"]["label_budget"][0]["calibration_positives_per_label"] == [3]
    # integer positions against string ids stay positions; ids given as strings stay ids
    sids = [f"r{i}" for i in range(n)]
    r1 = rg.compare(P, P, Y, [0.5], record_ids=sids, fixed_split={"calibration": [0, 1, 5], "evaluation": [2, 6]})
    r2 = rg.compare(P, P, Y, [0.5], record_ids=sids,
                    fixed_split={"calibration": ["r0", "r1", "r5"], "evaluation": ["r2", "r6"]})
    assert r1["components"] == r2["components"]
    with pytest.raises(ValueError, match="needs record_ids"):
        rg.compare(P, P, Y, [0.5], fixed_split=fs, fixed_split_by="ids")
    with pytest.raises(ValueError, match="only with fixed_split"):
        rg.compare(P, P, Y, [0.5], seed=0, splits=2, fixed_split_by="ids")


# ------------------------------------------ shared-fitter ties (documented) ---
def test_shared_fitter_tie_breaking_is_by_float_sum():
    """Documents the behaviour, the arithmetic of the paper's analyses: exact objectives
    7/9 at cutpoints 0.5 and 0, and the float sum makes the lower cutpoint win."""
    Ps = [np.array([[0.0], [1.0]]), np.array([[1.0], [0.75]]), np.array([[0.5], [0.0]])]
    Ys = [np.array([[1], [1]]), np.array([[0], [1]]), np.array([[1], [0]])]
    from fractions import Fraction as F

    def exact(t):
        tot = F(0)
        for P, Y in zip(Ps, Ys):
            pred = P[:, 0] >= t
            tp, cnt, npos = int((pred & (Y[:, 0] == 1)).sum()), int(pred.sum()), int(Y.sum())
            tot += F(2 * tp, cnt + npos) if cnt + npos else F(0)
        return tot / 3
    assert [exact(t) for t in (1.0, 0.75, 0.5, 0.0)] == [F(2, 9), F(4, 9), F(7, 9), F(7, 9)]
    assert fitters.fit_shared_exact(Ps, Ys)[0] == 0.0      # not 0.5, the highest exact maximiser
    # with one period the tie goes exactly to the highest cutpoint, as in fit_exact
    P = np.array([[0.9], [0.8], [0.7], [0.6]]); Y = np.array([[1], [0], [1], [0]])
    assert fitters.fit_shared_exact([P], [Y])[0] == fitters.fit_exact(P, Y)[0] == 0.7


# ------------------------------------------- nested Python lists in compare ---
def test_compare_refuses_a_matrix_given_as_nested_list():
    S = [[.9, .1], [.8, .2], [.2, .7], [.1, .9]]
    L = [[1, 0], [1, 0], [0, 1], [0, 1]]
    with pytest.raises(ValueError, match="reads as one 4 x 2 matrix"):
        rg.compare(S, S, L, [0.5], seed=0, splits=2)
    with pytest.raises(ValueError, match="list of numbers"):
        rg.compare([0.9, 0.1], [0.9, 0.1], [1, 0], [0.5], seed=0, splits=2)
    r = rg.compare(np.asarray(S), np.asarray(S), np.asarray(L), [0.5, 0.5], seed=0, splits=2)
    assert r["metadata"]["n_labels"] == 2 and r["metadata"]["period_ids"] == [0]
    # a list of per-period matrices (nested three deep) is still one matrix per period
    r = rg.compare([S, S], [S, S], [L, L], [0.5, 0.5], seed=0, splits=2)
    assert r["metadata"]["period_ids"] == [0, 1] and r["metadata"]["n_labels"] == 2


# ------------------------------------------------------- more than 20 seeds ---
def test_analyse_and_contrasts_run_with_25_seeds():
    a, b = _runs(seeds=range(25), N=1, n=(20,))
    res = analysis.analyse(a, b, splits=2, family="F")
    assert all(r["n"] == 25 and r["class"] is not None for r in res["contrasts"])
    x = np.random.default_rng(1).normal(0.01, 0.005, 25)
    c = inference.paired_contrast({s: v for s, v in enumerate(x)}, {s: 0.0 for s in range(25)})
    assert "Monte Carlo" in c["p_signflip_method"]


# ------------------------------------------------------ zero standard error ---
def test_zero_standard_error():
    z = inference.paired_stats(np.zeros(8))
    assert z["p_t"] == 1.0 and z["p_tost"] == 0.0
    assert inference.family({"z": np.zeros(8)})["z"]["class"] == "equivalent"
    assert inference.family({"z": np.zeros(8)}, margin=0.0)["z"]["class"] == "unresolved"
    c = inference.paired_stats(np.full(8, 0.25))               # constant non-zero: p_t 0 (documented)
    assert c["se"] == 0 and c["p_t"] == 0.0


# ----------------------------------------------------------- partition keys ---
def test_partition_keys_are_validated():
    P = np.random.default_rng(0).random((10, 2)); Y = (P > 0.5).astype(int)
    for bad in (dict(seed=-1), dict(seed=1.5), dict(seed=True)):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            crossfit.evaluate([P], [Y], bad["seed"], splits=2)
    for j in (100, -1, 1.5):
        with pytest.raises(ValueError, match=r"period id must be an integer in \[0, 100\)"):
            crossfit.evaluate([P], [Y], 0, period_ids=[j], splits=2)
        with pytest.raises(ValueError, match="period id"):
            crossfit.partitions(10, 0, j, splits=1)
    with pytest.raises(ValueError, match="period id"):
        analysis.make_run([P], [Y], 0, {"r": [0.5, 0.5]}, period_ids=[100])
    # the formula is unchanged for valid keys
    assert np.array_equal(crossfit.partitions(10, 3, 99, splits=1)[0],
                          np.random.default_rng(7000 + 300 + 99).permutation(10))
    ev = crossfit.evaluate([P], [Y], np.int64(5), period_ids=[np.int64(99)], splits=2)
    assert ev.period_ids == [99]


# ----------------------------------------------------------- BCa exceptions ---
def test_bca_catches_only_expected_errors(monkeypatch):
    x = np.array([0.01, 0.03, 0.02, 0.05, 0.04])
    assert inference.bca_ci(x) is not None

    def boom(exc):
        def f(*a, **k):
            raise exc
        return f
    monkeypatch.setattr(inference.stats, "bootstrap", boom(ValueError("degenerate")))
    assert inference.bca_ci(x) is None
    monkeypatch.setattr(inference.stats, "bootstrap", boom(TypeError("unexpected keyword 'random_state'")))
    with pytest.raises(TypeError):
        inference.bca_ci(x)


# ------------------------------------------------- independent reference values ---
def test_holm_hand_values():
    assert inference.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06], abs=1e-15)
    assert inference.holm([0.02, 0.02]) == pytest.approx([0.04, 0.04], abs=1e-15)
    assert inference.holm([0.5, 0.6]) == [1.0, 1.0]
    assert inference.holm([0.04, 0.001, 0.02, 0.3]) == pytest.approx([0.08, 0.004, 0.06, 0.3], abs=1e-15)
    assert inference.holm([0.2]) == [0.2]


@pytest.mark.parametrize("seed", range(5))
def test_t_and_tost_p_against_scipy(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(rng.uniform(-0.01, 0.01), rng.uniform(0.001, 0.01), int(rng.integers(3, 15)))
    for m in (0.005, 0.02):
        st = inference.paired_stats(x, m)
        lo = stats.ttest_1samp(x, -m, alternative="greater").pvalue      # H0 mean <= -m
        hi = stats.ttest_1samp(x, m, alternative="less").pvalue          # H0 mean >= +m
        assert st["p_lo"] == pytest.approx(lo, rel=1e-9, abs=1e-15)
        assert st["p_hi"] == pytest.approx(hi, rel=1e-9, abs=1e-15)
        assert st["p_tost"] == pytest.approx(max(lo, hi), rel=1e-9, abs=1e-15)
        n, se = len(x), x.std(ddof=1) / np.sqrt(len(x))
        assert st["p_t"] == pytest.approx(2 * stats.t.sf(abs(x.mean()) / se, n - 1), rel=1e-9, abs=1e-300)
        half = stats.t.ppf(0.975, n - 1) * se
        assert st["ci95"] == pytest.approx([x.mean() - half, x.mean() + half], abs=1e-15)
        assert inference.ni_p(x, m) == pytest.approx(lo, rel=1e-9, abs=1e-15)
        inf_p = stats.ttest_1samp(x, -m, alternative="less").pvalue      # H0 mean >= -m
        assert inference.ni_p(-x, -m) == pytest.approx(inf_p, rel=1e-9, abs=1e-15)


def _enumerate(x, centre=None):
    """Sign-flip p by a plain loop over the integers 0 .. 2^n - 1 (bit k = sign of value k)."""
    x = [float(v) for v in x]
    n = len(x)
    if centre is None:
        obs = abs(sum(x) / n)
        hits = sum(abs(sum(v if (b >> k) & 1 else -v for k, v in enumerate(x)) / n) >= obs - 1e-12
                   for b in range(2 ** n))
    else:
        y = [v + centre for v in x]
        obs = sum(y) / n
        if obs <= 0:
            return 1.0
        hits = sum(sum(abs(v) if (b >> k) & 1 else -abs(v) for k, v in enumerate(y)) / n >= obs - 1e-12
                   for b in range(2 ** n))
    return hits / 2 ** n


def test_signflip_exact_hand_values():
    # x = (1, 2, 3): |sums| over the 8 patterns are 6, 4, 2, 0, 0, 2, 4, 6; p = 2/8
    assert inference.signflip_p([1.0, 2.0, 3.0]) == 0.25
    # one-sided about -0 (margin 0): only the all-plus pattern reaches the observed mean; p = 1/8
    assert inference.ni_signflip_p([1.0, 2.0, 3.0], 0.0) == 0.125
    # x = (1, -1): sums 0, 2, -2, 0, observed |mean| 0, every pattern reaches it
    assert inference.signflip_p([1.0, -1.0]) == 1.0
    # the floor 2 / 2^n with n = 5 (README: no material class with 5 seeds)
    assert inference.signflip_p([0.01, 0.02, 0.015, 0.03, 0.012]) == 2 / 32
    rng = np.random.default_rng(9)
    for _ in range(20):
        x = rng.normal(0.003, 0.01, int(rng.integers(2, 11)))
        assert inference.signflip_p(x) == _enumerate(x)
        for m in (0.005, 0.0):
            assert inference.ni_signflip_p(x, m) == _enumerate(x, centre=m)
