"""The two-scorer entry point, the paired method contrast, the identification example and the
confusion-count bounds. Seconds to run.

compare() and method_contrast() must reproduce evaluate/decompose/paired_contrast exactly (they
add no estimator). The identification example is asserted in exact fractions. The count bounds
are checked against an exhaustive enumeration of every ranking of n <= 10 records."""
import itertools
import warnings
from fractions import Fraction as Fr

import numpy as np
import pytest

import refitgap as rg
from refitgap import crossfit, example, fitters

EXACT = 1e-12
SPLITS = 20


@pytest.fixture(scope="module")
def two_finals():
    """The worked example's seed 0: two final models (sequential, replay) on four periods, theta_1."""
    Ys, Yval, Pval_first, ref, sequential, replay = example.make_seed(0)
    theta = crossfit.fit_validation_rule(Pval_first, Yval)
    return Ys, sequential, replay, theta


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, float):
            out[prefix + k] = v
    return out


# ------------------------------------------------------------------ compare ---
@pytest.mark.parametrize("halves", [False, True])
def test_compare_reproduces_decompose(two_finals, halves):
    Ys, sequential, replay, theta = two_finals
    ev_r = crossfit.evaluate(sequential, Ys, 0, rules={"inherited": theta}, splits=SPLITS)
    ev_f = crossfit.evaluate(replay, Ys, 0, rules={"inherited": theta}, splits=SPLITS)
    d = crossfit.decompose(ev_r, ev_f, "inherited", halves=halves)
    res = rg.compare(sequential, replay, Ys, theta, seed=0, splits=SPLITS, halves=halves, rule_name="inherited")
    flat = _flat(d)
    assert set(flat) <= set(_flat(res["decomposition"]))
    for k, v in flat.items():
        assert abs(res["decomposition"][k] - v) <= EXACT, k
    c = res["components"]
    for ours, theirs in [("deficit", "D"), ("attainable_gap", "A"), ("mismatch_change", "R"),
                         ("mismatch_ref", "R_ref"), ("mismatch_final", "R_final"),
                         ("attainable_gap_shared", "A_shared"), ("shared_mismatch_change", "dM"),
                         ("shared_cost_change", "dC")]:
        assert abs(c[ours] - d[theirs]) <= EXACT, ours
    assert abs(res["carried"]["ref"] - d["U_ref"]) <= EXACT and abs(res["carried"]["final"] - d["U_final"]) <= EXACT
    assert abs(res["refitted"]["ref"] - d["V_ref"]) <= EXACT and abs(res["refitted"]["final"] - d["V_final"]) <= EXACT
    assert abs(c["deficit"] - c["attainable_gap"] - c["mismatch_change"]) <= EXACT
    # per period, against decompose on that period
    for j in range(len(Ys)):
        dj = crossfit.decompose(ev_r, ev_f, "inherited", periods=[j], halves=halves)
        pj = res["per_period"][j]["components"]
        assert abs(pj["deficit"] - dj["D"]) <= EXACT and abs(pj["attainable_gap"] - dj["A"]) <= EXACT


def test_compare_one_period_and_metadata(two_finals):
    Ys, sequential, replay, theta = two_finals
    res = rg.compare(sequential[0], replay[0], Ys[0], theta, seed=3, splits=SPLITS)
    ev_r = crossfit.evaluate([sequential[0]], [Ys[0]], 3, rules={"carried": theta}, splits=SPLITS)
    ev_f = crossfit.evaluate([replay[0]], [Ys[0]], 3, rules={"carried": theta}, splits=SPLITS)
    d = crossfit.decompose(ev_r, ev_f, "carried")
    assert res["components"]["deficit"] == d["D"] and res["components"]["attainable_gap"] == d["A"]
    m = res["metadata"]
    n = len(Ys[0])
    assert m["single_period"] and m["splits"] == SPLITS and m["seed"] == 3 and m["rule"] == theta.tolist()
    assert m["label_budget"][0]["fit_records_per_direction"] == [n // 2, n - n // 2]
    assert "score >= threshold" in m["ties"] and "zero_division" in m["absent_labels"]
    assert m["score_scale"] == "probability" and m["package_version"] == rg.__version__


def test_compare_aligns_record_and_label_ids(two_finals):
    Ys, sequential, replay, theta = two_finals
    base = rg.compare(sequential, replay, Ys, theta, seed=0, splits=SPLITS)
    rng = np.random.default_rng(1)
    ids_lab = [[f"p{j}r{i}" for i in range(len(Y))] for j, Y in enumerate(Ys)]
    perm_r = [rng.permutation(len(Y)) for Y in Ys]
    perm_f = [rng.permutation(len(Y)) for Y in Ys]
    K = Ys[0].shape[1]
    cols = rng.permutation(K)
    names = [f"L{k}" for k in range(K)]
    res = rg.compare(
        [P[p][:, cols] for P, p in zip(sequential, perm_r)],
        [P[p][:, cols] for P, p in zip(replay, perm_f)],
        Ys,
        {n: t for n, t in zip(names, theta)},
        seed=0,
        splits=SPLITS,
        record_ids={
            "labels": ids_lab,
            "ref": [[ids[i] for i in p] for ids, p in zip(ids_lab, perm_r)],
            "final": [[ids[i] for i in p] for ids, p in zip(ids_lab, perm_f)],
        },
        label_ids={"labels": names, "ref": [names[c] for c in cols], "final": [names[c] for c in cols]},
        labels_final=[Y[p][:, cols] for Y, p in zip(Ys, perm_f)],
    )
    for k in ("deficit", "attainable_gap", "mismatch_change"):
        assert res["components"][k] == base["components"][k]


def test_compare_refusals(two_finals):
    Ys, sequential, replay, theta = two_finals
    P, Q, Y = sequential[0], replay[0], Ys[0]
    n = len(Y)
    ids = list(range(n))
    with pytest.raises(ValueError, match="duplicate"):
        rg.compare(P, Q, Y, theta, seed=0, splits=2, record_ids=ids[:-1] + [0])
    with pytest.raises(ValueError, match="differ"):
        rg.compare(P, Q, Y, theta, seed=0, splits=2, record_ids={"labels": ids, "final": ids[:-1] + [n + 5]})
    with pytest.raises(ValueError, match="rows"):
        rg.compare(P, Q, Y, theta, seed=0, splits=2, record_ids=ids[:-1])
    Yf = Y.copy(); Yf[0, 0] = 1 - Yf[0, 0]
    with pytest.raises(ValueError, match="differ in 1 cells"):
        rg.compare(P, Q, Y, theta, seed=0, splits=2, labels_final=Yf)
    with pytest.raises(ValueError, match="shape"):
        rg.compare(P, Q[:-1], Y, theta, seed=0, splits=2)
    with pytest.raises(ValueError, match="one threshold per label"):
        rg.compare(P, Q, Y, theta[:-1], seed=0, splits=2)
    with pytest.raises(ValueError, match="all lists"):
        rg.compare([P], Q, [Y], theta, seed=0, splits=2)
    bad = P.copy(); bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rg.compare(bad, Q, Y, theta, seed=0, splits=2)
    with pytest.raises(ValueError, match="0/1"):
        rg.compare(P, Q, Y * 2, theta, seed=0, splits=2)
    with pytest.raises(ValueError, match="different scales"):          # logits against a probability rule
        rg.compare(np.log(P / (1 - P)), Q, Y, theta, seed=0, splits=2)
    # a rule outside [0, 1] needs score_scale='other' (symmetric with the scores)
    with pytest.raises(ValueError, match="score_scale='other'"):
        rg.compare(np.log(P / (1 - P)), np.log(Q / (1 - Q)), Y, theta - 1.0, seed=0, splits=2)
    with pytest.raises(ValueError, match="logit threshold against probabilities"):
        rg.compare(P, Q, Y, theta + 1.0, seed=0, splits=2)
    with pytest.warns(UserWarning, match="same scale"):
        rg.compare(np.log(P / (1 - P)), np.log(Q / (1 - Q)), Y, theta - 1.0, seed=0, splits=2, score_scale="other")
    with pytest.raises(ValueError, match="label sets"):
        rg.compare(P, Q, Y, theta, seed=0, splits=2,
                   label_ids={"labels": list("abcd"), "ref": list("abce")})
    with pytest.raises(ValueError, match="label_ids"):
        rg.compare(P, Q, Y, {"a": 0.5}, seed=0, splits=2)


def test_compare_other_scale_warns_and_is_invariant(two_finals):
    """A rule on the logit scale with logit scores: a warning, and the same numbers as on the
    probability scale (a strictly increasing map of scores and rule changes no level)."""
    Ys, sequential, replay, theta = two_finals
    lg = lambda p: np.log(p / (1 - p))
    base = rg.compare(sequential, replay, Ys, theta, seed=0, splits=SPLITS)
    with pytest.warns(UserWarning, match="same scale"):
        res = rg.compare([lg(P) for P in sequential], [lg(P) for P in replay], Ys, lg(theta), seed=0, splits=SPLITS,
                         score_scale="other")
    assert res["metadata"]["score_scale"] == "other"
    for k in ("deficit", "attainable_gap", "mismatch_change"):
        assert abs(res["components"][k] - base["components"][k]) <= EXACT


def test_compare_calibration_mask_and_absent_label(two_finals):
    Ys, sequential, replay, theta = two_finals
    P, Q, Y = sequential[0], replay[0], Ys[0].copy()
    m = np.zeros(len(Y), bool); m[::2] = True
    a = rg.compare(P, Q, Y, theta, seed=0, splits=SPLITS, calibration_mask=m)
    b = rg.compare(P[m], Q[m], Y[m], theta, seed=0, splits=SPLITS)
    assert a["components"] == b["components"]
    assert a["metadata"]["label_budget"][0]["records"] == m.sum()
    assert a["metadata"]["label_budget"][0]["records_before_mask"] == len(Y)
    Y[:, 2] = 0                                             # label 2 has no positive
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = rg.compare(P, Q, Y, theta, seed=0, splits=2)
    assert r["metadata"]["labels_without_positive"] == {0: [2]}
    with pytest.raises(ValueError, match="boolean"):
        rg.compare(P, Q, Y, theta, seed=0, splits=2, calibration_mask=m.astype(int))


# ------------------------------------------------------------------ method contrast ---
def test_method_contrast_matches_analyse():
    seeds = [0, 1, 2]
    runs_a, runs_b, ev_a, ev_b = [], [], {}, {}
    for s in seeds:
        Ys, Yval, Pval_first, ref, sequential, replay = example.make_seed(s)
        rules = {"inherited": crossfit.fit_validation_rule(Pval_first, Yval)}
        runs_a.append(rg.make_run(sequential, Ys, s, rules))
        runs_b.append(rg.make_run(replay, Ys, s, rules))
        r = rg.compare(sequential, replay, Ys, rules["inherited"], seed=s, splits=SPLITS, rule_name="inherited")
        ev_a[s], ev_b[s] = r["evaluations"]["ref"], r["evaluations"]["final"]
    res = rg.analyse(runs_a, runs_b, splits=SPLITS, mean_over=[0, 1, 2], names=("sequential", "replay"))
    mc = rg.method_contrast(ev_a, ev_b, "inherited", periods=[0, 1, 2], names=("sequential", "replay"))
    rows = {r["term"]: r for r in res["contrasts"] if r["comparison"] == "sequential minus replay"}
    for term, ours in (("at the rule", "lead_at_carried_rule"), ("after own-period refit", "lead_after_refit")):
        assert abs(rows[term]["mean"] - mc["terms"][ours]["mean"]) <= EXACT
        assert abs(rows[term]["ci95_low"] - mc["terms"][ours]["ci95"][0]) <= EXACT
    t = mc["terms"]
    assert abs(t["placement"]["mean"] - (t["lead_at_carried_rule"]["mean"] - t["lead_after_refit"]["mean"])) <= EXACT
    for s in seeds:
        assert abs(t["placement"]["per_seed"][s]
                   - (t["lead_at_carried_rule"]["per_seed"][s] - t["lead_after_refit"]["per_seed"][s])) <= EXACT
        assert mc["record_split_variation"]["values"]["lead_after_refit"][s] > 0
    assert "seed" in mc["training_run_uncertainty"] and "partition" in mc["record_split_variation"]["unit"]
    assert "not a standard error" in mc["record_split_variation"]["unit"]
    with pytest.raises(ValueError, match="different seeds"):
        rg.method_contrast(ev_a, {0: ev_b[0]}, "inherited")


def test_method_contrast_own_rules_matches_analyse():
    """same_rule=False: each method carries its own vector under one name, as analyse does."""
    seeds = [0, 1, 2]
    runs_a, runs_b, ev_a, ev_b = [], [], {}, {}
    for s in seeds:
        Ys, Yval, Pval_first, ref, sequential, replay = example.make_seed(s)
        th_a = crossfit.fit_validation_rule(Pval_first, Yval)
        th_b = th_a.copy()
        th_b[0] = min(th_b[0] + 0.05, 0.95)
        runs_a.append(rg.make_run(sequential, Ys, s, {"inherited": th_a}))
        runs_b.append(rg.make_run(replay, Ys, s, {"inherited": th_b}))
        ev_a[s] = rg.compare(sequential, replay, Ys, th_a, seed=s, splits=SPLITS, rule_name="inherited")["evaluations"][
            "ref"
        ]
        ev_b[s] = rg.compare(sequential, replay, Ys, th_b, seed=s, splits=SPLITS, rule_name="inherited")["evaluations"][
            "final"
        ]
    with pytest.raises(ValueError, match="different threshold vector"):
        rg.method_contrast(ev_a, ev_b, "inherited", periods=[0, 1, 2])
    res = rg.analyse(runs_a, runs_b, splits=SPLITS, mean_over=[0, 1, 2], names=("sequential", "replay"))
    mc = rg.method_contrast(ev_a, ev_b, "inherited", periods=[0, 1, 2], names=("sequential", "replay"), same_rule=False)
    assert not any(mc["same_rule_per_seed"].values())
    rows = {r["term"]: r for r in res["contrasts"] if r["comparison"] == "sequential minus replay"}
    for term, ours in (("at the rule", "lead_at_carried_rule"), ("after own-period refit", "lead_after_refit")):
        vals = [float(x) for x in rows[term]["values"].split()]
        assert max(abs(v - mc["terms"][ours]["per_seed"][s]) for v, s in zip(vals, seeds)) <= EXACT
        assert abs(rows[term]["ci95_high"] - mc["terms"][ours]["ci95"][1]) <= EXACT


# ------------------------------------------------------------------ identification example ---
def test_identification_example_exact():
    e = rg.identification_example()
    s, d = e["scorers"], e["decomposition"]
    assert [s[k]["carried_f1"] for k in ("reference", "final A", "final B")] == [Fr(1, 2), Fr(6, 13), Fr(6, 13)]
    assert [s[k]["best_f1"] for k in ("reference", "final A", "final B")] == [Fr(1, 2), Fr(1, 2), Fr(6, 13)]
    for k in s:
        assert s[k]["auroc"] == Fr(6, 11) and s[k]["average_precision"] == Fr(5, 18)
    assert d["final A"] == dict(deficit=Fr(1, 26), attainable_gap=Fr(0), mismatch_change=Fr(1, 26))
    assert d["final B"] == dict(deficit=Fr(1, 26), attainable_gap=Fr(1, 26), mismatch_change=Fr(0))
    # the same numbers from the package's float F1 and exact fitter, in sample (no cross-fitting:
    # fourteen records with three positives cannot be split into meaningful halves)
    th = np.array([e["threshold"]])
    for k, v in s.items():
        P, Y = v["scores"][:, None], v["labels"][:, None]
        assert int((P[:, 0] >= th[0]).sum()) == v["selected"]
        assert abs(fitters.macro_f1(P, Y, th) - float(v["carried_f1"])) <= EXACT
        assert abs(fitters.macro_f1(P, Y, fitters.fit_exact(P, Y)) - float(v["best_f1"])) <= EXACT
    skm = pytest.importorskip("sklearn.metrics")
    for v in s.values():
        assert abs(skm.roc_auc_score(v["labels"], v["scores"]) - 6 / 11) <= EXACT
        assert abs(skm.average_precision_score(v["labels"], v["scores"]) - 5 / 18) <= EXACT


# ------------------------------------------------------------------ count bounds ---
def test_f1_bounds_known_case():
    assert rg.f1_bounds_from_counts(40, 10, 10, 40, exact=True) == (Fr(4, 5), Fr(10, 11))
    lo, hi = rg.f1_bounds_from_counts(40, 10, 10, 40)
    assert lo == 0.8 and abs(hi - 10 / 11) <= EXACT
    assert rg.f1_bounds_from_counts(0, 3, 0, 4, exact=True) == (0, 0)
    # observed F1 0, but predicting every record is always open: lower bound 2P/(2P+N)
    assert rg.f1_bounds_from_counts(0, 5, 5, 0, exact=True) == (Fr(2, 3), Fr(10, 15))
    with pytest.raises(ValueError):
        rg.f1_bounds_from_counts(-1, 0, 0, 0)


def test_f1_bounds_sharp_by_enumeration():
    """For every n <= 10, every label sequence (a ranking up to the order of equal labels) and
    every observed cut s: the table at s and the best F1 over all cuts. The min and max of the
    best F1 per table must equal the bounds exactly, for every table that occurs."""
    checked = 0
    for n in range(1, 11):
        ext = {}
        for bits in itertools.product((0, 1), repeat=n):
            P = sum(bits)
            cum = [0]
            for b in bits:
                cum.append(cum[-1] + b)
            best = max((Fr(2 * cum[k], k + P) if k + P else Fr(0)) for k in range(n + 1))
            for s in range(n + 1):
                tp = cum[s]
                key = (tp, s - tp, P - tp, n - s - (P - tp))
                lo, hi = ext.get(key, (best, best))
                ext[key] = (min(lo, best), max(hi, best))
        # every table with these margins occurs
        assert len(ext) == sum(1 for t in itertools.product(range(n + 1), repeat=4) if sum(t) == n)
        for key, (lo, hi) in ext.items():
            assert rg.f1_bounds_from_counts(*key, exact=True) == (lo, hi), key
            checked += 1
    assert checked == sum(len([t for t in itertools.product(range(n + 1), repeat=4) if sum(t) == n])
                          for n in range(1, 11))


# ------------------------------------------------------------------ fixed calibration split ---
def _f1_frac(P, Y, th):
    """Macro-F1 as an exact Fraction from the confusion counts (no package code)."""
    out = Fr(0)
    for k in range(Y.shape[1]):
        pred = P[:, k] >= th[k]
        tp, npred, npos = int((pred & (Y[:, k] == 1)).sum()), int(pred.sum()), int(Y[:, k].sum())
        out += Fr(2 * tp, npred + npos) if npred + npos else Fr(0)
    return out / Y.shape[1]


def test_compare_fixed_split_identity_and_refusals(two_finals):
    Ys, sequential, replay, theta = two_finals
    P, Q, Y = sequential[0], replay[0], Ys[0]
    n = len(Y)
    perm = np.random.default_rng(5).permutation(n)
    cal, ev = perm[:300], perm[300:]
    r = rg.compare(P, Q, Y, theta, fixed_split={"calibration": cal, "evaluation": ev})
    assert "not the paper's cross-fitted estimator" in r["metadata"]["estimator"]
    assert "not the paper's cross-fitted estimator" in r["refitted"]["convention"]
    assert r["evaluations"] is None and r["metadata"]["label_budget"][0]["calibration_records"] == 300
    # the four levels in exact arithmetic, the rules fitted on the calibration records only
    Ur, Uf = _f1_frac(P[ev], Y[ev], theta), _f1_frac(Q[ev], Y[ev], theta)
    Vr = _f1_frac(P[ev], Y[ev], fitters.fit_exact(P[cal], Y[cal]))
    Vf = _f1_frac(Q[ev], Y[ev], fitters.fit_exact(Q[cal], Y[cal]))
    D, A, R = Ur - Uf, Vr - Vf, (Vf - Uf) - (Vr - Ur)
    assert D == A + R                                         # the identity, exactly
    c = r["components"]
    for got, want in ((c["deficit"], D), (c["attainable_gap"], A), (c["mismatch_change"], R)):
        assert abs(got - float(want)) <= 1e-15
    assert abs(c["deficit"] - (c["attainable_gap"] + c["mismatch_change"])) <= 1e-15
    # the same split given as masks and as record ids
    m_cal, m_ev = np.zeros(n, bool), np.zeros(n, bool); m_cal[cal] = True; m_ev[ev] = True
    ids = [f"r{i}" for i in range(n)]
    for fs, kw in (({"calibration": m_cal, "evaluation": m_ev}, {}),
                   ({"calibration": [ids[i] for i in cal], "evaluation": [ids[i] for i in ev]}, {"record_ids": ids})):
        assert rg.compare(P, Q, Y, theta, fixed_split=fs, **kw)["components"] == c
    # a record in both sets is refused
    with pytest.raises(ValueError, match="both the calibration and the evaluation"):
        rg.compare(P, Q, Y, theta, fixed_split={"calibration": cal, "evaluation": np.append(ev, cal[0])})
    with pytest.raises(ValueError, match="both the calibration and the evaluation"):
        rg.compare(P, Q, Y, theta, record_ids=ids,
                   fixed_split={"calibration": [ids[i] for i in cal], "evaluation": [ids[cal[3]]]})
    with pytest.raises(ValueError, match="not both"):
        rg.compare(P, Q, Y, theta, calibration_mask=m_cal | m_ev, fixed_split={"calibration": cal, "evaluation": ev})
    with pytest.raises(ValueError, match="seed is required"):
        rg.compare(P, Q, Y, theta)
    # several periods: the mean of the per-period components, identity per period
    cals = [np.arange(0, len(y), 2) for y in Ys]
    evs = [np.arange(1, len(y), 2) for y in Ys]
    rm = rg.compare(sequential, replay, Ys, theta, fixed_split={"calibration": cals, "evaluation": evs})
    per = [rm["per_period"][j]["components"]["deficit"] for j in range(len(Ys))]
    assert abs(rm["components"]["deficit"] - np.mean(per)) <= 1e-15
