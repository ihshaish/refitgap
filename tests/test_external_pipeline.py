"""A prediction pipeline that is not the paper's: scikit-learn logistic regression, one model
per label, on a synthetic four-period multi-label stream with drift. Its scores are written in
the documented stored-run format (P_<i>_<j>, Y_<j>, thresholds, Pval_<i>, Yval), with record
ids shuffled per model and put back in order by align_to, then analysed end to end with the
package only. Seconds to run; needs scikit-learn (the [test] extra)."""
import numpy as np
import pytest

import refitgap as rg
from refitgap import crossfit, inference

N, K, D, N_TEST, N_TRAIN, N_VAL = 4, 3, 8, 240, 300, 200


def period_data(rng, j, n):
    """Labels at drifting prevalence; features whose label directions rotate with the period."""
    prev = np.array([0.15, 0.30, 0.25]) * (1 + 0.2 * j)
    Y = (rng.random((n, K)) < np.minimum(prev, 0.9)).astype(int)
    base = np.random.default_rng(100).normal(0, 1, (K, D))      # fixed label directions
    rot = np.random.default_rng(200 + j).normal(0, 1, (K, D))   # period-specific part
    W = (1 - 0.25 * j) * base + 0.25 * j * rot
    X = Y @ W * 1.5 + rng.normal(0, 1, (n, D)) + 0.3 * j
    return X, Y


def fit_lr(X, Y):
    from sklearn.linear_model import LogisticRegression
    return [LogisticRegression(max_iter=500).fit(X, Y[:, k]) for k in range(K)]


def score(models, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models]).astype(np.float32)


def build_run(seed, cumulative, tmp_path):
    """Model i is fitted on period i's training records (sequential) or on periods 0..i (cumulative).
    Every model scores every period's test records; a validation split of period 0 fits theta_1."""
    rng = np.random.default_rng(seed)
    train = [period_data(rng, j, N_TRAIN) for j in range(N)]
    test = [period_data(rng, j, N_TEST) for j in range(N)]
    Xval, Yval = period_data(rng, 0, N_VAL)
    ids = [[f"p{j}r{r}" for r in range(N_TEST)] for j in range(N)]
    out = {}
    for i in range(N):
        Xtr = np.vstack([train[j][0] for j in range(i + 1)]) if cumulative else train[i][0]
        Ytr = np.vstack([train[j][1] for j in range(i + 1)]) if cumulative else train[i][1]
        m = fit_lr(Xtr, Ytr)
        for j in range(N):
            order = rng.permutation(N_TEST)                       # this pipeline writes rows in its own order
            shuffled_ids = [ids[j][r] for r in order]
            P_rows = score(m, test[j][0][order])
            out[f"P_{i}_{j}"] = rg.align_to(ids[j], shuffled_ids, P_rows)
        out[f"Pval_{i}"] = score(m, Xval)
    for j in range(N):
        out[f"Y_{j}"] = test[j][1].astype(np.int8)
    out["Yval"] = Yval.astype(np.int8)
    out["thresholds"] = crossfit.fit_validation_rule(out["Pval_0"], Yval)
    path = tmp_path / f"lr_{'cum' if cumulative else 'seq'}_s{seed}_scores.npz"
    np.savez_compressed(path, **out)
    return path


def test_sklearn_pipeline_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    seeds, hist = range(6), [0, 1, 2]
    per = {"seq": {}, "cum": {}}
    contrast = {}
    for s in seeds:
        paths = {"seq": build_run(s, False, tmp_path), "cum": build_run(s, True, tmp_path)}
        ev = {k: rg.evaluate_npz(str(p), s, splits=20, validation=True) for k, p in paths.items()}
        ref = rg.evaluate_npz(str(paths["seq"]), s, model="diagonal", splits=20)
        assert ref.single_model is False
        for k in ev:
            d = crossfit.decompose(ref, ev[k], "inherited", periods=hist)
            assert d["residual_D_A_R"] < 1e-12 and d["shared_terms"].startswith("omitted")
            per[k][s] = d
        m = crossfit.decompose(ev["cum"], ev["seq"], "inherited", periods=hist)
        assert m["shared_terms"] == "reported" and m["residual_R_C_M"] < 1e-12
        contrast[s] = m
        sm = ev["seq"].summary(periods=hist)
        assert sm["rules"]["inherited"]["identity_residual"] < 1e-12
        assert "validation" in sm["rules"]                       # same-scorer rule from Pval/Yval
        if s == 0:
            d = per["seq"][0]
            print(f"\n[external] sklearn LR, seq, seed 0, periods 0-2, theta_1: D {d['D']:+.4f} = "
                  f"A {d['A']:+.4f} + R {d['R']:+.4f}")
            print(f"[external] seq final against cumulative final: D {m['D']:+.4f} = A {m['A']:+.4f} + "
                  f"R {m['R']:+.4f}; R = dC {m['dC']:+.4f} + dM {m['dM']:+.4f}")
    _, x = inference.paired_vector({s: per["seq"][s]["D"] for s in seeds},
                                   {s: per["cum"][s]["D"] for s in seeds}, seeds=seeds)
    st = inference.family({"D": x})["D"]
    print(f"[external] D, seq minus cumulative, 6 seeds: mean {st['mean']:+.4f} "
          f"[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] class {st['class']}")
    assert st["class"] in {"material", "small non-zero", "equivalent", "unresolved"}
    assert all(np.isfinite(v) for v in x)
