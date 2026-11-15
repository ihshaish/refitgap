"""Stored run files (*_scores.npz): P_<i>_<j> is model i on period j's test records, Y_<j>
the labels, `thresholds` theta_1, Pval_<i> / Yval the validation scores. Model N-1 is final."""
import numpy as np


def n_periods(z):
    return max(int(k.split("_")[1]) for k in z.files if k.startswith("P_")) + 1


def load_run(path, model=None, periods=None):
    """npz path -> dict(Ps, Ys, period_ids, thresholds, Pval, Yval, N) for one model (default the
    final; "diagonal" gives P_j_j on period j)."""
    z = np.load(path)
    N = n_periods(z)
    periods = list(range(N)) if periods is None else [int(p) for p in periods]
    if model == "diagonal":
        Ps = [z[f"P_{j}_{j}"] for j in periods]
        mi = None
    else:
        mi = N - 1 if model is None else int(model)
        Ps = [z[f"P_{mi}_{j}"] for j in periods]
    Ys = [z[f"Y_{j}"] for j in periods]
    has_val = mi is not None and f"Pval_{mi}" in z.files and "Yval" in z.files
    return dict(Ps=Ps, Ys=Ys, period_ids=periods, N=N,
                thresholds=z["thresholds"].astype(np.float64) if "thresholds" in z.files else None,
                Pval=z[f"Pval_{mi}"] if has_val else None, Yval=z["Yval"] if has_val else None)


def evaluate_npz(path, seed, periods=None, model=None, splits=200, validation=False, scores_key="P",
                 shared_periods=None):
    """npz path and seed -> Evaluation of the final model (or `model`) at "inherited" and, with
    validation=True, the grid rule on its validation scores. Picklable, for process pools."""
    from .crossfit import evaluate, fit_validation_rule
    if scores_key == "P":
        r = load_run(path, model, periods)
    else:
        z = np.load(path)
        N = n_periods(z)
        mi = N - 1 if model is None else int(model)
        periods = list(range(N)) if periods is None else [int(p) for p in periods]
        r = dict(Ps=[z[f"{scores_key}_{mi}_{j}"] for j in periods], Ys=[z[f"Yx_{j}"] for j in periods],
                 period_ids=periods, thresholds=z["thresholds"].astype(np.float64), Pval=None, Yval=None)
    rules = {"inherited": r["thresholds"]}
    if validation and r["Pval"] is not None:
        rules["validation"] = fit_validation_rule(r["Pval"], r["Yval"])
    return evaluate(r["Ps"], r["Ys"], seed, r["period_ids"], rules=rules, splits=splits,
                    shared_periods=shared_periods, single_model=(model != "diagonal"))


def _evaluate_task(task):
    """Pool helper: task = (path, seed, periods, kwargs)."""
    path, seed, periods, kw = task
    return evaluate_npz(path, seed, periods, **kw)
