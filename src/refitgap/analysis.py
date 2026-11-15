"""One analysis path for the command line, the figures and the report.

analyse() evaluates one or two sets of runs (crossfit.evaluate), splits each deficit
(crossfit.decompose) and returns plain tables; write_tables() writes them as CSV and JSON.
Rows read "descriptive" unless the caller names a family fixed in advance, which then gets
the paper's class words (material, small non-zero, equivalent, unresolved, (non-)inferior)."""
import csv
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import crossfit, inference
from .crossfit import SPLITS, SPLIT_BASE
from .inference import MARGIN

U_CONVENTIONS = ("full", "half")
LEVEL_COLUMNS = ["method", "seed", "model", "period", "rule", "V_sep", "V_shared", "C", "U_half", "U_full",
                 "R_half", "R_full", "M", "direction", "sample_unit", "interval_type", "family", "adjustment",
                 "margin", "status"]
DECOMP_COLUMNS = ["method", "seed", "period", "rule", "U_convention", "U_ref", "U_final", "V_ref", "V_final",
                  "B", "A", "dR", "residual_B_A_dR", "direction", "sample_unit", "interval_type", "family",
                  "adjustment", "margin", "status"]
CONTRAST_COLUMNS = ["comparison", "term", "policy", "direction", "periods", "n", "seeds", "values", "mean", "ci95_low",
                    "ci95_high", "p_t_unadjusted", "p_t_holm", "class", "non_inferior_at_margin",
                    "inferior_at_margin", "sample_unit", "interval_type", "family", "adjustment", "margin",
                    "status"]
VOCABULARY_NOTE = (
    "The class words (material, small non-zero, equivalent, unresolved, inferior, non-inferior) are the "
    "vocabulary the paper fixed before its runs. Applied here they describe these numbers under the paper's "
    "rules and confer no confirmatory status on the data analysed."
)


# ------------------------------------------------------------------ runs ---
def make_run(final, labels, seed, rules, reference=None, period_ids=None, source="arrays", label_names=None):
    """One run: dict(seed, period_ids, final, labels, reference, rules, source, label_names).
    final, labels, reference: lists with one (n_j, K) array per period (reference: the
    reference model of each period on that period's records, or None)."""
    N = len(final)
    period_ids = list(range(N)) if period_ids is None else list(period_ids)
    for p in period_ids:
        crossfit.check_partition_key(seed, p)
    period_ids = [int(p) for p in period_ids]
    return dict(seed=int(seed), period_ids=period_ids, final=[np.asarray(P) for P in final],
                labels=[np.asarray(Y) for Y in labels],
                reference=None if reference is None else [np.asarray(P) for P in reference],
                rules={k: np.asarray(v, np.float64) for k, v in rules.items()}, source=str(source),
                label_names=None if label_names is None else [str(x) for x in label_names])


def run_from_npz(path, seed, periods=None, validation=False, reference=True):
    """A run from a stored-run file (*_scores.npz; see runs.load_run): the final model
    P_<N-1>_<j>, the per-period reference P_<j>_<j> (reference=True), the file's
    `thresholds` as rule "inherited" and, with validation=True, the grid rule fitted on the
    final model's validation scores as rule "validation" (as runs.evaluate_npz)."""
    from .runs import load_run
    r = load_run(path, None, periods)
    if r["thresholds"] is None:
        raise ValueError(f"{path}: no `thresholds` array (the inherited thresholds)")
    rules = {"inherited": r["thresholds"]}
    if validation and r["Pval"] is not None:
        rules["validation"] = crossfit.fit_validation_rule(r["Pval"], r["Yval"])
    ref = load_run(path, "diagonal", periods)["Ps"] if reference else None
    return make_run(r["Ps"], r["Ys"], seed, rules, ref, r["period_ids"], source=str(path))


def run_from_arrays(path, seed=None, reference=True):
    """A run from a per-period arrays file (npz): scores_<j> and labels_<j> for each period
    id j, optional reference_<j>, one or more rule_<name> vectors, optional `seed` (used
    when seed is None) and optional `label_names`. See save_arrays."""
    z = np.load(path, allow_pickle=False)
    ids = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("scores_"))
    if not ids:
        raise ValueError(f"{path}: no scores_<j> arrays")
    missing = [j for j in ids if f"labels_{j}" not in z.files]
    if missing:
        raise ValueError(f"{path}: labels_<j> missing for periods {missing}")
    rules = {k[len("rule_"):]: z[k] for k in z.files if k.startswith("rule_")}
    if not rules:
        raise ValueError(f"{path}: no rule_<name> vector")
    if seed is None:
        if "seed" not in z.files:
            raise ValueError(f"{path}: no `seed` array; pass the seed")
        seed = int(z["seed"])
    has_ref = reference and all(f"reference_{j}" in z.files for j in ids)
    ref = [z[f"reference_{j}"] for j in ids] if has_ref else None
    names = [str(x) for x in z["label_names"]] if "label_names" in z.files else None
    return make_run([z[f"scores_{j}"] for j in ids], [z[f"labels_{j}"] for j in ids], seed, rules, ref, ids,
                    source=str(path), label_names=names)


def save_arrays(path, run):
    """Write a run in the per-period arrays format read by run_from_arrays."""
    d = {f"scores_{j}": P for j, P in zip(run["period_ids"], run["final"])}
    d.update({f"labels_{j}": Y for j, Y in zip(run["period_ids"], run["labels"])})
    if run["reference"] is not None:
        d.update({f"reference_{j}": P for j, P in zip(run["period_ids"], run["reference"])})
    d.update({f"rule_{k}": v for k, v in run["rules"].items()})
    d["seed"] = np.array(run["seed"])
    if run["label_names"] is not None:
        d["label_names"] = np.array(run["label_names"])
    np.savez(path, **d)


# ------------------------------------------------------------- validation ---
def validate(runs_a, runs_b, rule):
    """Checks before any evaluation. Returns a list of dict(check, result, detail); raises
    ValueError on a failure that would make the tables meaningless."""
    out = []
    allruns = list(runs_a) + list(runs_b or [])
    if not runs_a:
        raise ValueError("no runs for method a")
    pid = allruns[0]["period_ids"]
    if len(set(pid)) != len(pid):
        raise ValueError("period ids must be distinct")
    bad = [r["source"] for r in allruns if r["period_ids"] != pid]
    if bad:
        raise ValueError(f"period ids differ between runs (e.g. {bad[0]})")
    out.append(dict(check="period ids", result="passed",
                    detail=f"ids {pid} in every run; the id keys the partition seed"))
    K = allruns[0]["final"][0].shape[1]
    for r in allruns:
        for P, Y, j in zip(r["final"], r["labels"], r["period_ids"]):
            if P.shape != Y.shape or P.shape[1] != K:
                raise ValueError(f"{r['source']}, period {j}: scores {P.shape} and labels {Y.shape} must be (n_j, {K})")
        if r["reference"] is not None:
            for P, Y, j in zip(r["reference"], r["labels"], r["period_ids"]):
                if P.shape != Y.shape:
                    raise ValueError(f"{r['source']}, period {j}: reference scores {P.shape} do not match labels")
    names = [r["label_names"] for r in allruns if r["label_names"] is not None]
    if names and any(n != names[0] for n in names):
        raise ValueError("label names or their order differ between runs")
    out.append(dict(check="label order", result="passed",
                    detail=f"K = {K} label columns in every period and run"
                           + (f"; names in one order: {', '.join(names[0])}" if names else
                              "; no label names supplied, so the column order is the caller's responsibility")))
    for r in allruns:
        if rule not in r["rules"]:
            raise ValueError(f"{r['source']}: rule {rule!r} not supplied (have {sorted(r['rules'])})")
    hashes = {}
    for side, runs in (("a", runs_a), ("b", runs_b or [])):
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"method {side}: a seed appears twice")
        for r in runs:
            hashes[(side, r["seed"])] = [crossfit.array_hash(np.asarray(Y).astype(int), np.int8) for Y in r["labels"]]
    nref = sum(r["reference"] is not None for r in allruns)
    out.append(dict(check="alignment", result="passed",
                    detail="records are paired by position; reference scores (where given) share the run's labels "
                           f"({nref} of {len(allruns)} runs carry a per-period reference)"))
    if runs_b:
        sa, sb = sorted(r["seed"] for r in runs_a), sorted(r["seed"] for r in runs_b)
        if sa != sb:
            raise ValueError(f"pairing: methods a and b hold different seeds ({sa} against {sb})")
        diff = [s for s in sa if hashes[("a", s)] != hashes[("b", s)]]
        if diff:
            raise ValueError(f"pairing: labels or record order differ between a and b for seeds {diff}")
        same_rule = all(np.array_equal(ra["rules"][rule], rb["rules"][rule])
                        for ra in runs_a for rb in runs_b if ra["seed"] == rb["seed"])
        out.append(dict(check="pairing", result="passed",
                        detail=f"seeds {sa} on both sides; labels equal in record order (label hash) for every seed; "
                               + ("rule '" + rule + "' is the same vector on both sides" if same_rule else
                                  "rule '" + rule + "' differs between a and b, so each model is scored at its own "
                                  "vector")))
    else:
        out.append(dict(check="pairing", result="not applicable",
                        detail="one method; contrasts are reference minus final within each run"))
    return out


# ------------------------------------------------------------- evaluation ---
def _evaluate_one(task):
    """Pool helper: (scores, labels, seed, period_ids, rules, kwargs) -> Evaluation."""
    Ps, Ys, seed, period_ids, rules, kw = task
    return crossfit.evaluate(Ps, Ys, seed, period_ids, rules=rules, **kw)


def _f(x):
    return None if x is None else float(x)


def analyse(runs_a, runs_b=None, rule="inherited", mean_over=None, splits=SPLITS, shared_periods=None,
            shared_weights=None, mean_weights=None, u_convention="full", margin=MARGIN, family=None,
            names=("a", "b"), workers=1):
    """Runs of method a (and optionally b, paired by seed) -> dict(settings, validation, levels,
    decomposition, contrasts, interpretation, evaluations).

    rule: the fixed rule; mean_over: period ids averaged; u_convention: "full" (U on the whole
    period, the paper's B = A + dR) or "half" (held-out halves, R = C + M per partition); margin:
    for TOST and the classes (0.005); family: None for descriptive rows, or a name to Holm-adjust
    the contrasts together."""
    if u_convention not in U_CONVENTIONS:
        raise ValueError(f"u_convention must be one of {U_CONVENTIONS}")
    runs_b = list(runs_b or [])
    runs_a = list(runs_a)
    validation = validate(runs_a, runs_b, rule)
    pid = runs_a[0]["period_ids"]
    mean_over = list(pid) if mean_over is None else [int(p) for p in mean_over]
    kw = dict(shared_periods=shared_periods, shared_weights=shared_weights, splits=splits)
    tasks, keys = [], []
    for side, runs in (("a", runs_a), ("b", runs_b)):
        for r in runs:
            tasks.append((r["final"], r["labels"], r["seed"], r["period_ids"], r["rules"], dict(kw)))
            keys.append((side, r["seed"], "final"))
            if r["reference"] is not None:
                tasks.append((r["reference"], r["labels"], r["seed"], r["period_ids"], r["rules"],
                              dict(kw, single_model=False)))
                keys.append((side, r["seed"], "reference"))
    if workers and workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            evs = list(ex.map(_evaluate_one, tasks))
    else:
        evs = [_evaluate_one(t) for t in tasks]
    E = dict(zip(keys, evs))
    label = {"a": names[0], "b": names[1]}
    meta = dict(direction="", sample_unit="", interval_type="none", family="none", adjustment="none",
                margin=float(margin), status="descriptive")

    levels = []
    for (side, seed, model), ev in E.items():
        pp = ev.per_period()
        sm = ev.summary(mean_over, mean_weights)
        rows = [(j, pp[j]) for j in ev.period_ids] + [("mean", None)]
        for j, p in rows:
            for rn in ev.rules:
                if p is not None:
                    vsep, vsh = p["V_sep"], p["V_shared"]
                    uh, uf = p["U_half"][rn], p["U_full"][rn]
                    C = None if vsh is None else vsep - vsh
                    M = None if vsh is None else vsh - uh
                    Rh = vsep - uh
                else:
                    vsep, vsh, C = sm["V_sep"], sm["V_shared"], sm["C"]
                    rr = sm["rules"][rn]
                    uh, uf, Rh, M = rr["U_half"], rr["U_full"], rr["R"], rr.get("M")
                if model == "reference":                 # one rule across different models: not reported
                    vsh = C = M = None
                levels.append(dict(meta, method=label[side], seed=seed, model=model,
                                   period=j if p is not None else "mean " + ",".join(map(str, mean_over)),
                                   rule=rn, V_sep=_f(vsep), V_shared=_f(vsh), C=_f(C), U_half=_f(uh), U_full=_f(uf),
                                   R_half=_f(Rh), R_full=_f(vsep - uf), M=_f(M), direction="level (no contrast)",
                                   sample_unit="mean over partitions of one run (split-half cross-fitting)"))

    halves = u_convention == "half"
    decomposition, decomp_by_run = [], {}
    for (side, seed, model), ev in E.items():
        if model != "reference":
            continue
        fin = E[(side, seed, "final")]
        per = {}
        for j in list(ev.period_ids) + ["mean"]:
            # one period has no weighting; weights=[1] keeps summary() from asking for weights
            # when the runs were evaluated with shared_weights (identical numbers otherwise)
            one = [1.0] if ev.shared_weights is not None else None
            d = crossfit.decompose(ev, fin, rule, periods=[j] if j != "mean" else mean_over,
                                   weights=one if j != "mean" else mean_weights, halves=halves)
            per[j] = d
            decomposition.append(dict(meta, method=label[side], seed=seed,
                                      period=j if j != "mean" else "mean " + ",".join(map(str, mean_over)),
                                      rule=rule, U_convention=d["U"], U_ref=d["U_ref"], U_final=d["U_final"],
                                      V_ref=d["V_ref"], V_final=d["V_final"], B=d["D"], A=d["A"], dR=d["R"],
                                      residual_B_A_dR=d["residual_D_A_R"], direction="reference minus final",
                                      sample_unit="mean over partitions of one run (split-half cross-fitting)"))
        decomp_by_run[(side, seed)] = per

    # contrasts: per-seed values over mean_over
    ukey = "U_half" if halves else "U_full"
    def per_seed(side, model, what):
        out = {}
        for (s_, seed, m_), ev in E.items():
            if s_ == side and m_ == model:
                sm = ev.summary(mean_over, mean_weights)
                out[seed] = sm["rules"][rule][ukey] if what == "U" else sm[what]
        return out

    comparisons = []
    if runs_b:
        cname = f"{label['a']} minus {label['b']}"
        comparisons.append((cname, f"{label['a']} minus {label['b']} (final models, paired by seed)", [
            ("at the rule", f"fixed rule '{rule}' ({ukey})", per_seed("a", "final", "U"), per_seed("b", "final", "U")),
            ("after own-period refit", "own-period rules (V_sep)", per_seed("a", "final", "V_sep"),
             per_seed("b", "final", "V_sep")),
            ("after one shared refit", "one shared rule (V_shared)", per_seed("a", "final", "V_shared"),
             per_seed("b", "final", "V_shared"))]))
    for side in ("a", "b"):
        if any(k[0] == side and k[2] == "reference" for k in E):
            cname = f"reference minus final ({label[side]})"
            comparisons.append((cname, f"per-period reference minus final model of {label[side]}, within each run", [
                ("at the rule", f"fixed rule '{rule}' ({ukey})", per_seed(side, "reference", "U"),
                 per_seed(side, "final", "U")),
                ("after own-period refit", "own-period rules (V_sep)", per_seed(side, "reference", "V_sep"),
                 per_seed(side, "final", "V_sep"))]))

    contrasts, interpretation = [], []
    for cname, direction, terms in comparisons:
        vecs = {}
        for term, policy, a, b in terms:
            if any(v is None for v in list(a.values()) + list(b.values())):
                continue                                       # shared level undefined for these periods
            seeds, x = inference.paired_vector(a, b, strict=True)
            vecs[term] = (policy, seeds, x)
        fam = None
        if family is not None and vecs and all(len(v[2]) >= 2 for v in vecs.values()):
            fam = inference.family_ni({t: v[2] for t, v in vecs.items()}, margin=margin)
        for term, (policy, seeds, x) in vecs.items():
            n = len(x)
            st = inference.paired_stats(x, margin) if n >= 2 else dict(n=n, mean=float(x.mean()))
            row = dict(meta, comparison=cname, term=term, policy=policy, direction=direction,
                       periods=",".join(map(str, mean_over)), n=n, seeds=" ".join(map(str, seeds)),
                       values=" ".join(repr(float(v)) for v in x),
                       mean=float(st["mean"]), ci95_low=st["ci95"][0] if n >= 2 else None,
                       ci95_high=st["ci95"][1] if n >= 2 else None,
                       p_t_unadjusted=st.get("p_t"), p_t_holm=None, non_inferior_at_margin=None,
                       inferior_at_margin=None,
                       sample_unit="seed (one value per run, paired)" if n >= 2 else "one run",
                       interval_type=("marginal 95% t interval over seeds" if n >= 2 else
                                      "none (one run; an interval needs two or more seeds)"))
            row["class"] = None
            if fam is not None:
                f = fam[term]
                row.update(p_t_holm=f["p_holm"], non_inferior_at_margin=bool(f["ni"]),
                           inferior_at_margin=bool(f["inf"]), family=str(family),
                           adjustment=f"Holm over {len(vecs)} contrasts", status=f"planned family {family}")
                row["class"] = f["class"]
            contrasts.append(row)
        interpretation.extend(_sentences(cname, contrasts))
    settings = dict(seeds=sorted({r["seed"] for r in runs_a}), splits=splits, split_base=SPLIT_BASE,
                    split_rule="partition b of period id j, seed s: b-th permutation of "
                               "numpy default_rng(split_base + 100*s + j); half 0 is the first n_j // 2 records",
                    rules={r["source"]: {k: v.tolist() for k, v in r["rules"].items()} for r in runs_a + runs_b},
                    rule=rule, period_ids=list(pid), mean_over=mean_over,
                    shared_periods=list(pid) if shared_periods is None else [int(p) for p in shared_periods],
                    shared_weights=None if shared_weights is None else [float(w) for w in shared_weights],
                    mean_weights=None if mean_weights is None else [float(w) for w in mean_weights],
                    U_convention=u_convention, margin=float(margin), family=family, names=list(names),
                    methods={label["a"]: [r["source"] for r in runs_a], label["b"]: [r["source"] for r in runs_b]}
                    if runs_b else {label["a"]: [r["source"] for r in runs_a]})
    from . import __version__
    settings["refitgap_version"] = __version__
    return dict(settings=settings, validation=validation, levels=levels, decomposition=decomposition,
                contrasts=contrasts, interpretation=interpretation, vocabulary_note=VOCABULARY_NOTE,
                evaluations=E, decomposition_by_run=decomp_by_run)


def _num(x):
    return f"{x:+.4f}"


def _sentences(cname, contrasts):
    rows = {r["term"]: r for r in contrasts if r["comparison"] == cname}
    if "at the rule" not in rows:
        return []
    x = rows["at the rule"]
    out = []
    for term in ("after own-period refit", "after one shared refit"):
        if term not in rows:
            continue
        y = rows[term]
        tail = ("the remaining lead is shown with its interval "
                f"[{y['ci95_low']:+.4f}, {y['ci95_high']:+.4f}]." if y["ci95_low"] is not None else
                "with one run no interval is computed.")
        out.append(f"{cname}: {term} the lead goes from {_num(x['mean'])} to {_num(y['mean'])}; {tail}")
    return out


# ------------------------------------------------------------------ output ---
def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return repr(float(v))        # float(): numpy >= 2 reprs np.float64 as 'np.float64(x)'
    return str(v)


def write_tables(result, outdir):
    """levels.csv, decomposition.csv, contrasts.csv and result.json in outdir. Floats are
    written with repr, so a CSV value read back with float() equals the API value exactly."""
    from pathlib import Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, cols in (("levels", LEVEL_COLUMNS), ("decomposition", DECOMP_COLUMNS), ("contrasts", CONTRAST_COLUMNS)):
        p = outdir / f"{name}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in result[name]:
                w.writerow([_cell(r.get(c)) for c in cols])
        paths[name] = p
    p = outdir / "result.json"
    p.write_text(json.dumps(to_json(result), indent=1))
    paths["json"] = p
    return paths


def to_json(result):
    keep = ("settings", "validation", "levels", "decomposition", "contrasts", "interpretation", "vocabulary_note")
    return {k: result[k] for k in keep}
