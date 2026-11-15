"""Two scorers on the same records at one carried rule, and the paired method contrast.

compare() takes a reference and a final scorer on the same labelled records and one carried
threshold vector; method_contrast() gives method a's lead over b, carried and refitted, paired
by seed. Every level comes from evaluate(), every component from decompose(), every interval
from inference.paired_contrast()."""
import warnings

import numpy as np

from . import inference
from .crossfit import IDENTITY_TOL, SPLITS, SPLIT_BASE, decompose, evaluate
from .fitters import fit_exact, macro_f1
from .records import align_to

TIE_CONVENTION = ("a record is predicted positive when score >= threshold; fitted cutpoints are score values, so "
                  "tied records are predicted together; when two cutpoints give the same F1 the own-period fitter "
                  "chooses the higher; the shared fitter maximises a float sum over periods, so two cutpoints whose "
                  "objectives are equal in exact arithmetic may differ in the last bit and either can be chosen")
ABSENT_LABEL_CONVENTION = ("a label with no positive in a period has F1 0 there whatever is predicted "
                           "(zero_division = 0) and still counts in the macro mean; its fitted cutpoint lies above "
                           "every score")
SPLIT_RULE = ("partition b of period id j, seed s: b-th permutation of numpy default_rng(split_base + 100*s + j); "
              "half 0 is the first n_j // 2 records; every rule is fitted on one half and scored on the other, "
              "in both directions")
SIGNS = dict(deficit="U_ref - U_final (positive: the final scorer is worse at the carried rule)",
             attainable_gap="V_ref - V_final (positive: the final scorer is worse after its own refit)",
             mismatch_change="R_final - R_ref, R_x = V_x - U_x (positive: the carried rule fits the final "
                             "scorer worse than it fits the reference)",
             identity="deficit = attainable_gap + mismatch_change")


# ------------------------------------------------------------------ input handling ---
def _periods(x, what):
    """One (n, K) array, or a list/tuple of them (one per period). Returns (list, single).
    A list or tuple always means one matrix per period, so a single matrix written as a
    nested Python list (rows of numbers) is refused rather than read as one period per row."""
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError(f"{what}: empty list of periods")
        arrs = [np.asarray(a) for a in x]
        if any(a.ndim == 0 for a in arrs):
            raise ValueError(f"{what}: a list of numbers, not a list of per-period matrices; pass one period as a "
                             "numpy array, or several as a list with one (n_j, K) matrix per period")
        if all(a.ndim == 1 for a in arrs) and len({a.shape for a in arrs}) == 1:
            raise ValueError(f"{what}: a nested list that reads as one {len(arrs)} x {arrs[0].shape[0]} matrix, "
                             "while a list means one matrix per period; pass np.asarray(...) for one period, or a "
                             "list with one 2-d (n_j, K) matrix per period (an (n_j, 1) matrix for one label)")
        return arrs, False
    return [np.asarray(x)], True


def _as_matrix(a, what):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{what}: need a (records, labels) matrix, got shape {a.shape}")
    return a


def _ids_per_period(ids, N, single, what):
    if single:
        return [list(ids)]
    if not isinstance(ids, (list, tuple)) or len(ids) != N:
        raise ValueError(f"{what}: one id sequence per period ({N} periods)")
    return [list(x) for x in ids]


def _check_ids(ref_ids, ids, what, period):
    if len(set(ids)) != len(ids):
        dup = next(i for i in ids if ids.count(i) > 1)
        raise ValueError(f"period {period}: duplicate record id in {what} (e.g. {dup!r})")
    a, b = set(ref_ids), set(ids)
    if a != b:
        raise ValueError(f"period {period}: the record ids of {what} and of the labels differ "
                         f"({len(b - a)} only in {what}, {len(a - b)} only in the labels); "
                         "compare() needs the same records for both scorers")


def _column_order(names_lab, names, what):
    names_lab, names = list(names_lab), list(names)
    for nm, w in ((names_lab, "labels"), (names, what)):
        if len(set(nm)) != len(nm):
            raise ValueError(f"label_ids of {w} contain duplicates")
    if set(names_lab) != set(names):
        raise ValueError(f"label_ids of {what} and of the labels are different label sets")
    pos = {n: i for i, n in enumerate(names)}
    return np.array([pos[n] for n in names_lab], dtype=np.int64)


def _prepare(scores_ref, scores_final, labels, rule, record_ids, label_ids, calibration_mask, labels_final,
             period_ids, score_scale=None):
    P_ref, single = _periods(scores_ref, "scores_ref")
    P_fin, s2 = _periods(scores_final, "scores_final")
    Y, s3 = _periods(labels, "labels")
    if not (single == s2 == s3) or not (len(P_ref) == len(P_fin) == len(Y)):
        raise ValueError("scores_ref, scores_final and labels must all be one matrix, or all lists with one "
                         "matrix per period")
    N = len(Y)
    Yf = None if labels_final is None else _periods(labels_final, "labels_final")[0]
    if Yf is not None and len(Yf) != N:
        raise ValueError("labels_final: one matrix per period")
    period_ids = list(range(N)) if period_ids is None else [int(p) for p in period_ids]
    if len(period_ids) != N:
        raise ValueError(f"period_ids: one id per period ({N})")
    P_ref = [_as_matrix(a, "scores_ref") for a in P_ref]
    P_fin = [_as_matrix(a, "scores_final") for a in P_fin]
    Y = [_as_matrix(a, "labels") for a in Y]
    Yf = None if Yf is None else [_as_matrix(a, "labels_final") for a in Yf]

    # records: align the scorers (and labels_final, which travels with the final scorer) to the labels' ids
    if record_ids is not None:
        rid = record_ids if isinstance(record_ids, dict) else dict(labels=record_ids)
        bad = set(rid) - {"ref", "final", "labels"}
        if bad or "labels" not in rid:
            raise ValueError("record_ids: one id sequence (shared by every input) or a dict with key 'labels' "
                             "and optional keys 'ref' and 'final'")
        ids_lab = _ids_per_period(rid["labels"], N, single, "record_ids['labels']")
        ids_ref = _ids_per_period(rid.get("ref", rid["labels"]), N, single, "record_ids['ref']")
        ids_fin = _ids_per_period(rid.get("final", rid["labels"]), N, single, "record_ids['final']")
        for c in range(N):
            j = period_ids[c]
            for ids, a, w in ((ids_lab[c], Y[c], "labels"), (ids_ref[c], P_ref[c], "scores_ref"),
                              (ids_fin[c], P_fin[c], "scores_final")):
                if len(ids) != a.shape[0]:
                    raise ValueError(f"period {j}: {w} has {a.shape[0]} rows for {len(ids)} record ids")
            _check_ids(ids_lab[c], ids_lab[c], "the labels", j)
            _check_ids(ids_lab[c], ids_ref[c], "scores_ref", j)
            _check_ids(ids_lab[c], ids_fin[c], "scores_final", j)
            P_ref[c] = align_to(ids_lab[c], ids_ref[c], P_ref[c])
            if Yf is not None:
                P_fin[c], Yf[c] = align_to(ids_lab[c], ids_fin[c], P_fin[c], Yf[c])
            else:
                P_fin[c] = align_to(ids_lab[c], ids_fin[c], P_fin[c])

    # labels (columns): reorder the scorers' columns, and a rule given by name, to the labels' column order
    names = None
    if label_ids is not None:
        lid = label_ids if isinstance(label_ids, dict) else dict(labels=label_ids)
        bad = set(lid) - {"ref", "final", "labels"}
        if bad or "labels" not in lid:
            raise ValueError("label_ids: one name sequence (shared by every input) or a dict with key 'labels' "
                             "and optional keys 'ref' and 'final'")
        names = [x for x in lid["labels"]]
        o_ref = _column_order(names, lid.get("ref", names), "scores_ref")
        o_fin = _column_order(names, lid.get("final", names), "scores_final")
        for c in range(N):
            for a, w in ((Y[c], "labels"), (P_ref[c], "scores_ref"), (P_fin[c], "scores_final")):
                if a.shape[1] != len(names):
                    raise ValueError(f"period {period_ids[c]}: {w} has {a.shape[1]} columns for {len(names)} label ids")
            P_ref[c], P_fin[c] = P_ref[c][:, o_ref], P_fin[c][:, o_fin]
            if Yf is not None:
                Yf[c] = Yf[c][:, o_fin]
    if isinstance(rule, dict):
        if names is None:
            raise ValueError("a rule given as {label: threshold} needs label_ids")
        if set(rule) != set(names):
            raise ValueError("the rule's labels and the labels' label_ids are different label sets")
        rule = np.array([rule[n] for n in names], np.float64)
    rule = np.asarray(rule, np.float64).reshape(-1)

    # shapes, labels, scores
    K = Y[0].shape[1]
    for c in range(N):
        j = period_ids[c]
        for a, w in ((P_ref[c], "scores_ref"), (P_fin[c], "scores_final")):
            if a.shape != Y[c].shape:
                raise ValueError(f"period {j}: {w} has shape {a.shape}, the labels {Y[c].shape}")
        if Y[c].shape[1] != K:
            raise ValueError(f"period {j}: {Y[c].shape[1]} labels, period {period_ids[0]} has {K}")
        if not np.isin(Y[c], (0, 1)).all():
            raise ValueError(f"period {j}: labels must be 0/1 (remove unlabelled records with drop_unlabelled)")
        if Yf is not None:
            if Yf[c].shape != Y[c].shape:
                raise ValueError(f"period {j}: labels_final has shape {Yf[c].shape}, the labels {Y[c].shape}")
            ndiff = int((np.asarray(Yf[c]) != np.asarray(Y[c])).sum())
            if ndiff:
                raise ValueError(f"period {j}: the two label matrices differ in {ndiff} cells after alignment; "
                                 "both scorers must be evaluated against the same labels")
    if rule.shape != (K,):
        raise ValueError(f"rule: one threshold per label ({K}), got {rule.shape[0]}")
    if not np.all(np.isfinite(rule)):
        raise ValueError("rule: non-finite threshold")

    # calibration mask: which records' labels the evaluation may use (in the labels' record order)
    n_all = [int(y.shape[0]) for y in Y]
    if calibration_mask is not None:
        masks = [np.asarray(m) for m in (calibration_mask if not single else [calibration_mask])]
        if len(masks) != N:
            raise ValueError("calibration_mask: one boolean vector per period")
        for c in range(N):
            m = masks[c]
            if m.dtype != bool or m.shape != (n_all[c],):
                raise ValueError(f"period {period_ids[c]}: calibration_mask must be a boolean vector of "
                                 f"length {n_all[c]} (the labels' records, in the labels' order)")
            P_ref[c], P_fin[c], Y[c] = P_ref[c][m], P_fin[c][m], Y[c][m]

    scale = "probability"
    for c in range(N):
        j = period_ids[c]
        if Y[c].shape[0] < 2:
            raise ValueError(f"period {j}: needs at least 2 labelled records to split in half")
        for a, w in ((P_ref[c], "scores_ref"), (P_fin[c], "scores_final")):
            if not np.all(np.isfinite(a)):
                raise ValueError(f"period {j}: {w} has non-finite scores")
    outside = [w for c in range(N) for a, w in ((P_ref[c], "scores_ref"), (P_fin[c], "scores_final"))
               if a.size and (a.min() < 0 or a.max() > 1)]
    if score_scale not in (None, "other"):
        raise ValueError("score_scale: None (probability scale expected) or 'other'")
    rule_out = bool(np.any((rule < 0) | (rule > 1)))
    if outside or rule_out or score_scale == "other":
        if score_scale is None and not rule_out:
            raise ValueError(f"{sorted(set(outside))[0]} has scores outside [0, 1] while the rule lies in [0, 1]: "
                             "the scores and the carried rule look like different scales (for example logits "
                             "against a probability threshold). Map the scores onto the rule's scale first, "
                             "or pass score_scale='other' if both are on one other scale.")
        if score_scale is None:
            what = ("the rule lies outside [0, 1] while every score lies in [0, 1]: the scores and the carried rule "
                    "look like different scales (for example a logit threshold against probabilities)" if not outside
                    else "the scores and the rule lie outside [0, 1], so they are not on the probability scale")
            raise ValueError(f"{what}. Map them onto one scale first, or pass score_scale='other' if both are on one "
                             "other scale.")
        scale = "other"
        warnings.warn("scores or rule not on the probability scale; the levels are valid only if the scores and "
                      "the carried rule are on the same scale", UserWarning, stacklevel=3)
    return P_ref, P_fin, [y.astype(int) for y in Y], rule, names, period_ids, single, n_all, scale


# ------------------------------------------------------------------ compare ---
def _components(d):
    out = dict(deficit=d["D"], attainable_gap=d["A"], mismatch_change=d["R"], mismatch_ref=d["R_ref"],
               mismatch_final=d["R_final"])
    if d.get("shared_terms") == "reported":
        out.update(attainable_gap_shared=d["A_shared"], shared_cost_change=d["dC"], shared_mismatch_change=d["dM"],
                   shared_cost_ref=d["C_ref"], shared_cost_final=d["C_final"], shared_mismatch_ref=d["M_ref"],
                   shared_mismatch_final=d["M_final"])
    return out


def _block(d, sr, sf, rule_name):
    out = dict(carried=dict(ref=d["U_ref"], final=d["U_final"], convention=d["U"],
                            ref_half=sr["rules"][rule_name]["U_half"], final_half=sf["rules"][rule_name]["U_half"],
                            ref_full=sr["rules"][rule_name]["U_full"], final_full=sf["rules"][rule_name]["U_full"]),
               refitted=dict(ref=d["V_ref"], final=d["V_final"]),
               components=_components(d))
    if sr["V_shared"] is not None and sf["V_shared"] is not None:
        out["refitted_shared"] = dict(ref=sr["V_shared"], final=sf["V_shared"])
    return out


def compare(scores_ref, scores_final, labels, rule, *, record_ids=None, label_ids=None, seed=None, splits=SPLITS,
            calibration_mask=None, labels_final=None, period_ids=None, halves=False, rule_name="carried",
            score_scale=None, fixed_split=None, fixed_split_by=None):
    """Two (n, K) score matrices, (n, K) 0/1 labels and a (K,) carried rule (lists for several
    periods) -> dict(carried, refitted, components, per_period, metadata, decomposition,
    evaluations): both scorers' carried and refitted macro-F1, and deficit = attainable gap +
    change in mismatch.

    record_ids / label_ids: one sequence (inputs already in that order) or a dict of each
    input's own ids, which are aligned to the labels; duplicates or mismatched sets raise.
    seed keys the partitions (required unless fixed_split). splits: partitions per period.
    calibration_mask: records whose labels may be used. labels_final: the labels exported with
    the final scorer, checked cell by cell. halves: carried level on held-out halves.
    score_scale=None wants scores and rule in [0, 1]; "other" declares another common scale.
    fixed_split={"calibration": c, "evaluation": e} (masks, positions or ids; fixed_split_by
    says which) swaps the cross-fitted estimator for one fixed split. Labels with no positive
    take F1 0 and stay in the macro mean."""
    from . import __version__
    P_ref, P_fin, Y, rule, names, period_ids, single, n_all, scale = _prepare(
        scores_ref, scores_final, labels, rule, record_ids, label_ids, calibration_mask, labels_final, period_ids,
        score_scale)
    if fixed_split is not None:
        return _compare_fixed(P_ref, P_fin, Y, rule, names, period_ids, single, n_all, scale, record_ids,
                              calibration_mask, fixed_split, rule_name, __version__, fixed_split_by)
    if fixed_split_by is not None:
        raise ValueError("fixed_split_by applies only with fixed_split")
    if seed is None:
        raise ValueError("seed is required for the cross-fitted estimator (it keys the partitions)")
    rules = {rule_name: rule}
    ev_ref = evaluate(P_ref, Y, seed, period_ids, rules=rules, splits=splits)
    ev_fin = evaluate(P_fin, Y, seed, period_ids, rules=rules, splits=splits)
    d = decompose(ev_ref, ev_fin, rule_name, halves=halves)
    out = _block(d, ev_ref.summary(), ev_fin.summary(), rule_name)
    out["per_period"] = {}
    for j in period_ids:
        dj = decompose(ev_ref, ev_fin, rule_name, periods=[j], halves=halves)
        out["per_period"][j] = _block(dj, ev_ref.summary([j]), ev_fin.summary([j]), rule_name)
    K = Y[0].shape[1]
    lab = (lambda k: names[k]) if names is not None else (lambda k: k)
    absent = {j: [lab(k) for k in range(K) if Y[c][:, k].sum() == 0] for c, j in enumerate(period_ids)}
    budget = {j: dict(records=int(len(Y[c])), records_before_mask=n_all[c],
                      fit_records_per_direction=[int(len(Y[c]) // 2), int(len(Y[c]) - len(Y[c]) // 2)],
                      positives_per_label=[int(v) for v in Y[c].sum(0)])
              for c, j in enumerate(period_ids)}
    out["metadata"] = dict(
        rule_name=rule_name, rule=rule.tolist(), label_ids=names, n_labels=K, period_ids=period_ids,
        single_period=single, seed=int(seed), splits=int(splits), split_base=SPLIT_BASE, split_rule=SPLIT_RULE,
        carried_level=d["U"], label_budget=budget,
        label_budget_note=("labelled records each refit is fitted on, per period and direction (half 0 then half 1); "
                           "the shared rule pools the fitting halves of all periods"),
        calibration_mask_used=calibration_mask is not None, ties=TIE_CONVENTION,
        absent_labels=ABSENT_LABEL_CONVENTION, labels_without_positive={j: v for j, v in absent.items() if v},
        periods_without_positive=[j for c, j in enumerate(period_ids) if Y[c].sum() == 0],
        score_scale=scale, signs=SIGNS, shared_terms=d["shared_terms"],
        estimator="crossfit.evaluate on each scorer, crossfit.decompose on the pair (split-half cross-fitting)",
        package_version=__version__)
    out["decomposition"] = d
    out["evaluations"] = dict(ref=ev_ref, final=ev_fin)
    return out


# ------------------------------------------------------------------ fixed calibration split ---
FIXED_SPLIT_LABEL = "fixed calibration split, not the paper's cross-fitted estimator"


FIXED_SPLIT_BY = (None, "ids", "positions")


def _is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _positions(sel, n, ids, what, j, by=None):
    """One fixed_split set -> row positions. Masks are masks; otherwise `by` ("positions" or
    "ids", or a guess when None). Values valid both ways that select different rows raise."""
    sel = np.asarray(sel)
    if sel.dtype == bool:
        if sel.shape != (n,):
            raise ValueError(f"period {j}: fixed_split['{what}'] mask must have length {n}")
        return np.flatnonzero(sel)
    if by is None:
        if ids is None or (np.issubdtype(sel.dtype, np.integer) and not all(_is_int(r) for r in ids)):
            by = "positions"
        elif not np.issubdtype(sel.dtype, np.integer):
            by = "ids"
        else:
            pos = {r: i for i, r in enumerate(ids)}
            vals = sel.reshape(-1).tolist()
            if all(v in pos for v in vals) and all(0 <= v < n for v in vals) and any(pos[v] != v for v in vals):
                raise ValueError(f"period {j}: fixed_split['{what}'] holds integers that are valid record ids and "
                                 "valid row positions, and the two readings select different records; pass "
                                 "fixed_split_by='ids' or fixed_split_by='positions'")
            by = "ids"
    if by == "ids" and ids is None:
        raise ValueError("fixed_split_by='ids' needs record_ids")
    if by == "ids":
        pos = {r: i for i, r in enumerate(ids)}
        miss = [r for r in sel.tolist() if r not in pos]
        if miss:
            raise ValueError(f"period {j}: fixed_split['{what}'] has unknown record id {miss[0]!r}")
        out = np.array([pos[r] for r in sel.tolist()], dtype=np.int64)
    else:
        out = sel.astype(np.int64).reshape(-1)
        if out.size and (out.min() < 0 or out.max() >= n):
            raise ValueError(f"period {j}: fixed_split['{what}'] positions outside 0..{n - 1}")
    if len(set(out.tolist())) != len(out):
        raise ValueError(f"period {j}: fixed_split['{what}'] lists a record twice")
    return out


def _compare_fixed(P_ref, P_fin, Y, rule, names, period_ids, single, n_all, scale, record_ids, calibration_mask,
                   fixed_split, rule_name, version, fixed_split_by=None):
    if fixed_split_by not in FIXED_SPLIT_BY:
        raise ValueError("fixed_split_by: None, 'ids' or 'positions'")
    if calibration_mask is not None:
        raise ValueError("pass either calibration_mask or fixed_split, not both")
    if not isinstance(fixed_split, dict) or set(fixed_split) != {"calibration", "evaluation"}:
        raise ValueError("fixed_split: {'calibration': ..., 'evaluation': ...}")
    N = len(Y)
    cal, ev = fixed_split["calibration"], fixed_split["evaluation"]
    cal, ev = ([cal], [ev]) if single else (list(cal), list(ev))
    if len(cal) != N or len(ev) != N:
        raise ValueError(f"fixed_split: one calibration and one evaluation set per period ({N})")
    ids = None
    if record_ids is not None:
        lab = record_ids["labels"] if isinstance(record_ids, dict) else record_ids
        ids = [list(lab)] if single else [list(x) for x in lab]
    per, budget = {}, {}
    for c, j in enumerate(period_ids):
        n = len(Y[c])
        ic = _positions(cal[c], n, None if ids is None else ids[c], "calibration", j, fixed_split_by)
        ie = _positions(ev[c], n, None if ids is None else ids[c], "evaluation", j, fixed_split_by)
        both = np.intersect1d(ic, ie)
        if both.size:
            r = both[0] if ids is None else ids[c][both[0]]
            raise ValueError(f"period {j}: {both.size} record(s) in both the calibration and the evaluation set "
                             f"(e.g. {r!r}); the two sets must be disjoint")
        if ic.size == 0 or ie.size == 0:
            raise ValueError(f"period {j}: the calibration and the evaluation set must both be non-empty")
        lv = {}
        for side, P in (("ref", P_ref[c]), ("final", P_fin[c])):
            th = fit_exact(P[ic], Y[c][ic])
            lv["V_" + side] = macro_f1(P[ie], Y[c][ie], th)
            lv["U_" + side] = macro_f1(P[ie], Y[c][ie], rule)
        per[j] = lv
        budget[j] = dict(records=n, calibration_records=int(ic.size), evaluation_records=int(ie.size),
                         unused_records=int(n - ic.size - ie.size),
                         calibration_positives_per_label=[int(v) for v in Y[c][ic].sum(0)])

    def block(levels):
        Ur, Uf, Vr, Vf = levels["U_ref"], levels["U_final"], levels["V_ref"], levels["V_final"]
        D, A = Ur - Uf, Vr - Vf
        Rr, Rf = Vr - Ur, Vf - Uf
        comp = dict(deficit=D, attainable_gap=A, mismatch_change=Rf - Rr, mismatch_ref=Rr, mismatch_final=Rf)
        res = abs(D - (A + comp["mismatch_change"]))
        if res > IDENTITY_TOL:
            raise AssertionError(f"deficit != attainable gap + change in mismatch: {res:.3g}")
        return dict(carried=dict(ref=Ur, final=Uf, convention="U on the evaluation records"),
                    refitted=dict(ref=Vr, final=Vf, convention=FIXED_SPLIT_LABEL), components=comp)

    mean = {k: float(np.mean([per[j][k] for j in period_ids])) for k in per[period_ids[0]]}
    out = block(mean)
    out["per_period"] = {j: block(per[j]) for j in period_ids}
    K = Y[0].shape[1]
    lab_ = (lambda k: names[k]) if names is not None else (lambda k: k)
    out["metadata"] = dict(
        rule_name=rule_name, rule=rule.tolist(), label_ids=names, n_labels=K, period_ids=period_ids,
        single_period=single, seed=None, splits=None, carried_level="U on the evaluation records",
        label_budget=budget, label_budget_note="labelled records the refit is fitted on (calibration), per period",
        ties=TIE_CONVENTION, absent_labels=ABSENT_LABEL_CONVENTION,
        labels_without_positive={j: v for j, v in
                                 {j: [lab_(k) for k in range(K) if Y[c][:, k].sum() == 0]
                                  for c, j in enumerate(period_ids)}.items() if v},
        score_scale=scale, signs=SIGNS, shared_terms="not computed under a fixed calibration split",
        estimator=FIXED_SPLIT_LABEL + " (fitters.fit_exact on the calibration records, macro_f1 on the evaluation "
                                      "records; periods averaged uniformly)",
        package_version=version)
    out["decomposition"] = None
    out["evaluations"] = None
    return out


# ------------------------------------------------------------------ method contrast ---
def _per_partition(ev, key, rule, cols, w):
    X = ev.sep if key == "V_sep" else ev.U_half[rule]
    H = 0.5 * (X[:, 0, cols] + X[:, 1, cols])
    return H @ w


TERMS = ("lead_at_carried_rule", "lead_after_refit", "placement")


def method_contrast(evals_a, evals_b, rule, *, periods=None, weights=None, halves=False, names=("a", "b"),
                    seeds=None, same_rule=True):
    """{seed: Evaluation} for methods a and b, and the carried rule's name -> leads of a over b
    at the carried rule (G_U), after both refit (G_V), and the change after refitting
    (change = G_V - G_U, as in the paper), each with mean, 95 % t interval over seeds and
    unadjusted class. Positive means a is ahead. "placement" (G_U - G_V) is the negative of
    the change, kept for compatibility. The spread over
    partitions within each seed is reported apart and never added to the seed interval.
    same_rule=False lets each method carry its own vector under that name."""
    if set(evals_a) != set(evals_b) and seeds is None:
        raise ValueError(f"the two methods hold different seeds ({sorted(evals_a)} against {sorted(evals_b)})")
    seeds = sorted(evals_a) if seeds is None else sorted(int(s) for s in seeds)
    per_seed, vals, spread = {}, {t: {} for t in TERMS}, {t: {} for t in TERMS}
    for s in seeds:
        a, b = evals_a[s], evals_b[s]
        d = decompose(a, b, rule, periods=periods, weights=weights, halves=halves, same_rule=same_rule)
        per_seed[s] = d
        vals["lead_at_carried_rule"][s], vals["lead_after_refit"][s], vals["placement"][s] = d["D"], d["A"], d["R"]
        P, cols = a._cols(periods)
        ww = weights
        if ww is None and a.shared_weights is not None and P == list(a.shared_periods):
            ww = a.shared_weights                       # summary()'s default on the shared periods
        w = np.full(len(cols), 1.0 / len(cols)) if ww is None else np.asarray(ww, float) / np.sum(ww)
        u = _per_partition(a, "U", rule, cols, w) - _per_partition(b, "U", rule, cols, w)
        v = _per_partition(a, "V_sep", rule, cols, w) - _per_partition(b, "V_sep", rule, cols, w)
        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else None
        spread["lead_at_carried_rule"][s], spread["lead_after_refit"][s], spread["placement"][s] = (
            sd(u),
            sd(v),
            sd(u - v),
        )
    U_a = {s: per_seed[s]["U_ref"] for s in seeds}
    U_b = {s: per_seed[s]["U_final"] for s in seeds}
    V_a = {s: per_seed[s]["V_ref"] for s in seeds}
    V_b = {s: per_seed[s]["V_final"] for s in seeds}
    terms = dict(lead_at_carried_rule=inference.paired_contrast(U_a, U_b, seeds=seeds),
                 lead_after_refit=inference.paired_contrast(V_a, V_b, seeds=seeds),
                 # the paired difference of the two leads, seed by seed
                 placement=inference.paired_contrast(vals["lead_at_carried_rule"], vals["lead_after_refit"],
                                                     seeds=seeds))
    for t in TERMS:
        terms[t]["per_seed"] = {s: vals[t][s] for s in seeds}
    # the change after refitting, refitted lead minus carried lead (the negative of placement)
    vals["change"] = {s: vals["lead_after_refit"][s] - vals["lead_at_carried_rule"][s] for s in seeds}
    terms["change"] = inference.paired_contrast(vals["lead_after_refit"], vals["lead_at_carried_rule"], seeds=seeds)
    terms["change"]["per_seed"] = dict(vals["change"])
    spread["change"] = dict(spread["placement"])
    m_rule = terms["lead_at_carried_rule"]["mean"]
    return dict(direction=f"{names[0]} minus {names[1]} (positive: {names[0]} scores higher)", names=list(names),
                rule=rule, same_rule_per_seed={s: per_seed[s]["same_rule"] for s in seeds},
                periods=per_seed[seeds[0]]["periods"], carried_level=per_seed[seeds[0]]["U"],
                seeds=seeds, terms=terms,
                placement_share=(terms["placement"]["mean"] / m_rule if m_rule != 0 else None),
                training_run_uncertainty=("seed (training run): one value per seed, paired across methods; mean, "
                                          "marginal 95% t interval over seeds, unadjusted class"),
                record_split_variation=dict(
                    values=spread,
                    unit=("partition (record split) within one run: SD over the partitions of the per-partition "
                          "value, carried level on the held-out halves; not a standard error of record sampling "
                          "and not combined with the seed interval")),
                per_seed=per_seed)
