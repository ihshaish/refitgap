#!/usr/bin/env python3
"""Paired refit, DistilBERT on the PubMed stream (initialisations 0-7): sequential training, replay, extra updates
on current data and LwF, compared in pairs of arms that share one initialisation (a seed block). For arm a against
arm b, on the final model: the lead at each arm's inherited thresholds theta_1 (G_U), the lead after both refit one
threshold per label by exact split-half cross-fitting (G_V), and the change G_V - G_U. Trains nothing.

  python scripts/paired_refit_pubmed.py --scores-dir DIR [--lwf-alpha 0.02] [--seeds 0-7] [--workers 4]
      --out results/pubmed_paired_refit/pubmed_paired_refit.json > results/pubmed_paired_refit/pubmed_paired_refit.out
  two arms:   --files-a A_s0_scores.npz ... --files-b B_s0_scores.npz ... --names replay sequential [--out F]
  three arms: --files-a M1... --files-b M2... --files-c C... --names lwf replay sequential [--out F]
  cells:      python scripts/paired_refit_pubmed.py cells --dir DIR | --files F... --out-dir O

The first form follows plans/pubmed-paired-refit.txt: replay minus sequential training, LwF minus sequential
training and replay minus extra updates, each its own Holm family {G_U, G_V} at the 0.005 margin; the contrast D
for LwF minus replay; and the decomposition of backward transfer for the sequential arm. The LwF setting (0.02)
is the one in plans/pubmed-lwf-selection.txt. The cells subcommand writes macro-F1 at theta_1 and macro-AUPRC of
every (model, period) cell of each score file to CSV.

Needs the 32 PubMed score files pubmed_distilbert_{naive,replay,replaycur,lwfa0.02}_s<k>_scores.npz, k = 0..7
(not in the repository; pass their directory), refitgap (../src) and, for cells, scikit-learn.
Backs Figure 2 (PubMed rows), Table C.1 (PubMed rows) and Table C.6 (the other PubMed calculations: same folds,
cut of 0.5, 500 labels, other periods, first learning, D, decomposition). Stored output:
results/pubmed_paired_refit/.

Each pair is analysed on its complete seed blocks. When the score directory holds COLLECTION_CLOSED.txt or
<run>.done.json files (written by the runs), they are checked: the closure file must name a date and a reason, and
a score file enters only if its done file lists it with its current sha16. Without them every arm whose score file
is present is used, and the output says so. No computed value depends on these files.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:                                    # the cells subcommand needs only numpy and scikit-learn
    import refitgap as rg
    from refitgap import inference
    from refitgap.crossfit import array_hash, decompose, evaluate, partitions
    from refitgap.fitters import f1_vec, fit_exact
    from refitgap.runs import load_run
except ImportError:                     # pragma: no cover
    rg = None

SPLITS, MARGIN, RESAMPLES, BOOT_SEED = 200, 0.005, 20000, 20260920
# period ranges by 0-based period id; names are 1-indexed (period 1 is the first)
RANGES = {"periods 2-7": list(range(1, 7)), "periods 1-7": list(range(0, 7)), "period 8": [7]}
PRIMARY = "periods 2-7"
SEED_RE = re.compile(r"_s(\d+)_scores\.npz$")
LIMITED_N, LIMITED_BASE = 500, 9000     # refit on 500 labelled records per fitting half
CUT = 0.5
CLASS_MIN_PAIRS = 8                     # outcome classes need eight complete pairs
CLOSED_NAME = "COLLECTION_CLOSED.txt"
PUBMED_RE = re.compile(r"^pubmed_distilbert_(?!.*_carve).*_s(\d+)_scores\.npz$")
PUBMED_SEEDS = tuple(range(10))
DIR_LABEL = "SCORES_DIR"                # directories are not written into the output, only file names
ARM_FILES = {"sequential": "naive", "replay": "replay", "extra updates": "replaycur", "lwf": "lwfa{alpha:g}"}
PARTITION_SPEC = ("refitgap.crossfit.partitions: partition b (b = 0..199) of period id j (0-based) in the block of "
                  "initialisation s is the b-th of 200 successive permutations of range(n_j) drawn from numpy "
                  "default_rng(7000 + 100*s + j); half 0 is its first n_j // 2 positions, half 1 the rest; thresholds "
                  "are fitted on one half (exact per-label F1 maximiser, the paper's ties and labels without "
                  "positives) and scored on the other, both ways; values are averaged over the two directions and "
                  "the 200 partitions. The same partitions serve every arm of a block (checked by hash).")
RULE_MARKERS = ("this comparison's complete pairs: both arms have their scores file and a done marker listing it "
                "with its sha16")
RULE_FILES = "this comparison's complete pairs: both arms have their scores file (no done markers were found)"


def missing_files_message(where):
    return (f"No PubMed score files found ({where}). This analysis reads the 32 files "
            "pubmed_distilbert_<arm>_s<k>_scores.npz, with <arm> one of naive, replay, replaycur, lwfa0.02 "
            "(sequential training, replay, extra updates on current data, LwF at alpha 0.02) and k = 0 to 7. "
            "They are not in the repository; "
            "pass the directory that holds them with --scores-dir DIR (or name the files with --files-a and "
            "--files-b), and run from paper/.")


def seed_of(path):
    m = SEED_RE.search(Path(path).name)
    if not m:
        raise SystemExit(f"cannot read the seed from {path} (expected ..._s<k>_scores.npz)")
    return int(m.group(1))


def refuse_development_file(path):
    if "_carve" in Path(path).name:
        raise SystemExit(f"REFUSED: {path} is a development file (held-out records), not a score file of the "
                         "eight seed blocks")


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def has_markers(path):
    """True when the directory of this score file holds any <run>.done.json."""
    return any(Path(os.path.abspath(path)).parent.glob("*.done.json"))


def done_status(path, need_marker):
    """(usable?, reason). The scores file must exist; where the directory holds done markers, its own marker must
    list it with the sha16 it has now."""
    p = Path(path)
    if not p.is_file():
        return False, "scores file missing"
    if not need_marker:
        return True, "scores file present"
    marker = p.with_name(p.name[:-len("_scores.npz")] + ".done.json")
    if not marker.is_file():
        return False, "no done marker"
    try:
        outs = json.loads(marker.read_text()).get("outputs", {})
    except (json.JSONDecodeError, OSError):
        return False, "done marker unreadable"
    if p.name not in outs:
        return False, "done marker does not list the scores file"
    if sha16(p) != outs[p.name]:
        return False, "scores file differs from the sha16 in its done marker"
    return True, "complete"


def is_pubmed(path):
    m = PUBMED_RE.match(Path(path).name)
    return bool(m) and int(m.group(1)) in PUBMED_SEEDS


def closure(paths):
    """Where a directory of PubMed score files holds COLLECTION_CLOSED.txt, it must carry the lines
    'Collection closed: YYYY-MM-DD' and 'Reason: ...'; the closures found are recorded in the output. A directory
    without the file is noted and used."""
    out, seen, labels = {}, set(), {}
    for p in paths:
        if not is_pubmed(p):
            continue
        d = Path(os.path.abspath(p)).parent       # the directory named, not a link target
        if d in seen:
            continue
        seen.add(d)
        lab = DIR_LABEL if not labels else f"{DIR_LABEL}_{len(labels) + 1}"
        labels[d] = lab
        f = d / CLOSED_NAME
        if not f.is_file():
            print(f"  note: no {CLOSED_NAME} in {d}; the closure of the collection is not checked")
            continue
        t = f.read_text()
        m = re.search(r"^Collection closed: (\d{4}-\d{2}-\d{2})\s*$", t, re.M)
        r = re.search(r"^Reason: (.+)$", t, re.M)
        if not m or not r:
            raise SystemExit(f"REFUSED: {f} lacks the lines 'Collection closed: YYYY-MM-DD' and 'Reason: ...'")
        out[lab] = dict(file=CLOSED_NAME, date=m.group(1), reason=r.group(1).strip(),
                        sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    return out


def ids_hash(z, j):
    """Hash of the stream rows of period j's test records, when the file carries them."""
    k = f"ids_test_{j}"
    return hashlib.sha256(np.asarray(z[k], dtype=np.int64).tobytes()).hexdigest()[:16] if k in z.files else None


# ----------------------------------------------------------------- one arm ---
def one_arm(task):
    """(file, seed, period ids[, backward-transfer flag]) -> the arm's refitgap Evaluation and the per-label arrays
    behind the descriptive outputs; with the flag also the Evaluation of the models that first learned each period
    (P_j_j), the reference of the backward-transfer decomposition.

    The Evaluation is refitgap.crossfit.evaluate with two rules, the arm's theta_1 ("inherited") and 0.5 on every
    label ("cut0.5", 0 if the scores are not probabilities); the refitted levels and the partitions do not depend
    on the rule. U is macro-F1 at theta_1 on the whole period's test records; V is macro-F1 after refit, averaged
    over the 200 partitions and both directions."""
    path, s, pids = task[:3]
    bt = task[3] if len(task) > 3 else False
    r = load_run(path, periods=pids)
    if r["N"] != 8:
        raise ValueError(f"{path}: expected 8 periods, found {r['N']}")
    Ps = [np.asarray(P) for P in r["Ps"]]
    Ys = [np.asarray(Y) for Y in r["Ys"]]
    theta = r["thresholds"]
    lo, hi = min(float(P.min()) for P in Ps), max(float(P.max()) for P in Ps)
    prob = lo >= 0.0 and hi <= 1.0
    cut = np.full(len(theta), CUT if prob else 0.0)          # 0.5 on the probability scale = logit 0
    ev = evaluate(Ps, Ys, s, pids, rules={"inherited": theta, "cut0.5": cut}, splits=SPLITS)
    N, K = len(pids), len(theta)
    F_sep, F_uh, F_lim = (np.full((SPLITS, 2, N, K), np.nan) for _ in range(3))
    U_lab, Ucut_lab = np.empty((N, K)), np.empty((N, K))
    part_hash, sub_hash, short = [], [], []
    for c, j in enumerate(pids):
        P, Y = Ps[c].astype(np.float64), Ys[c].astype(int)
        perms = partitions(len(Y), s, j, SPLITS)
        part_hash.append(array_hash(np.stack(perms), np.int64))
        # limited labels: per partition and direction, 500 positions drawn within the fitting half; the draw
        # depends on (seed, period id) only, so every arm of a block gets the same subsample
        rng = np.random.default_rng(LIMITED_BASE + 100 * s + j)
        h = len(Y) // 2
        subs = []
        for b in range(SPLITS):
            A, B = perms[b][:h], perms[b][h:]
            for d in (0, 1):
                fit, sc = (A, B) if d == 0 else (B, A)
                Pe, Ye = P[sc], Y[sc]
                F_sep[b, d, c] = f1_vec(Pe, Ye, fit_exact(P[fit], Y[fit]))
                F_uh[b, d, c] = f1_vec(Pe, Ye, theta)
                if len(fit) >= LIMITED_N:
                    pos = rng.choice(len(fit), LIMITED_N, replace=False)
                    sub = fit[pos]
                else:                                        # a short half is used whole and flagged
                    sub = fit
                    short.append(dict(period_id=j, partition=b, direction=d, fitting_records=int(len(fit))))
                subs.append(pos if len(fit) >= LIMITED_N else np.arange(len(fit)))
                F_lim[b, d, c] = f1_vec(Pe, Ye, fit_exact(P[sub], Y[sub]))
        sub_hash.append(array_hash(np.concatenate(subs), np.int64))
        U_lab[c] = f1_vec(P, Y, theta)
        Ucut_lab[c] = f1_vec(P, Y, cut)
    # the per-label arrays must reproduce refitgap's macro levels
    dev = max(float(np.max(np.abs(F_sep.mean(-1) - ev.sep))), float(np.max(np.abs(F_uh.mean(-1) - ev.U_half["inherited"]))),
              float(np.max(np.abs(U_lab.mean(-1) - ev.U_full["inherited"]))),
              float(np.max(np.abs(Ucut_lab.mean(-1) - ev.U_full["cut0.5"]))))
    z = np.load(path)
    diag = np.array([f1_vec(z[f"P_{j}_{j}"].astype(np.float64), z[f"Y_{j}"].astype(int), theta).mean() for j in range(8)])
    final = np.array([f1_vec(z[f"P_7_{j}"].astype(np.float64), z[f"Y_{j}"].astype(int), theta).mean() for j in range(8)])
    labels = [str(x) for x in z["labels"]] if "labels" in z.files else [f"label{k}" for k in range(K)]
    rec_hash = [ids_hash(z, j) for j in pids]
    ev_ref = None
    if bt:
        rd = load_run(path, model="diagonal", periods=pids)
        ev_ref = evaluate([np.asarray(P) for P in rd["Ps"]], [np.asarray(Y) for Y in rd["Ys"]], s, pids,
                          rules={"inherited": theta}, splits=SPLITS, single_model=False)
    support = [dict(records=int(len(Y)), positives=[int(v) for v in Y.sum(0)],
                    negatives=[int(v) for v in (len(Y) - Y.sum(0))]) for Y in Ys]
    return dict(seed=s, path=str(path), ev=ev, F_sep=F_sep, F_uh=F_uh, F_lim=F_lim, U_lab=U_lab, Ucut_lab=Ucut_lab,
                diag=diag, final=final, labels=labels, support=support, theta=theta, score_scale=(
                    "probability" if prob else "other (cut at 0)"), score_range=[lo, hi], partition_hash=part_hash,
                subsample_hash=sub_hash, short_halves=short, label_hash=list(ev.label_hash),
                per_label_reproduction_dev=dev, records_per_period=[int(len(y)) for y in Ys], record_id_hash=rec_hash,
                ev_ref=ev_ref)


# ----------------------------------------------------------- the pair's family ---
def bootstrap(cols):
    """Percentile bootstrap over pairs; one resample matrix for every column."""
    v = np.asarray(cols, float)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(v), (RESAMPLES, len(v)))
    b = v[idx].mean(axis=1)
    return [[float(np.percentile(b[:, c], 2.5)), float(np.percentile(b[:, c], 97.5))] for c in range(v.shape[1])]


def summarise(evals_a, evals_b, pids, names, min_pairs=CLASS_MIN_PAIRS):
    """G_U, G_V and the change on one period range (refitgap.method_contrast, each arm at its own theta_1). G_U and
    G_V form the Holm family (refitgap.inference.family: equivalent, material, small non-zero or unresolved, with
    the sign-flip condition), classed only with at least min_pairs pairs; the change is never classed."""
    mc = rg.method_contrast(evals_a, evals_b, "inherited", periods=pids, same_rule=False, names=names)
    seeds = mc["seeds"]
    gu = np.array([mc["terms"]["lead_at_carried_rule"]["per_seed"][s] for s in seeds])
    gv = np.array([mc["terms"]["lead_after_refit"]["per_seed"][s] for s in seeds])
    ch = gv - gu
    classed = len(seeds) >= min_pairs
    fam = inference.family({"G_U": gu, "G_V": gv}, margin=MARGIN)
    fam["change"] = inference.paired_stats(ch, MARGIN, name="change")
    ci = bootstrap(np.column_stack([gu, gv, ch]))
    lv = {k: {s: mc["per_seed"][s][k] for s in seeds} for k in ("U_ref", "U_final", "V_ref", "V_final")}
    out = dict(period_ids=list(pids), periods_1_indexed=[j + 1 for j in pids], seeds=seeds, n_pairs=len(seeds),
               classed=classed, class_rule=f"classes only with at least {min_pairs} complete pairs",
               levels={f"U_{names[0]}": lv["U_ref"], f"U_{names[1]}": lv["U_final"],
                       f"V_{names[0]}": lv["V_ref"], f"V_{names[1]}": lv["V_final"]})
    for k, x, c in (("G_U", gu, ci[0]), ("G_V", gv, ci[1]), ("change", ch, ci[2])):
        f = fam[k]
        row = dict(per_pair=x.tolist(), mean=float(x.mean()), t95=f["ci95"], bootstrap95=c,
                   positive_pairs=int((x > 0).sum()), negative_pairs=int((x < 0).sum()), zero_pairs=int((x == 0).sum()))
        if k != "change" and classed:
            row.update(p_t=f["p_t"], p_tost=f["p_tost"], p_t_holm=f["p_holm"], p_tost_holm=f["p_tost_holm"],
                       holm_family_size=f["holm_family_size"], p_signflip=f["p_signflip"], class_word=f["class"],
                       class_before_signflip=f["class_before_signflip"], status="Holm family {G_U, G_V}")
        elif k != "change":
            row.update(class_word=None, status=f"descriptive: {len(seeds)} complete pairs, fewer than {min_pairs}; "
                                               "no outcome class")
        else:
            row.update(class_word=None, status="not classed: mean, 95 % paired t interval, blocks by sign")
        out[k] = row
    # refitgap's own per-term t intervals, which must equal the family's ci95
    out["method_contrast_t95"] = {k: mc["terms"][t]["ci95"] for k, t in
                                  (("G_U", "lead_at_carried_rule"), ("G_V", "lead_after_refit"), ("change", "change"))}
    return out


def check(res, expected=None, key=None, stored=None, negate=False):
    """Largest absolute difference from stored values on the primary range: expected is a file in the Figure 2
    data layout (entry key, terms inherited/refitted/change), stored a paired_refit.json with G_U, G_V and Delta
    (Delta = -change). negate: the stored contrast is B minus A."""
    r, worst, lines = res["ranges"][PRIMARY], 0.0, []
    sg = -1.0 if negate else 1.0
    if expected is not None:
        e = json.loads(Path(expected).read_text())[key]
        for ours, theirs in (("G_U", "inherited"), ("G_V", "refitted"), ("change", "change")):
            t95 = sorted(sg * x for x in e[theirs]["ci"])
            d = max([abs(r[ours]["mean"] - sg * e[theirs]["mean"])] +
                    [abs(x - sg * y) for x, y in zip(r[ours]["per_pair"], e[theirs]["per_pair"])] +
                    [abs(x - y) for x, y in zip(r[ours]["t95"], t95)])
            worst = max(worst, d)
            lines.append(f"  {ours:6s} ours {r[ours]['mean']:+.6f}  expected.json {sg * e[theirs]['mean']:+.6f}  "
                         f"largest difference (per pair, mean, t95) {d:.1e}")
    if stored is not None:
        st = json.loads(Path(stored).read_text())
        for ours in ("G_U", "G_V"):
            s = st[ours]
            nums = [(r[ours]["mean"], s["mean"])] + list(zip(r[ours]["per_pair"], s["per_pair"])) + \
                list(zip(r[ours]["t95"], s["t95"])) + list(zip(r[ours]["bootstrap95"], s["bootstrap95"])) + \
                [(r[ours]["p_t_holm"], s["p_t_holm"]), (r[ours]["p_tost_holm"], s["p_tost_holm"]),
                 (r[ours]["p_signflip"], s["p_signflip"])]
            d = max(abs(x - y) for x, y in nums)
            worst = max(worst, d)
            same = r[ours]["class_word"] == s["class_word"] and r[ours]["positive_pairs"] == s["positive_pairs"]
            if not same:
                worst = float("inf")
            lines.append(f"  {ours:6s} stored paired_refit.json: class {s['class_word']} / ours {r[ours]['class_word']}"
                         f"; positive pairs {s['positive_pairs']} / {r[ours]['positive_pairs']}; largest difference "
                         f"(per pair, mean, t95, bootstrap, Holm p, sign-flip p) {d:.1e}")
        d = max(abs(-x - y) for x, y in zip(r["change"]["per_pair"], st["Delta"]["per_pair"]))
        worst = max(worst, d)
        lines.append(f"  change = -Delta of paired_refit.json, per pair: largest difference {d:.1e}")
    return worst, lines


# ------------------------------------------------------ descriptive outputs ---
def describe(x):
    """Paired values -> mean, two-sided 95 % paired t interval, counts by sign, n. No class."""
    x = np.asarray(x, float)
    st = inference.paired_stats(x, MARGIN)
    return dict(per_pair=x.tolist(), mean=float(x.mean()), t95=st.get("ci95"), positive_pairs=int((x > 0).sum()),
                negative_pairs=int((x < 0).sum()), n=int(len(x)))


def _agg(X, cols):
    """(splits, 2, periods) -> per partition the mean over the range's periods of the two directions' mean, then
    the mean over partitions (the uniform weighting of refitgap's summaries)."""
    return float(np.mean(np.mean(0.5 * (X[:, 0, cols] + X[:, 1, cols]), axis=1)))


def additional(A, B, seeds, pids, names, primary):
    """For one range of one pair: the inherited lead on the same folds as the refit (refitgap U_half), the lead at
    a common cut of 0.5, and the refit on 500 labelled records per fitting half. primary: that range's summarise()."""
    ev_a, ev_b = {s: A[s]["ev"] for s in seeds}, {s: B[s]["ev"] for s in seeds}
    gu = np.array(primary["G_U"]["per_pair"])
    gv = np.array(primary["G_V"]["per_pair"])
    mh = rg.method_contrast(ev_a, ev_b, "inherited", periods=pids, same_rule=False, halves=True, names=names)
    gu_h = np.array([mh["terms"]["lead_at_carried_rule"]["per_seed"][s] for s in seeds])
    gv_h = np.array([mh["terms"]["lead_after_refit"]["per_seed"][s] for s in seeds])
    mc = rg.method_contrast(ev_a, ev_b, "cut0.5", periods=pids, same_rule=True, names=names)
    gu_c = np.array([mc["terms"]["lead_at_carried_rule"]["per_seed"][s] for s in seeds])
    gv_c = np.array([mc["terms"]["lead_after_refit"]["per_seed"][s] for s in seeds])
    cols = [A[seeds[0]]["ev"].period_ids.index(p) for p in pids]
    v5 = {k: np.array([_agg(X[s]["F_lim"].mean(-1), cols) for s in seeds]) for k, X in (("a", A), ("b", B))}
    gv5 = v5["a"] - v5["b"]
    return dict(
        same_folds=dict(
            convention="U at theta_1 on the scoring halves of the refit's partitions (refitgap U_half)",
            G_U_same_folds=describe(gu_h), change_same_folds=describe(gv - gu_h),
            G_U_whole_period=describe(gu), change_whole_period=describe(gv - gu),
            G_U_same_minus_whole=describe(gu_h - gu),
            largest_abs_shift_per_pair=float(np.max(np.abs(gu_h - gu))),
            G_V_identical=bool(np.array_equal(gv_h, gv))),
        fixed_cut=dict(
            cut=CUT, score_scale=sorted({A[s]["score_scale"] for s in seeds} | {B[s]["score_scale"] for s in seeds}),
            G_cut=describe(gu_c), G_V=describe(gv), change_from_cut=describe(gv - gu_c),
            G_cut_minus_G_U=describe(gu_c - gu), G_V_identical=bool(np.array_equal(gv_c, gv))),
        limited_labels=dict(
            records_per_fitting_half=LIMITED_N,
            draw=f"default_rng({LIMITED_BASE} + 100*seed + period id); per partition and direction "
                 f"rng.choice(len(fitting half), {LIMITED_N}, replace=False); scored on the whole other half",
            **{f"V_{names[0]}": dict(zip(map(str, seeds), v5["a"].tolist())),
               f"V_{names[1]}": dict(zip(map(str, seeds), v5["b"].tolist()))},
            G_V_limited=describe(gv5), change_limited=describe(gv5 - gu), G_V_limited_minus_G_V=describe(gv5 - gv),
            short_halves=sorted({(x["period_id"], x["fitting_records"]) for s in seeds for X in (A, B)
                                 for x in X[s]["short_halves"] if x["period_id"] in pids})))


def per_label(A, B, seeds, pids):
    """Support of every label and its part in the macro-average G_U, G_V and change (the parts sum to the macro
    values)."""
    r0 = A[seeds[0]]
    cols = [r0["ev"].period_ids.index(p) for p in pids]
    K = len(r0["labels"])
    pu = np.array([(A[s]["U_lab"][cols] - B[s]["U_lab"][cols]).mean(0) / K for s in seeds])        # (seeds, K)

    def vlab(F):
        return np.mean(np.mean(0.5 * (F[:, 0, cols, :] + F[:, 1, cols, :]), axis=1), axis=0)       # (K,)
    pv = np.array([(vlab(A[s]["F_sep"]) - vlab(B[s]["F_sep"])) / K for s in seeds])
    rows = []
    for k, lab in enumerate(r0["labels"]):
        rows.append(dict(label=lab,
                         positives_per_period=[r0["support"][c]["positives"][k] for c in cols],
                         negatives_per_period=[r0["support"][c]["negatives"][k] for c in cols],
                         part_G_U=float(pu[:, k].mean()), part_G_V=float(pv[:, k].mean()),
                         part_change=float((pv[:, k] - pu[:, k]).mean())))
    return dict(period_ids=list(pids), n_labels=K, records_per_period=[r0["support"][c]["records"] for c in cols],
                note="part = mean over pairs of the mean over periods of (F1 of A - F1 of B) on that label, / K",
                labels=rows, sum_part_G_U=float(pu.sum(1).mean()), sum_part_G_V=float(pv.sum(1).mean()))


def arm_levels(X, seeds):
    """One arm's mean over seeds of macro-F1 at theta_1, from the model that first learned each period (P_j_j)
    and from the final model (P_7_j)."""
    d = np.array([X[s]["diag"] for s in seeds])
    f = np.array([X[s]["final"] for s in seeds])
    return dict(current_period_8_final=float(f[:, 7].mean()),
                first_learned=d.mean(0).tolist(), final=f.mean(0).tolist(), final_minus_first=(f - d).mean(0).tolist(),
                periods_2_7=dict(first_learned=float(d[:, 1:7].mean()), final=float(f[:, 1:7].mean()),
                                 final_minus_first=float((f - d)[:, 1:7].mean())),
                note="descriptive; macro-F1 at the arm's theta_1 on each period's test records; index 0 = period 1")


def pair_levels(A, B, seeds):
    da = np.array([A[s]["diag"] - B[s]["diag"] for s in seeds])
    fa = np.array([A[s]["final"] - B[s]["final"] for s in seeds])
    return dict(lead_first_learned=da.mean(0).tolist(), lead_final=fa.mean(0).tolist(),
                periods_2_7=dict(lead_first_learned=describe(da[:, 1:7].mean(1)), lead_final=describe(fa[:, 1:7].mean(1))),
                note="descriptive; lead at theta_1 of each arm, per period (index 0 = period 1): at first learning "
                     "(P_j_j) and from the final models (P_7_j)")


def partition_record(X, seeds):
    """The partition specification and, per block, the partition hashes (and record-id hashes when stored)."""
    return dict(specification=PARTITION_SPEC, splits=SPLITS, split_base=7000,
                refitgap_version=rg.__version__, numpy=np.__version__,
                per_block={str(s): dict(partition_hash=X[s]["partition_hash"], record_id_hash=X[s]["record_id_hash"])
                           for s in seeds})


def same_fold_D(pair):
    """D per block on the common scoring halves, i.e. the pair's change with U on the same folds:
    (V_a - V_b) - (U_sf,a - U_sf,b); the whole-period variant beside."""
    out = {}
    for k in RANGES:
        sf = pair["additional"][k]["same_folds"]
        out[k] = dict(D=sf["change_same_folds"], D_whole_period=sf["change_whole_period"],
                      blocks=len(pair["seeds"]), seeds=pair["seeds"])
    return out


def analyse_pair(A, B, seeds, names, files_a, files_b, min_pairs=CLASS_MIN_PAIRS):
    ev_a = {s: A[s]["ev"] for s in seeds}
    ev_b = {s: B[s]["ev"] for s in seeds}
    out = dict(contrast=f"{names[0]} minus {names[1]}", files_a=files_a, files_b=files_b, seeds=seeds,
               n_pairs=len(seeds), splits=SPLITS,
               margin=MARGIN, resamples=RESAMPLES, bootstrap_seed=BOOT_SEED,
               partition_rule="default_rng(7000 + 100*seed + period id), period ids 0-based",
               refitgap_version=rg.__version__, numpy=np.__version__,
               labels_with_different_theta1={s: int(np.sum(A[s]["theta"] != B[s]["theta"])) for s in seeds},
               records_per_period=A[seeds[0]]["records_per_period"], primary_range=PRIMARY,
               ranges={k: summarise(ev_a, ev_b, v, names, min_pairs) for k, v in RANGES.items()})
    out["additional"] = {k: additional(A, B, seeds, v, names, out["ranges"][k]) for k, v in RANGES.items()}
    out["secondary_D_same_folds"] = same_fold_D(out)
    out["per_label"] = per_label(A, B, seeds, RANGES[PRIMARY])
    out["levels_descriptive"] = {names[0]: arm_levels(A, seeds), names[1]: arm_levels(B, seeds),
                                 "pair": pair_levels(A, B, seeds)}
    out["partitions"] = partition_record(A, seeds)
    return out


def backward_transfer(X, seeds, name):
    """Decomposition of one arm's backward transfer (refitgap.crossfit.decompose): reference = the model that
    first learned each period (P_j_j), final = the last model (P_7_j); B = U_ref - U_final = A + dR at theta_1,
    U on the whole period; per block and over blocks, descriptive."""
    out = {}
    for k, pids in RANGES.items():
        rows = {s: decompose(X[s]["ev_ref"], X[s]["ev"], "inherited", periods=pids) for s in seeds}
        per = {q: np.array([rows[s][q] for s in seeds]) for q in ("B", "A", "dR", "U_ref", "U_final", "V_ref", "V_final")}
        out[k] = dict({q: describe(v) for q, v in per.items()}, seeds=seeds,
                      largest_residual=float(max(rows[s]["residual_D_A_R"] for s in seeds)))
    return dict(arm=name, reference="P_j_j (the model that first learned period j)", final="P_7_j (the last model)",
                identity="B = A + dR (refitgap.crossfit.decompose, rule theta_1, U on the whole period)",
                status="descriptive", ranges=out)


def shared_partitions(arms, seeds):
    """Every arm of a block is evaluated with one seed, one set of period ids, one number of records and the same
    labels, so its partitions and limited-label subsamples are the same arrays; compared by hash, as are the
    record ids of every period when the files carry them."""
    rows, ok = {}, True
    for s in seeds:
        sig = [(X[s]["ev"].seed, tuple(X[s]["ev"].period_ids), X[s]["ev"].splits, X[s]["ev"].split_base,
                tuple(X[s]["label_hash"]), tuple(X[s]["partition_hash"]), tuple(X[s]["subsample_hash"]),
                tuple(X[s]["record_id_hash"])) for X in arms]
        same = all(x == sig[0] for x in sig)
        ok &= same
        rows[s] = dict(identical=same, partition_hash=list(sig[0][5]), subsample_hash=list(sig[0][6]),
                       record_id_hash=list(sig[0][7]))
    return ok, rows


def secondary_D(c1, c2, direct, names):
    """Three arms: D = (V_m1 - V_m2) - (U_sf,m1 - U_sf,m2) per block from the direct m1-minus-m2 pair, with the
    whole-period variant; where the three arms share blocks, also the largest difference from Delta_m1 - Delta_m2
    (each the change against the comparator), which should vanish because the comparator's levels cancel."""
    out = {}
    dd = direct["secondary_D_same_folds"]
    for k in RANGES:
        r = dict(dd[k], status="secondary contrast, not classed; positive: the refit moves the lead towards "
                               f"{names[0]} relative to {names[1]}")
        common = [s for s in direct["seeds"] if s in c1["seeds"] and s in c2["seeds"]]
        if common:
            def by_seed(c, key):
                vals = c["additional"][k]["same_folds"][key]["per_pair"]
                return dict(zip(c["seeds"], vals))
            d1, d2 = by_seed(c1, "change_same_folds"), by_seed(c2, "change_same_folds")
            dD = dict(zip(direct["seeds"], dd[k]["D"]["per_pair"]))
            r["largest_abs_D_minus_Delta_difference_same_folds"] = float(max(abs(dD[s] - (d1[s] - d2[s])) for s in common))
            w1, w2 = by_seed(c1, "change_whole_period"), by_seed(c2, "change_whole_period")
            dW = dict(zip(direct["seeds"], dd[k]["D_whole_period"]["per_pair"]))
            r["largest_abs_D_whole_minus_Delta_difference"] = float(max(abs(dW[s] - (w1[s] - w2[s])) for s in common))
        out[k] = r
    return out


# ------------------------------------------------------------------ output ---
def fmt_ci(d, nd=5):
    if d.get("t95") is None:
        return f"{d['mean']:+.{nd}f} (no interval: one pair)"
    return f"{d['mean']:+.{nd}f} [{d['t95'][0]:+.{nd}f}, {d['t95'][1]:+.{nd}f}]"


def print_pair(out):
    n = out["n_pairs"]
    print(f"\n=== {out['contrast']}, initialisations {out['seeds']} ({n} complete pairs); labels with different "
          f"theta_1 per block {list(out['labels_with_different_theta1'].values())}")
    for k, r in out["ranges"].items():
        print(f"\n{k}{' (primary)' if k == PRIMARY else ' (descriptive)'}"
              + ("" if r["classed"] or k != PRIMARY else f"  DESCRIPTIVE: {n} pairs, fewer than eight; no outcome class"))
        for q in ("G_U", "G_V", "change"):
            x = r[q]
            line = (f"  {q:6s} {x['mean']:+.4f}  t95 [{x['t95'][0]:+.4f}, {x['t95'][1]:+.4f}]  boot95 "
                    f"[{x['bootstrap95'][0]:+.4f}, {x['bootstrap95'][1]:+.4f}]  blocks +{x['positive_pairs']} "
                    f"-{x['negative_pairs']} 0:{x['zero_pairs']} of {n}")
            if q != "change" and r["classed"]:
                line += (f"  p_t(Holm) {x['p_t_holm']:.2g}  TOST(Holm) {x['p_tost_holm']:.2g}  sign-flip "
                         f"{x['p_signflip']:.3g}  {x['class_word']}")
            elif q == "change":
                line += "  [not classed]"
            else:
                line += "  [descriptive]"
            print(line)
            print("         per pair " + " ".join(f"{v:+.4f}" for v in x["per_pair"]))
        ad = out["additional"][k]
        print(f"  beside (not classed): G_U same folds {fmt_ci(ad['same_folds']['G_U_same_folds'])}; change same folds "
              f"{fmt_ci(ad['same_folds']['change_same_folds'])}; largest |shift| "
              f"{ad['same_folds']['largest_abs_shift_per_pair']:.1e}")
        print(f"                        G at cut 0.5 {fmt_ci(ad['fixed_cut']['G_cut'])}; change from cut "
              f"{fmt_ci(ad['fixed_cut']['change_from_cut'])}")
        print(f"                        G_V with {LIMITED_N} labels {fmt_ci(ad['limited_labels']['G_V_limited'])}; change "
              f"{fmt_ci(ad['limited_labels']['change_limited'])}; short halves {ad['limited_labels']['short_halves']}")


def print_D(sd, names):
    print(f"\nsecondary contrast D = (V_{names[0]} - V_{names[1]}) - (U_sf,{names[0]} - U_sf,{names[1]}), on the common "
          "scoring halves, per block; not classed")
    for k, r in sd.items():
        x = r["D"]
        print(f"  {k:12s} D {fmt_ci(x)}  blocks {r['blocks']} {r['seeds']}  per block "
              + " ".join(f"{v:+.5f}" for v in x["per_pair"]))
        print(f"  {'':12s} whole-period variant (U on the whole period) {fmt_ci(r['D_whole_period'])}"
              + (f"; |D - (Delta_m1 - Delta_m2)| <= {r['largest_abs_D_minus_Delta_difference_same_folds']:.1e}"
                 if "largest_abs_D_minus_Delta_difference_same_folds" in r else ""))


def print_bt(bt):
    print(f"\ndecomposition of backward transfer, {bt['arm']} (reference P_j_j, final P_7_j; B = A + dR), descriptive")
    for k, r in bt["ranges"].items():
        print(f"  {k:12s} B {fmt_ci(r['B'])}  A {fmt_ci(r['A'])}  dR {fmt_ci(r['dR'])}  blocks {len(r['seeds'])}")


# ------------------------------------------------------------- evaluation ---
def evaluate_sides(maps, names, workers, bt_names=()):
    """Evaluate every usable (arm, seed) file once. -> (arms {name: {seed: result}}, status, largest per-label
    reproduction difference, rule text for the complete pairs)."""
    all_seeds = sorted(set().union(*maps))
    files = [f for m in maps for f in m.values()]
    need = {f: has_markers(f) for f in files}
    for d in sorted({str(Path(os.path.abspath(f)).parent) for f in files if not need[f]}):
        print(f"  note: no done markers in {d}; every arm whose score file is there is used")
    status = {s: {nm: (done_status(m[s], need[m[s]]) if s in m else (False, "no file given"))
                  for nm, m in zip(names, maps)} for s in all_seeds}
    pids = sorted({j for v in RANGES.values() for j in v})
    tasks, keys = [], []
    for nm, m in zip(names, maps):
        for s in sorted(m):
            if status[s][nm][0]:
                tasks.append((m[s], s, pids, nm in bt_names))
                keys.append((nm, s))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(one_arm, tasks))
    arms = {nm: {} for nm in names}
    for (nm, s), r in zip(keys, res):
        arms[nm][s] = r
    dev = max([r["per_label_reproduction_dev"] for r in res] or [0.0])
    if dev > 1e-12:
        raise SystemExit(f"per-label arrays do not reproduce refitgap's levels (largest difference {dev:.2e})")
    rule = RULE_MARKERS if all(need.values()) else RULE_FILES
    return arms, status, dev, rule


def complete_pairs(arms, a, b, status):
    seeds = sorted(set(arms[a]) & set(arms[b]))
    missing = {s: {nm: status[s][nm][1] for nm in (a, b) if not status[s][nm][0]} for s in status if s not in seeds}
    return seeds, missing


def pair_or_note(arms, a, b, names, maps_by_name, status, min_pairs, rule, extra_exclude=None):
    """One comparison on its own complete pairs; a missing arm outside it does not affect it."""
    seeds, missing = complete_pairs(arms, a, b, status)
    for s, why in (extra_exclude or {}).items():
        if s in seeds:
            seeds.remove(s)
            missing[s] = why
    if len(seeds) < 2:
        return None, dict(contrast=f"{names[0]} minus {names[1]}", seeds=seeds, incomplete=missing,
                          note="fewer than two complete pairs: no estimate with an interval")
    ok, rows = shared_partitions([arms[a], arms[b]], seeds)
    if not ok:
        raise SystemExit(f"{a} and {b}: the arms of a block do not share partitions, subsamples, labels or record ids")
    out = analyse_pair(arms[a], arms[b], seeds, names, [Path(maps_by_name[a][s]).name for s in seeds],
                       [Path(maps_by_name[b][s]).name for s in seeds], min_pairs)
    out["blocks"] = dict(complete=seeds, incomplete=missing, partitions_shared=ok, per_block=rows, rule=rule)
    return out, None


def steps_mismatch(maps_by_name, a, b):
    """Blocks whose replay and extra-updates step counts differ in some period; left out of that comparison."""
    out = {}
    for s in sorted(set(maps_by_name[a]) & set(maps_by_name[b])):
        try:
            sa = np.load(maps_by_name[a][s])["steps"]
            sb = np.load(maps_by_name[b][s])["steps"]
        except (OSError, KeyError):
            continue
        if not np.array_equal(sa, sb):
            out[s] = f"step counts differ: {sa.tolist()} against {sb.tolist()}"
    return out


# ------------------------------------------------------------------ cells ---
def cmd_cells(argv):
    """Macro-F1 at theta_1 and macro-AUPRC of every (model i, period j) cell of each score file: sklearn f1_score
    per label with zero_division 0, average_precision_score, 0 for a label without positives."""
    ap = argparse.ArgumentParser(prog="paired_refit_pubmed.py cells")
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    files = [Path(f) for f in (a.files or [])] + (sorted(Path(a.dir).glob("*_scores.npz")) if a.dir else [])
    files = [f for f in files if "_carve" not in f.name and f.is_file()]
    if not files:
        raise SystemExit(missing_files_message(a.dir or "none of the files named exists"))
    closed = closure(files)
    from sklearn.metrics import average_precision_score, f1_score
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(closure=closed, files={})
    for f in files:
        ok, why = done_status(f, has_markers(f))
        if not ok:
            summary["files"][f.name] = dict(skipped=why)
            print(f"  {f.name}: skipped ({why})")
            continue
        z = np.load(f)
        N = 1 + max(int(k.split("_")[1]) for k in z.files if k.startswith("P_"))
        theta = z["thresholds"]
        rid = f.name[:-len("_scores.npz")]
        rows = ["run_id,i,j,n_records,macro_f1,macro_auprc"]
        for i in range(N):
            for j in range(N):
                P, Y = z[f"P_{i}_{j}"].astype(np.float32), z[f"Y_{j}"].astype(int)
                f1 = f1_score(Y, (P >= theta).astype(int), average=None, zero_division=0)
                pr = np.array([average_precision_score(Y[:, k], P[:, k]) if Y[:, k].any() else 0.0
                               for k in range(Y.shape[1])])
                rows.append(f"{rid},{i},{j},{len(Y)},{float(f1.mean())!r},{float(pr.mean())!r}")
        cf = out / f"{rid}_cells.csv"
        cf.write_text("\n".join(rows) + "\n")
        summary["files"][f.name] = dict(cells=cf.name, n=N * N, sha16=sha16(f))
        print(f"  {f.name}: {N * N} cells -> {cf}")
    (out / "CELLS_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    return 0


# ------------------------------------------------------------------ main ---
def seed_range(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out += list(range(lo, hi + 1))
        elif part.strip():
            out.append(int(part))
    return out


def plan_mode(a):
    """The three comparisons of the plan, D for LwF minus replay and the sequential arm's decomposition, on the
    score files of one directory."""
    d = Path(a.scores_dir)
    seeds = seed_range(a.seeds)
    arm_files = {nm: tag.format(alpha=float(a.lwf_alpha)) for nm, tag in ARM_FILES.items()}
    maps_by_name = {nm: {s: str(d / f"{a.stream}_distilbert_{tag}_s{s}_scores.npz") for s in seeds}
                    for nm, tag in arm_files.items()}
    present = [f for m in maps_by_name.values() for f in m.values() if Path(f).is_file()]
    if not present:
        raise SystemExit(missing_files_message(f"looked in {d}"))
    closed = closure([f for m in maps_by_name.values() for f in m.values()])
    names = list(maps_by_name)
    arms, status, dev, rule = evaluate_sides([maps_by_name[n] for n in names], names, a.workers,
                                             bt_names=("sequential",))
    bad_steps = steps_mismatch(maps_by_name, "replay", "extra updates")
    res = dict(mode="plan", block_dir=DIR_LABEL, stream=a.stream, lwf_alpha=float(a.lwf_alpha), seeds_planned=seeds,
               closure=closed, class_min_pairs=a.class_min_pairs, comparisons={}, notes={},
               complete_files={nm: sorted(arms[nm]) for nm in names}, per_label_reproduction_largest_difference=dev,
               replay_extra_updates_step_mismatch=bad_steps, partition_specification=PARTITION_SPEC)
    for x, y, extra in (("replay", "sequential", None), ("lwf", "sequential", None),
                        ("replay", "extra updates", bad_steps)):
        out, note = pair_or_note(arms, x, y, (x, y), maps_by_name, status, a.class_min_pairs, rule, extra)
        key = f"{x} minus {y}"
        if out is None:
            res["notes"][key] = note
            print(f"\n=== {key}: {note['note']} (complete {note['seeds']})")
            continue
        res["comparisons"][key] = out
        print_pair(out)
    dpair, note = pair_or_note(arms, "lwf", "replay", ("lwf", "replay"), maps_by_name, status, a.class_min_pairs, rule)
    if dpair is not None:
        res["secondary_D"] = same_fold_D(dpair)
        res["secondary_D_definition"] = ("D = (V_LwF - V_replay) - (U_sf,LwF - U_sf,replay) per block, on the common "
                                         "scoring halves; whole-period variant beside; not classed")
        res["secondary_D_pair_blocks"] = dpair["blocks"]
        print_D(res["secondary_D"], ("lwf", "replay"))
    else:
        res["notes"]["D"] = note
    seq = sorted(arms["sequential"])
    if len(seq) >= 2:
        res["backward_transfer_sequential"] = backward_transfer(arms["sequential"], seq, "sequential")
        print_bt(res["backward_transfer_sequential"])
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["cells"]:
        return cmd_cells(argv[1:])
    if rg is None:
        raise SystemExit("refitgap is not importable: install it, or set PYTHONPATH to the repository's src/")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores-dir", default=None, help="the plan's comparisons on the PubMed score files in DIR")
    ap.add_argument("--lwf-alpha", default="0.02", help="LwF setting in the file names (default 0.02)")
    ap.add_argument("--stream", default="pubmed")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--files-a", nargs="+", default=None)
    ap.add_argument("--files-b", nargs="+", default=None)
    ap.add_argument("--files-c", nargs="+", default=None, help="the common comparator (three arms)")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--class-min-pairs", type=int, default=CLASS_MIN_PAIRS)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expected", help="optional check (two arms): a file in the Figure 2 data layout")
    ap.add_argument("--expected-key")
    ap.add_argument("--stored-refit", help="optional check (two arms): a paired_refit.json with G_U, G_V, Delta")
    ap.add_argument("--tol", type=float, default=1e-10)
    a = ap.parse_args(argv)
    if a.scores_dir:
        if a.class_min_pairs != CLASS_MIN_PAIRS and a.stream == "pubmed":
            raise SystemExit(f"REFUSED: --class-min-pairs other than {CLASS_MIN_PAIRS} on the PubMed score files")
        out = plan_mode(a)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, indent=1, default=float))
        return 0
    if not a.files_a or not a.files_b:
        raise SystemExit(missing_files_message("no --scores-dir and no --files-a/--files-b given"))
    three = a.files_c is not None
    names = tuple(a.names or (["A", "B", "C"] if three else ["A", "B"]))
    if len(names) != (3 if three else 2):
        raise SystemExit(f"--names needs {3 if three else 2} names")
    if three and (a.expected or a.stored_refit):
        raise SystemExit("--expected and --stored-refit apply to two arms only")
    sides = [a.files_a, a.files_b] + ([a.files_c] if three else [])
    every = [f for files in sides for f in files]
    if not any(Path(f).is_file() for f in every):
        raise SystemExit(missing_files_message("none of the files named exists"))
    if a.class_min_pairs != CLASS_MIN_PAIRS and any(is_pubmed(f) for f in every):
        raise SystemExit(f"REFUSED: --class-min-pairs other than {CLASS_MIN_PAIRS} on the PubMed score files")
    closed = closure(every)
    maps = []
    for files in sides:
        for f in files:
            refuse_development_file(f)
        m = {seed_of(f): f for f in files}
        if len(m) != len(files):
            raise SystemExit("a seed appears twice on one side")
        maps.append(m)
    maps_by_name = dict(zip(names, maps))
    arms, status, dev, rule = evaluate_sides(maps, names, a.workers)
    for s in sorted(status):
        bad = {nm: v[1] for nm, v in status[s].items() if not v[0]}
        if bad:
            print(f"  initialisation {s}: incomplete files: " + "; ".join(f"{nm}: {w}" for nm, w in bad.items()))
    if not three:
        out, note = pair_or_note(arms, names[0], names[1], names, maps_by_name, status, a.class_min_pairs, rule)
        if out is None:
            raise SystemExit(note["note"])
        out["closure"] = closed
        out["blocks"]["per_label_reproduction_largest_difference"] = dev
        print_pair(out)
        print_D(out["secondary_D_same_folds"], names)
        part_ok = out["blocks"]["partitions_shared"]
    else:
        c1, n1 = pair_or_note(arms, names[0], names[2], (names[0], names[2]), maps_by_name, status,
                              a.class_min_pairs, rule)
        c2, n2 = pair_or_note(arms, names[1], names[2], (names[1], names[2]), maps_by_name, status,
                              a.class_min_pairs, rule)
        d12, n3 = pair_or_note(arms, names[0], names[1], (names[0], names[1]), maps_by_name, status,
                               a.class_min_pairs, rule)
        done = [c for c in (c1, c2, d12) if c is not None]
        out = dict(mode="three arms", names=list(names), comparator=names[2], closure=closed,
                   contrasts={c["contrast"]: c for c in done}, notes=[n for n in (n1, n2, n3) if n],
                   partition_specification=PARTITION_SPEC,
                   D_definition=f"D = (V_{names[0]} - V_{names[1]}) - (U_sf,{names[0]} - U_sf,{names[1]}) per block, "
                                "on the common scoring halves (U_sf: theta_1 on the scoring halves of the refit's "
                                "partitions); whole-period variant beside")
        if d12 is not None:
            out["secondary_D"] = secondary_D(c1 or dict(seeds=[]), c2 or dict(seeds=[]), d12, names[:2])
        for c in done:
            print_pair(c)
        if d12 is not None:
            print_D(out["secondary_D"], names[:2])
        part_ok = all(c["blocks"]["partitions_shared"] for c in done)
    print(f"\npartitions and {LIMITED_N}-record subsamples shared by every arm of each block: {part_ok}; per-label "
          f"arrays reproduce refitgap's levels to {dev:.1e}")
    rc = 0
    if a.expected or a.stored_refit:
        worst, lines = check(out, a.expected, a.expected_key, a.stored_refit)
        print("\ncheck against stored values (" + PRIMARY + "):")
        print("\n".join(lines))
        ok = worst <= a.tol
        print(f"  largest difference {worst:.1e} (tolerance {a.tol:g}): {'MATCH' if ok else 'MISMATCH'}")
        out["check"] = dict(largest_difference=worst, tolerance=a.tol, ok=ok, lines=lines)
        rc = 0 if ok else 1
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    return rc


if __name__ == "__main__":
    sys.exit(main())
