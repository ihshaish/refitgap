"""Carried rules against the refitted lead (plans/carried-rules.txt): the shared engine. Stored scores
only; nothing is trained. Imported by the carried_rules_*.py scripts.

For each DistilBERT comparison (replay minus sequential training on ASRS, MultiEURLEX and PubMed;
replay minus extra updates on current data on MultiEURLEX and PubMed), both final models are scored on
every scoring half of the paired refit's 200 partitions (default_rng(7000 + 100*seed + period id)),
periods 2 to 7, under seven rules:
  a theta1         the inherited thresholds of each arm
  b cut0.5         a cut of 0.5 on every label
  c own_val        each final model's grid rule on the first-period validation split
  d prev_period    thresholds fitted on 500 test records of the previous period, one draw per partition
                   from default_rng(11000 + 100*seed + period id), the same records for both arms
  f matched_rate   per label and half, both arms flag the same number of records: the mean of the two
                   arms' counts at theta1, rounded up, taken from each arm's highest scores
  e refit          the paired refit (fit on the opposite half)
  e500 refit500    the same refit on 500 records of the opposite half, drawn per partition and direction
                   from default_rng(9000 + 100*seed + period id)
The counts (tp, fp, fn per label) are kept for every half, so the CSVs and every level come from them.
Thresholds, F1, partitions and t intervals are refitgap's. Rules d and e500 are also in the package:
refitgap.evaluate(..., previous=..., refit_n=500) and refitgap.rule_lead() give the same leads.

Score files (not in the repository): one directory, passed as SCORES_DIR, holding
  primary_distilbert_{naive,replay}_s0-5_scores.npz                    ASRS
  eurlex_distilbert_{naive,replay,replaycur}_s0-5_scores.npz           MultiEURLEX
  pubmed_distilbert_{naive,replay,replaycur,lwfa0.02}_s0-7_scores.npz  PubMed
  eurlex_C2_lwf_a0.1_T2_lam1_s0-9_scores.npz                           MultiEURLEX BiGRU LwF (item 3)
  {asrsprior100,asrsprior00,asrsperm100,primary}_C2_{naive,replay}_lam1_s0-9_scores.npz
                                                                       constructed streams (item 4)
The BiGRU sequential and replay runs on MultiEURLEX are read from results/multieurlex_bigru/.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from refitgap.crossfit import fit_validation_rule, partitions
from refitgap.fitters import fit_exact
from refitgap import inference

PAPER = Path(__file__).resolve().parents[1]          # paper/
PLAN = PAPER / "plans" / "carried-rules.txt"
DEFAULT_OUT = PAPER / "results" / "carried_rules_rerun"   # the stored outputs stay in results/carried_rules/
BIGRU_EURLEX = PAPER / "results" / "multieurlex_bigru"

SPLITS, MARGIN = 200, 0.005
PIDS = list(range(1, 7))                            # periods 2 to 7
FINAL = 7
LIMITED_N, LIMITED_BASE = 500, 9000
PREV_N, PREV_BASE = 500, 11000
BOOT, BOOT_SEED = 20000, 20260920
RULES = ["theta1", "cut0.5", "own_val", "prev_period", "matched_rate", "refit", "refit500"]
RULE_LETTER = {"theta1": "a", "cut0.5": "b", "own_val": "c", "prev_period": "d", "matched_rate": "f",
               "refit": "e", "refit500": "e500"}
WHOLE_RULES = ["theta1", "cut0.5", "own_val"]

STREAMS = {
    "ASRS": dict(prefix="primary_distilbert", seeds=list(range(6))),
    "MultiEURLEX": dict(prefix="eurlex_distilbert", seeds=list(range(6))),
    "PubMed": dict(prefix="pubmed_distilbert", seeds=list(range(8))),
}
ARM_TAG = {"sequential training": "naive", "replay": "replay", "extra updates on current data": "replaycur",
           "LwF": "lwfa0.02"}
COMPARISONS = [("ASRS", "replay", "sequential training"),
               ("MultiEURLEX", "replay", "sequential training"),
               ("MultiEURLEX", "replay", "extra updates on current data"),
               ("PubMed", "replay", "sequential training"),
               ("PubMed", "replay", "extra updates on current data")]

SCORES = None                                       # set by arguments()


def arguments(doc, scores=True):
    """Command line shared by the scripts: [SCORES_DIR] [--out DIR] [--workers N]."""
    global SCORES
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    if scores:
        ap.add_argument("scores", help="directory holding the score files (not in the repository)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default results/carried_rules_rerun)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if scores:
        SCORES = Path(a.scores)
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def need(paths):
    """Stop with a plain message when score files are missing."""
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(paths)} score files needed here are missing (first: {missing[0]}).\n"
            "These analyses read stored score files that are not in the repository (README.txt, paper/README.txt).\n"
            "Pass the directory that holds them as the first argument; the file names are listed at the top of "
            "scripts/carried_rules_core.py.")


def arm_file(stream, arm, seed):
    return SCORES / f"{STREAMS[stream]['prefix']}_{ARM_TAG[arm]}_s{seed}_scores.npz"


def comparison_files(comparisons=COMPARISONS):
    return [arm_file(st, arm, s) for st, a, b in comparisons for arm in (a, b) for s in STREAMS[st]["seeds"]]


def sha256(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def comp_name(stream, a, b):
    return f"{stream}: {a} minus {b}"


# ------------------------------------------------------------------ loading ---
def load_arm(path):
    z = np.load(path)
    P = {j: z[f"P_{FINAL}_{j}"].astype(np.float64) for j in range(8)}
    Y = {j: z[f"Y_{j}"].astype(int) for j in range(8)}
    theta = z["thresholds"].astype(np.float64)
    pval = z[f"Pval_{FINAL}"] if f"Pval_{FINAL}" in z.files else None
    yval = z["Yval"] if "Yval" in z.files else None
    own = fit_validation_rule(pval, yval) if pval is not None else None
    return dict(P=P, Y=Y, theta=theta, own=own, n_val=(int(len(yval)) if yval is not None else None),
                score_min=float(min(P[j].min() for j in range(8))), score_max=float(max(P[j].max() for j in range(8))))


# ------------------------------------------------------------------ counting ---
def counts(pred, Y):
    """Boolean predictions and 0/1 labels (n, K) -> tp, fp, fn per label (int arrays)."""
    Yb = Y.astype(bool)
    tp = (pred & Yb).sum(0)
    fp = (pred & ~Yb).sum(0)
    fn = (~pred & Yb).sum(0)
    return tp, fp, fn


def f1_from(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, np.float64) for x in (tp, fp, fn))
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2.0 * tp / np.maximum(den, 1), 0.0)


def matched_flags(PA, PB_, thA, thB, S_sorted):
    """Rule f on one scoring half. PA, PB_: the two arms' scores of the period (n, K); S_sorted: the half's row
    positions in ascending order. Returns boolean flags (len(S), K) for A and B, and m per label."""
    sa, sb = PA[S_sorted], PB_[S_sorted]
    cA = (sa >= thA).sum(0)
    cB = (sb >= thB).sum(0)
    m = (cA + cB + 1) // 2                                   # ceil((cA + cB) / 2) in integers
    out = []
    for s in (sa, sb):
        order = np.argsort(-s, axis=0, kind="stable")        # ties: ascending row position
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(len(s))[:, None].repeat(s.shape[1], 1), 0)
        out.append(rank < m[None, :])
    return out[0], out[1], m


def prev_draws(seed, j, n_prev, splits=SPLITS):
    """Rule d: one draw per partition of 500 row positions of period j-1, default_rng(11000 + 100*seed + j).
    The same as refitgap.previous_draws(n_prev, seed, j, 500, splits)."""
    rng = np.random.default_rng(PREV_BASE + 100 * seed + j)
    return [rng.choice(n_prev, PREV_N, replace=False) for _ in range(splits)]


def limited_draws(seed, j, h0, h1, splits=SPLITS):
    """Rule e500: default_rng(9000 + 100*seed + j), per partition and direction, positions within the fitting
    half (whole half when it has fewer than 500 records). The same as refitgap.limited_draws(n, seed, j, 500)."""
    rng = np.random.default_rng(LIMITED_BASE + 100 * seed + j)
    out = []
    for b in range(splits):
        row = []
        for d in (0, 1):
            nf = h0 if d == 0 else h1
            row.append(rng.choice(nf, LIMITED_N, replace=False) if nf >= LIMITED_N else np.arange(nf))
        out.append(row)
    return out


def one_comparison(task):
    """(scores_dir, stream, armA, armB, seed, splits[, rules]) -> per-rule counts for both arms on every scoring
    half of periods 2-7, whole-period counts for rules a-c, and checks. Arrays: tp/fp/fn
    [arm, rule, period, partition, half, label]; n_scoring [period, partition, half]; pos [period, partition,
    half, label]."""
    global SCORES
    SCORES = Path(task[0])
    stream, a, b, seed, splits = task[1:6]
    only = set(task[6]) if len(task) > 6 and task[6] else set(RULES)
    X = [load_arm(arm_file(stream, a, seed)), load_arm(arm_file(stream, b, seed))]
    for j in range(8):
        if not np.array_equal(X[0]["Y"][j], X[1]["Y"][j]):
            raise ValueError(f"{stream} seed {seed}: labels differ between arms in period {j}")
    K = len(X[0]["theta"])
    R, NP = len(RULES), len(PIDS)
    TP, FP, FN = (np.zeros((2, R, NP, splits, 2, K), np.int32) for _ in range(3))
    NS = np.zeros((NP, splits, 2), np.int32)
    POS = np.zeros((NP, splits, 2, K), np.int32)
    NFIT = np.zeros((NP, splits, 2), np.int32)
    W = {k: np.zeros((2, NP, 3, K), np.int32) for k in WHOLE_RULES}       # whole period: tp, fp, fn
    WN = np.zeros(NP, np.int32)
    WPOS = np.zeros((NP, K), np.int32)
    checks = dict(prev_draw_hash=[], limited_draw_hash=[], partition_hash=[], matched_equal_counts=True,
                  short_halves=[])
    cut = np.full(K, 0.5)
    for c, j in enumerate(PIDS):
        Y = X[0]["Y"][j]
        n = len(Y)
        h = n // 2
        perms = partitions(n, seed, j, splits)
        checks["partition_hash"].append(hashlib.sha256(np.stack(perms).astype(np.int64).tobytes()).hexdigest()[:16])
        pdraw = prev_draws(seed, j, len(X[0]["Y"][j - 1]), splits)
        checks["prev_draw_hash"].append(hashlib.sha256(np.stack(pdraw).astype(np.int64).tobytes()).hexdigest()[:16])
        ldraw = limited_draws(seed, j, h, n - h, splits)
        checks["limited_draw_hash"].append(hashlib.sha256(np.concatenate(
            [x for r in ldraw for x in r]).astype(np.int64).tobytes()).hexdigest()[:16])
        WN[c] = n
        WPOS[c] = Y.sum(0)
        for m, Xm in enumerate(X):
            P = Xm["P"][j]
            for k, th in (("theta1", Xm["theta"]), ("cut0.5", cut), ("own_val", Xm["own"])):
                W[k][m, c] = np.stack(counts(P >= th, Y))
        Yp = X[0]["Y"][j - 1]
        th_prev = {}
        for bb in range(splits):
            sel = pdraw[bb]
            th_prev[bb] = ([fit_exact(Xm["P"][j - 1][sel], Yp[sel]) for Xm in X] if "prev_period" in only
                           else [None, None])
        for bb in range(splits):
            A_, B_ = perms[bb][:h], perms[bb][h:]
            for d in (0, 1):
                F, S = (A_, B_) if d == 0 else (B_, A_)
                if len(F) < LIMITED_N:
                    checks["short_halves"].append((j, bb, d, len(F)))
                Fsub = F[ldraw[bb][d]]
                Ss = np.sort(S)
                Ys = Y[Ss]
                NS[c, bb, d] = len(S)
                NFIT[c, bb, d] = len(F)
                POS[c, bb, d] = Ys.sum(0)
                if "matched_rate" in only:
                    fA, fB, mm = matched_flags(X[0]["P"][j], X[1]["P"][j], X[0]["theta"], X[1]["theta"], Ss)
                    if not (np.array_equal(fA.sum(0), mm) and np.array_equal(fB.sum(0), mm)):
                        checks["matched_equal_counts"] = False
                    flags = (fA, fB)
                for m, Xm in enumerate(X):
                    P = Xm["P"][j]
                    Pf, Ps = P[F], P[Ss]
                    ths = dict(theta1=Xm["theta"], cut0_5=cut, own_val=Xm["own"], prev_period=th_prev[bb][m],
                               refit=fit_exact(Pf, Y[F]),
                               refit500=(fit_exact(P[Fsub], Y[Fsub]) if "refit500" in only else None))
                    for r, rule in enumerate(RULES):
                        if rule not in only:
                            continue
                        if rule == "matched_rate":
                            pred = flags[m]
                        else:
                            pred = Ps >= ths["cut0_5" if rule == "cut0.5" else rule]
                        tp, fp, fn = counts(pred, Ys)
                        TP[m, r, c, bb, d], FP[m, r, c, bb, d], FN[m, r, c, bb, d] = tp, fp, fn
    return dict(stream=stream, arms=[a, b], seed=seed, splits=splits, K=K, rules_computed=sorted(only), TP=TP, FP=FP,
                FN=FN, NS=NS, POS=POS, NFIT=NFIT, W=W, WN=WN, WPOS=WPOS, checks=checks,
                theta=[Xm["theta"].tolist() for Xm in X], own=[Xm["own"].tolist() for Xm in X],
                n_val=X[0]["n_val"], score_range=[min(x["score_min"] for x in X), max(x["score_max"] for x in X)],
                own_differs_from_theta1=[int(np.sum(Xm["own"] != Xm["theta"])) for Xm in X],
                theta1_differs_between_arms=int(np.sum(X[0]["theta"] != X[1]["theta"])))


# ------------------------------------------------------------------ levels ---
def f1_halves(res, m, rule):
    """Per-label F1 on the scoring halves, (period, partition, half, K)."""
    r = RULES.index(rule)
    return f1_from(res["TP"][m, r], res["FP"][m, r], res["FN"][m, r])


def level_halves(res, m, rule):
    """Macro-F1 averaged over halves, partitions and periods (uniform), one arm."""
    return float(f1_halves(res, m, rule).mean(-1).mean(axis=(1, 2)).mean())


def label_level_halves(res, m, rule):
    """Per-label F1 averaged over halves, partitions, periods: (K,)."""
    return f1_halves(res, m, rule).mean(axis=(1, 2)).mean(0)


def f1_whole(res, m, rule):
    w = res["W"][rule][m]
    return f1_from(w[:, 0], w[:, 1], w[:, 2])                  # (period, K)


def level_whole(res, m, rule):
    return float(f1_whole(res, m, rule).mean(-1).mean())


# ------------------------------------------------------------------ inference ---
def t_summary(x):
    x = np.asarray(x, float)
    st = inference.paired_stats(x, MARGIN)
    return dict(mean=float(x.mean()), t95=st.get("ci95"), per_pair=x.tolist(), positive=int((x > 0).sum()),
                negative=int((x < 0).sum()), zero=int((x == 0).sum()), n=int(len(x)))


def boot_summary(cols, names):
    """Percentile bootstrap over pairs, one index matrix shared by the columns."""
    v = np.column_stack([np.asarray(c, float) for c in cols])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(v), (BOOT, len(v)))
    b = v[idx].mean(axis=1)
    out = {}
    for c, nm in enumerate(names):
        x = v[:, c]
        out[nm] = dict(mean=float(x.mean()), boot95=[float(np.percentile(b[:, c], 2.5)),
                                                     float(np.percentile(b[:, c], 97.5))],
                       per_pair=x.tolist(), positive=int((x > 0).sum()), negative=int((x < 0).sum()),
                       zero=int((x == 0).sum()), n=int(len(x)))
    return out


# ------------------------------------------------------------- BiGRU (items 3 and 4) ---
def refit_arm(task):
    """(path, seed, splits) -> one final model's counts on periods 2-7: theta1 on the whole period, theta1 on each
    scoring half and the full refit (fit_exact on the opposite half), as for Table C.1. Arrays as one_comparison
    with a single arm and rules ["theta1", "refit"]."""
    path, seed, splits = task
    z = np.load(path)
    theta = z["thresholds"].astype(np.float64)
    K = len(theta)
    NP = len(PIDS)
    TP, FP, FN = (np.zeros((2, NP, splits, 2, K), np.int32) for _ in range(3))
    NS = np.zeros((NP, splits, 2), np.int32)
    POS = np.zeros((NP, splits, 2, K), np.int32)
    NFIT = np.zeros((NP, splits, 2), np.int32)
    W = np.zeros((NP, 3, K), np.int32)
    WN = np.zeros(NP, np.int32)
    WPOS = np.zeros((NP, K), np.int32)
    phash = []
    for c, j in enumerate(PIDS):
        P = z[f"P_{FINAL}_{j}"].astype(np.float64)
        Y = z[f"Y_{j}"].astype(int)
        n = len(Y)
        h = n // 2
        perms = partitions(n, seed, j, splits)
        phash.append(hashlib.sha256(np.stack(perms).astype(np.int64).tobytes()).hexdigest()[:16])
        W[c] = np.stack(counts(P >= theta, Y))
        WN[c] = n
        WPOS[c] = Y.sum(0)
        for bb in range(splits):
            A_, B_ = perms[bb][:h], perms[bb][h:]
            for d in (0, 1):
                F, S = (A_, B_) if d == 0 else (B_, A_)
                Ps, Ys = P[S], Y[S]
                NS[c, bb, d], NFIT[c, bb, d] = len(S), len(F)
                POS[c, bb, d] = Ys.sum(0)
                for r, th in enumerate((theta, fit_exact(P[F], Y[F]))):
                    tp, fp, fn = counts(Ps >= th, Ys)
                    TP[r, c, bb, d], FP[r, c, bb, d], FN[r, c, bb, d] = tp, fp, fn
    return dict(path=str(path), seed=seed, splits=splits, K=K, TP=TP, FP=FP, FN=FN, NS=NS, POS=POS, NFIT=NFIT, W=W,
                WN=WN, WPOS=WPOS, theta=theta.tolist(), partition_hash=phash,
                label_hash=[hashlib.sha256(z[f"Y_{j}"].astype(np.int8).tobytes()).hexdigest()[:16] for j in PIDS])


def arm_U(res):
    """Whole-period macro-F1 at theta1, mean over periods 2-7; and per period."""
    f = f1_from(res["W"][:, 0], res["W"][:, 1], res["W"][:, 2]).mean(-1)
    return float(f.mean()), f


def arm_V(res, rule=1):
    """Macro-F1 on the scoring halves (rule 1 = refit, 0 = theta1), averaged as V; and per period."""
    f = f1_from(res["TP"][rule], res["FP"][rule], res["FN"][rule]).mean(-1).mean(axis=(1, 2))
    return float(f.mean()), f


if __name__ == "__main__":
    sys.exit("imported by the carried_rules_*.py scripts; see plans/carried-rules.txt")
