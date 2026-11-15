#!/usr/bin/env python3
"""Decomposition B = A + dR on the DistilBERT ASRS score matrices, as decomposition_benchmarks.py, with dR - A
(bootstrap and sign-flip p), the first-period same-scorer refresh, threshold updates on n = 500 and 200 test
records (scored on the rest), and U-bar with restricted backward transfer. Reads
results/distilbert_decomposition/*_scores.npz and e4_results_seed*.parquet; writes distilbert_bridge{.parquet,
_summary.json} there. Backs the DistilBERT decomposition table of Appendix C. Needs nothing beyond the repository."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from constructed_drifts import fit_exact

ROOT = Path(".")
OUT = ROOT / "results/distilbert_decomposition"
SPLITS = 200
BOOT, BSEED = 20000, 20260920
GRID = np.linspace(0.05, 0.95, 19)
DRAWS, NS = 20, (200, 500)
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else None


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def fit_grid(P, Y):
    ths = []
    for k in range(Y.shape[1]):
        f = [f1_score(Y[:, k], (P[:, k] >= g).astype(int), zero_division=0) for g in GRID]
        ths.append(GRID[int(np.argmax(f))])
    return np.asarray(ths)


def split_half_pair(PN, Pj, Y, rng):
    vN, vj = [], []
    for _ in range(SPLITS):
        perm = rng.permutation(len(Y))
        h = len(Y) // 2
        a, b = perm[:h], perm[h:]
        vN.append(0.5 * (mf1(PN[b], Y[b], fit_exact(PN[a], Y[a])) + mf1(PN[a], Y[a], fit_exact(PN[b], Y[b]))))
        vj.append(0.5 * (mf1(Pj[b], Y[b], fit_exact(Pj[a], Y[a])) + mf1(Pj[a], Y[a], fit_exact(Pj[b], Y[b]))))
    return float(np.mean(vN)), float(np.mean(vj))


rows, table_rows = [], []
for npz in sorted(OUT.glob("primary_distilbert_s*_scores.npz")):
    seed = int(npz.stem.split("_s")[1].split("_")[0])
    if SEEDS is not None and seed not in SEEDS:
        continue
    z = np.load(npz)
    th = z["thresholds"].astype(np.float64)
    N = max(int(k.split("_")[1]) for k in z.files if k.startswith("P_")) + 1
    th_refresh = fit_grid(z[f"Pval_{N-1}"].astype(np.float64), z["Yval"].astype(int))
    pq = OUT / f"e4_results_seed{seed}.parquet"
    res = pd.read_parquet(pq) if pq.exists() else None
    for j in range(N - 1):
        Y = z[f"Y_{j}"].astype(int)
        PN = z[f"P_{N-1}_{j}"].astype(np.float64)
        Pj = z[f"P_{j}_{j}"].astype(np.float64)
        U_N, U_j = mf1(PN, Y, th), mf1(Pj, Y, th)
        V_N, V_j = split_half_pair(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j))
        rec = dict(seed=seed, j=j, U_N=U_N, U_j=U_j, V_N=V_N, V_j=V_j,
                   B=U_j - U_N, A=V_j - V_N, R_N=V_N - U_N, R_ref=V_j - U_j)
        rec["dR"] = rec["R_N"] - rec["R_ref"]
        rec["dR_minus_A"] = rec["dR"] - rec["A"]
        assert abs(rec["B"] - (rec["A"] + rec["dR"])) < 1e-9
        rec["U_refresh"] = mf1(PN, Y, th_refresh)
        rec["refresh_gain"] = rec["U_refresh"] - U_N
        for n in NS:
            gains = []
            for d in range(DRAWS):
                srng = np.random.default_rng(100000 + seed * 1000 + j * 40 + d)
                sel = srng.choice(len(Y), min(n, len(Y) // 2), replace=False)
                rest = np.setdiff1d(np.arange(len(Y)), sel)
                th_n = fit_grid(PN[sel], Y[sel])
                gains.append(mf1(PN[rest], Y[rest], th_n) - mf1(PN[rest], Y[rest], th))
            rec[f"gain_n{n}"] = float(np.mean(gains))
        if res is not None:
            r = res[(res.i == N - 1) & (res.j == j)]
            rec["pq_f1_final"] = float(r.macro_f1.iloc[0])
            rec["pq_auprc_bwt_j"] = float(r.macro_auprc.iloc[0]
                                          - res[(res.i == j) & (res.j == j)].macro_auprc.iloc[0])
        rows.append(rec)
    # U-bar and restricted backward transfer
    Ubar = float(np.mean([mf1(z[f"P_{N-1}_{j}"].astype(np.float64), z[f"Y_{j}"].astype(int), th) for j in range(N)]))
    bwt_f1 = float(np.mean([r["B"] for r in rows if r["seed"] == seed and r["j"] >= 1]) * -1)
    trow = dict(seed=seed, Ubar=Ubar, bwt_f1_restricted=bwt_f1)
    if res is not None:
        M = np.full((N, N), np.nan)
        for _, r in res.iterrows():
            M[int(r.i), int(r.j)] = r.macro_auprc
        trow["bwt_auprc_restricted"] = float(np.mean([M[N - 1, j] - M[j, j] for j in range(1, N - 1)]))
        trow["Ubar_parquet"] = float(np.mean([res[(res.i == N - 1) & (res.j == j)].macro_f1.iloc[0] for j in range(N)]))
    table_rows.append(trow)
    print(f"  seed {seed} done", flush=True)

t = pd.DataFrame(rows)
t.to_parquet(OUT / "distilbert_bridge.parquet", index=False)
if "pq_f1_final" in t:
    gap = float((t.U_N - t.pq_f1_final).abs().max())
    print(f"  max |U_N - parquet macro-F1| over cells = {gap:.5f}", flush=True)
    assert gap < 3e-3, "saved scores disagree with the run's rows beyond storage precision"

QUANT = ["B", "A", "dR", "dR_minus_A", "R_N", "R_ref", "refresh_gain"] + [f"gain_n{n}" for n in NS]
if "pq_auprc_bwt_j" in t:
    QUANT.append("pq_auprc_bwt_j")


def boot_mean(blk, fn):
    rng = np.random.default_rng(BSEED)
    n = len(blk)
    vals = blk.to_numpy()
    draws = np.empty(BOOT)
    for r in range(BOOT):
        k = rng.integers(0, n, n)
        draws[r] = fn(pd.DataFrame(vals[k], columns=blk.columns))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def signflip_p(x):
    x = np.asarray(x, dtype=float)
    obs = abs(x.mean())
    hits = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        if abs((x * np.asarray(signs)).mean()) >= obs - 1e-15:
            hits += 1
    return hits / 2 ** len(x)


summary = dict(splits=SPLITS, bootstrap=dict(B=BOOT, seed=BSEED, blocks="seed"),
               seeds=sorted(int(s) for s in t.seed.unique()), ranges={})
for rname, rsub in (("restricted_j1to6", t[t.j >= 1]), ("conventional_j0to6", t)):
    blk = rsub.groupby("seed")[QUANT].mean()
    res = {}
    for q in QUANT:
        res[q] = dict(mean=float(blk[q].mean()), ci=boot_mean(blk, lambda d, q=q: d[q].mean()),
                      sd_seeds=float(blk[q].std(ddof=0)), n_seeds_positive=int((blk[q] > 0).sum()),
                      n_seeds=int(len(blk)))
    res["dR_minus_A"]["signflip_p_two_sided"] = signflip_p(blk["dR_minus_A"].to_numpy())
    for num, den, name in (
        ("A", "B", "A_over_B"),
        ("dR", "B", "dR_over_B"),
        ("refresh_gain", "R_N", "refresh_share_of_R_N"),
        ("gain_n500", "R_N", "capture_n500"),
        ("gain_n200", "R_N", "capture_n200"),
    ):
        res[name] = dict(ratio_of_means=float(blk[num].mean() / blk[den].mean()),
                         ci=boot_mean(blk, lambda d, a=num, b=den: d[a].mean() / d[b].mean()))
    res["per_period_mean"] = rsub.groupby("j")[["B", "A", "dR", "R_N", "R_ref"]].mean().round(6).to_dict(orient="index")
    summary["ranges"][rname] = res

tb = pd.DataFrame(table_rows)
summary["table3_row"] = {c: dict(mean=float(tb[c].mean()), sd_ddof1=float(tb[c].std(ddof=1)) if len(tb) > 1 else None)
                         for c in tb.columns if c != "seed"}
summary["table3_row"]["n"] = int(len(tb))
summary["finite_label_update_note"] = (
    "n records drawn from period j's test set (no train-pool scores stored); "
    "refit evaluated on the remaining test records against theta_1 on the same remainder"
)
(OUT / "distilbert_bridge_summary.json").write_text(json.dumps(summary, indent=1))

r = summary["ranges"]["restricted_j1to6"]
print(f"seeds {summary['seeds']}")
print(f"restricted  B {r['B']['mean']:+.4f} {r['B']['ci']}  A {r['A']['mean']:+.4f} {r['A']['ci']}  "
      f"dR {r['dR']['mean']:+.4f} {r['dR']['ci']}")
print(
    f"            dR-A {r['dR_minus_A']['mean']:+.4f} {r['dR_minus_A']['ci']}  positive blocks "
    f"{r['dR_minus_A']['n_seeds_positive']}/{r['dR_minus_A']['n_seeds']}  sign-flip p "
    f"{r['dR_minus_A']['signflip_p_two_sided']:.3f}"
)
print(
    f"            R_N {r['R_N']['mean']:+.4f}  R_ref {r['R_ref']['mean']:+.4f}  dR/B "
    f"{r['dR_over_B']['ratio_of_means']:.2f} "
    f"{r['dR_over_B']['ci']}  A/B {r['A_over_B']['ratio_of_means']:.2f}"
)
print(
    f"            refresh share of R_N {r['refresh_share_of_R_N']['ratio_of_means']:.2f} "
    f"{r['refresh_share_of_R_N']['ci']}  "
    f"capture n500 {r['capture_n500']['ratio_of_means']:.2f}  n200 {r['capture_n200']['ratio_of_means']:.2f}"
)
c = summary["ranges"]["conventional_j0to6"]
print(
    f"conventional B {c['B']['mean']:+.4f}  A {c['A']['mean']:+.4f}  dR {c['dR']['mean']:+.4f}  dR-A "
    f"{c['dR_minus_A']['mean']:+.4f} {c['dR_minus_A']['ci']}"
)
print("table 3 row:", json.dumps(summary["table3_row"]))
print("wrote", OUT / "distilbert_bridge_summary.json")
