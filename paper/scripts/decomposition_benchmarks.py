#!/usr/bin/env python3
"""Decomposition B = A + dR on the benchmark streams from the stored score matrices: U at the run's theta_1,
V by exact split-half (200 partitions shared by both models), seed-block bootstrap over both ranges, and
agreement with results/e1_results.parquet. Reads results/benchmarks/scores/ (env OUT_DIR, default
results/benchmarks; env COND, default C2); writes external_bridge.parquet and external_bridge_summary.json
there. Backs Table D.1. Needs nothing beyond the repository."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from constructed_drifts import fit_exact

ROOT = Path(".")
OUT = Path(os.environ.get("OUT_DIR", str(ROOT / "results/benchmarks")))
COND = os.environ.get("COND", "C2")
SPLITS = 200
BOOT, BSEED = 20000, 20260920
STREAMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["huffpost", "amazon", "arxivfine", "arxiv"]


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def split_half_pair(PN, Pj, Y, rng):
    vN, vj = [], []
    for _ in range(SPLITS):
        perm = rng.permutation(len(Y))
        h = len(Y) // 2
        a, b = perm[:h], perm[h:]
        vN.append(0.5 * (mf1(PN[b], Y[b], fit_exact(PN[a], Y[a])) + mf1(PN[a], Y[a], fit_exact(PN[b], Y[b]))))
        vj.append(0.5 * (mf1(Pj[b], Y[b], fit_exact(Pj[a], Y[a])) + mf1(Pj[a], Y[a], fit_exact(Pj[b], Y[b]))))
    return float(np.mean(vN)), float(np.mean(vj))


pub = pd.read_parquet(ROOT / "results/e1_results.parquet")
pub = pub[(pub.condition == COND) & (pub.protocol == "v2") & (pub.strategy == "naive")]
rerun_path = OUT / "e1_external_rerun.parquet"
rerun = pd.read_parquet(rerun_path) if rerun_path.exists() else None

rows = []
for stream in STREAMS:
    for npz in sorted((OUT / "scores").glob(f"{stream}_{COND}_s*_scores.npz")):
        seed = int(npz.stem.split("_s")[1].split("_")[0])
        z = np.load(npz)
        th = z["thresholds"].astype(np.float64)
        N = max(int(k.split("_")[1]) for k in z.files if k.startswith("P_")) + 1
        for j in range(N - 1):
            Y = z[f"Y_{j}"].astype(int)
            PN = z[f"P_{N-1}_{j}"].astype(np.float64)
            Pj = z[f"P_{j}_{j}"].astype(np.float64)
            U_N, U_j = mf1(PN, Y, th), mf1(Pj, Y, th)
            V_N, V_j = split_half_pair(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j))
            rec = dict(stream=stream, seed=seed, j=j, U_N=U_N, U_j=U_j, V_N=V_N, V_j=V_j,
                       B=U_j - U_N, A=V_j - V_N, R_N=V_N - U_N, R_ref=V_j - U_j)
            rec["dR"] = rec["R_N"] - rec["R_ref"]
            assert abs(rec["B"] - (rec["A"] + rec["dR"])) < 1e-9
            p = pub[(pub.stream == stream) & (pub.seed == seed)]
            if len(p):
                rec["pub_f1_final"] = float(p[(p.i == N - 1) & (p.j == j)].macro_f1.iloc[0])
                rec["pub_f1_diag"] = float(p[(p.i == j) & (p.j == j)].macro_f1.iloc[0])
                rec["pub_auprc_bwt_j"] = float(p[(p.i == N - 1) & (p.j == j)].macro_auprc.iloc[0]
                                               - p[(p.i == j) & (p.j == j)].macro_auprc.iloc[0])
            if rerun is not None:
                r = rerun[(rerun.stream == stream) & (rerun.seed == seed)]
                rec["rerun_f1_final"] = float(r[(r.i == N - 1) & (r.j == j)].macro_f1.iloc[0])
                rec["rerun_auprc_bwt_j"] = float(r[(r.i == N - 1) & (r.j == j)].macro_auprc.iloc[0]
                                                 - r[(r.i == j) & (r.j == j)].macro_auprc.iloc[0])
            rows.append(rec)
        print(f"  {stream} seed {seed} done", flush=True)

t = pd.DataFrame(rows)
t.to_parquet(OUT / "external_bridge.parquet", index=False)
if rerun is not None:
    # scores are stored in reduced precision, so a record on a threshold can flip
    gap = float((t.U_N - t.rerun_f1_final).abs().max())
    print(f"  max |U_N - rerun macro-F1| over cells = {gap:.5f}", flush=True)
    assert gap < 3e-3, "saved scores disagree with the rerun rows beyond storage precision"

QUANT = ["B", "A", "dR", "R_N", "R_ref"]
if "pub_auprc_bwt_j" in t:
    QUANT += ["pub_auprc_bwt_j"]
if "rerun_auprc_bwt_j" in t:
    QUANT += ["rerun_auprc_bwt_j"]


def boot_mean(blk, fn):
    rng = np.random.default_rng(BSEED)
    n = len(blk)
    vals = blk.to_numpy()
    draws = np.empty(BOOT)
    for r in range(BOOT):
        k = rng.integers(0, n, n)
        draws[r] = fn(pd.DataFrame(vals[k], columns=blk.columns))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


summary = dict(splits=SPLITS, bootstrap=dict(B=BOOT, seed=BSEED, blocks="seed"), streams={})
for stream, sub in t.groupby("stream"):
    out = {}
    for rname, rsub in (("restricted_j1to6", sub[sub.j >= 1]), ("conventional_j0to6", sub)):
        blk = rsub.groupby("seed")[QUANT].mean()
        res = {}
        for q in QUANT:
            res[q] = dict(mean=float(blk[q].mean()), ci=boot_mean(blk, lambda d, q=q: d[q].mean()),
                          sd_seeds=float(blk[q].std(ddof=0)), n_seeds_positive=int((blk[q] > 0).sum()))
        for num, den, name in (("A", "B", "A_over_B"), ("dR", "B", "dR_over_B")):
            res[name] = dict(ratio_of_means=float(blk[num].mean() / blk[den].mean()),
                             ci=boot_mean(blk, lambda d, a=num, b=den: d[a].mean() / d[b].mean()))
        res["per_period_mean"] = rsub.groupby("j")[["B", "A", "dR"]].mean().round(6).to_dict(orient="index")
        out[rname] = res
    if "pub_f1_final" in sub:
        out["reproduction"] = dict(
            maxabs_f1_final_vs_published=float((sub.U_N - sub.pub_f1_final).abs().max()),
            meanabs_f1_final_vs_published=float((sub.U_N - sub.pub_f1_final).abs().mean()),
            B_published_restricted=float((sub[sub.j >= 1].pub_f1_diag - sub[sub.j >= 1].pub_f1_final).mean()),
            B_rerun_restricted=float(sub[sub.j >= 1].B.mean()))
    summary["streams"][stream] = out
    r = out["restricted_j1to6"]
    print(f"{stream:10s} B {r['B']['mean']:+.4f}  A {r['A']['mean']:+.4f}  dR {r['dR']['mean']:+.4f}  "
          f"R_N {r['R_N']['mean']:+.4f}  R_ref {r['R_ref']['mean']:+.4f}  dR/B {r['dR_over_B']['ratio_of_means']:.2f}")

(OUT / "external_bridge_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", OUT / "external_bridge_summary.json")
