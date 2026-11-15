#!/usr/bin/env python3
"""Paired refit, BiGRU: replay against sequential training, paired by seed, at theta_1 (G_U), after both refit
by exact split-half (G_V) and the change Delta = G_U - G_V, with threshold updates on n = 500 and 200 records
and current-period leads; bootstrap and sign-flip p over pairs. python scripts/paired_refit_bigru.py [--stream eurlex]
Reads results/asrs_bigru_paired/ (or results/multieurlex_bigru/); writes
results/paired_refit/paired_replay[_eurlex].json.
Backs Table C.1 and Figure 2 (BiGRU rows). Needs the stream files for the resource counts at the end (not shipped)."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
from itertools import product
from sklearn.metrics import f1_score
from constructed_drifts import fit_exact

STREAM = sys.argv[sys.argv.index("--stream") + 1] if "--stream" in sys.argv else "primary"
D = "results/asrs_bigru_paired" if STREAM == "primary" else "results/multieurlex_bigru"
OUT = (
    "results/paired_refit/paired_replay.json"
    if STREAM == "primary"
    else f"results/paired_refit/paired_replay_{STREAM}.json"
)
M, MARGIN, BOOT, DRAWS, NFIN = 200, 0.005, 20000, 20, (500, 200)


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def split_half_pair(PA, PB, Y, rng):
    n = len(Y); h = n // 2; va, vb = [], []
    for _ in range(M):
        perm = rng.permutation(n); a, b = perm[:h], perm[h:]
        va.append(0.5 * (mf1(PA[b], Y[b], fit_exact(PA[a], Y[a])) + mf1(PA[a], Y[a], fit_exact(PA[b], Y[b]))))
        vb.append(0.5 * (mf1(PB[b], Y[b], fit_exact(PB[a], Y[a])) + mf1(PB[a], Y[a], fit_exact(PB[b], Y[b]))))
    return float(np.mean(va)), float(np.mean(vb))


def finite_update(PA, PB, Y, seed, j):
    out = {n: [[], []] for n in NFIN}
    for d in range(DRAWS):
        perm = np.random.default_rng(100000 + seed * 1000 + j * 40 + d).permutation(len(Y))
        for n in NFIN:
            fit, ev = perm[:n], perm[n:]
            out[n][0].append(mf1(PA[ev], Y[ev], fit_exact(PA[fit], Y[fit])))
            out[n][1].append(mf1(PB[ev], Y[ev], fit_exact(PB[fit], Y[fit])))
    return {n: (float(np.mean(v[0])), float(np.mean(v[1]))) for n, v in out.items()}


seeds = sorted(
    {int(re.search(r"_s(\d+)_scores", f).group(1)) for f in glob.glob(f"{D}/{STREAM}_C2_naive_lam1_s*_scores.npz")}
    & {int(re.search(r"_s(\d+)_scores", f).group(1)) for f in glob.glob(f"{D}/{STREAM}_C2_replay_lam1_s*_scores.npz")}
)
pairs = {}
for s in seeds:
    A = np.load(f"{D}/{STREAM}_C2_naive_lam1_s{s}_scores.npz")
    B = np.load(f"{D}/{STREAM}_C2_replay_lam1_s{s}_scores.npz")
    same_start = bool(np.array_equal(A["P_0_0"], B["P_0_0"]) and np.array_equal(A["thresholds"], B["thresholds"]))
    th = A["thresholds"].astype(np.float64); N = 1 + max(int(k.split("_")[1]) for k in A.files if k.startswith("P_"))
    per = {}
    for j in range(1, N - 1):
        PA = A[f"P_{N-1}_{j}"].astype(np.float64)
        PB = B[f"P_{N-1}_{j}"].astype(np.float64)
        Y = A[f"Y_{j}"].astype(int)
        assert np.array_equal(Y, B[f"Y_{j}"])
        uA, uB = mf1(PA, Y, th), mf1(PB, Y, th)
        vA, vB = split_half_pair(PA, PB, Y, np.random.default_rng(7000 + 100 * s + j))
        fin = finite_update(PA, PB, Y, s, j)
        per[j] = dict(
            U_seq=uA,
            U_rep=uB,
            V_seq=vA,
            V_rep=vB,
            **{f"U{n}_seq": fin[n][0] for n in NFIN},
            **{f"U{n}_rep": fin[n][1] for n in NFIN},
        )
    cur = {}
    for j in range(1, N):
        PA = A[f"P_{j}_{j}"].astype(np.float64); PB = B[f"P_{j}_{j}"].astype(np.float64); Y = A[f"Y_{j}"].astype(int)
        vA, vB = split_half_pair(PA, PB, Y, np.random.default_rng(9000 + 100 * s + j))
        cur[j] = dict(U_seq=mf1(PA, Y, th), U_rep=mf1(PB, Y, th), V_seq=vA, V_rep=vB)
    agg = {k: float(np.mean([per[j][k] for j in per])) for k in per[1]}
    agg.update(G_U=agg["U_rep"] - agg["U_seq"], G_V=agg["V_rep"] - agg["V_seq"],
               R_seq=agg["V_seq"] - agg["U_seq"], R_rep=agg["V_rep"] - agg["U_rep"])
    agg["Delta"] = agg["G_U"] - agg["G_V"]
    for n in NFIN:
        agg[f"G{n}"] = agg[f"U{n}_rep"] - agg[f"U{n}_seq"]
    cagg = {k: float(np.mean([cur[j][k] for j in cur])) for k in cur[1]}
    agg.update(
        cur_G_U=cagg["U_rep"] - cagg["U_seq"],
        cur_G_V=cagg["V_rep"] - cagg["V_seq"],
        cur_U_seq=cagg["U_seq"],
        cur_V_seq=cagg["V_seq"],
    )
    pairs[s] = dict(same_start=same_start, per_period={j: per[j] for j in per}, current={j: cur[j] for j in cur}, **agg)
    print(
        f"seed {s} same start {same_start}: U_seq {agg['U_seq']:.4f} U_rep {agg['U_rep']:.4f} V_seq "
        f"{agg['V_seq']:.4f} V_rep {agg['V_rep']:.4f} "
        f"G_U {agg['G_U']:+.4f} G_V {agg['G_V']:+.4f} Delta {agg['Delta']:+.4f} G500 {agg['G500']:+.4f} G200 "
        f"{agg['G200']:+.4f} "
        f"cur G_U {agg['cur_G_U']:+.4f} G_V {agg['cur_G_V']:+.4f}",
        flush=True,
    )

keys = [
    "G_U",
    "G_V",
    "Delta",
    "R_seq",
    "R_rep",
    "G500",
    "G200",
    "cur_G_U",
    "cur_G_V",
    "U_seq",
    "U_rep",
    "V_seq",
    "V_rep",
]
X = {k: np.array([pairs[s][k] for s in seeds]) for k in keys}
rng = np.random.default_rng(20260920); n = len(seeds)
idx = rng.integers(0, n, (BOOT, n))
summary = {}
for k in keys:
    b = X[k][idx].mean(1)
    summary[k] = dict(mean=float(X[k].mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                      n_positive=int((X[k] > 0).sum()), sd=float(X[k].std(ddof=1)) if n > 1 else None)
d = X["Delta"]; obs = abs(d.mean())
flips = np.array([abs((d * np.array(sg)).mean()) for sg in product([1, -1], repeat=n)])
summary["Delta"]["signflip_p"] = float((flips >= obs - 1e-12).mean())
gv = summary["G_V"]["ci"]
summary["G_V"]["margin"] = MARGIN
summary["G_V"]["outcome"] = (
    "advantage retained" if gv[0] > MARGIN else "no material advantage" if gv[1] < MARGIN else "unresolved"
)
st = pd.read_parquet(f"streams/stream_{STREAM}.parquet")
pool = int((st.split == "train_pool").sum()); test = st[st.split == "test"].groupby("period").size()
resources = dict(train_pool_total=pool, replay_buffer_records=int(round(0.05 * pool)),
                 split_half_calibration_per_period=int(test.mean() // 2), finite_update_records=list(NFIN))
out = dict(
    stream=STREAM,
    n_pairs=n,
    seeds=seeds,
    all_same_start=all(pairs[s]["same_start"] for s in seeds),
    M=M,
    boot=BOOT,
    draws=DRAWS,
    summary=summary,
    resources=resources,
    pairs={str(s): pairs[s] for s in seeds},
)
os.makedirs("results/paired_refit", exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
print(f"\n{n} pairs {seeds}; same period-1 scores and thresholds in every pair: {out['all_same_start']}")
for k in keys:
    s_ = summary[k]
    print(f"{k:8s} {s_['mean']:+.4f} [{s_['ci'][0]:+.4f}, {s_['ci'][1]:+.4f}]  positive {s_['n_positive']}/{n}")
print(
    f"Delta sign-flip p {summary['Delta']['signflip_p']:.4f}; G_V against margin {MARGIN}: {summary['G_V']['outcome']}"
)
print("resources", resources)
