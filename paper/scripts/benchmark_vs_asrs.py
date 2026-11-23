#!/usr/bin/env python3
"""Each benchmark stream's dR and A minus the ASRS values on the restricted range, resampling
the benchmark's and ASRS's initialisations independently.
Reads results/benchmarks/external_bridge.parquet and results/asrs_inherited_rule/bwt_bridge.parquet;
writes results/asrs_supplementary/benchmark_vs_asrs.json. Not reported in the current supplement.
Needs nothing beyond the repository."""
import json, numpy as np, pandas as pd
b = pd.read_parquet("results/benchmarks/external_bridge.parquet"); b = b[b.j >= 1]
a = pd.read_parquet("results/asrs_inherited_rule/bwt_bridge.parquet"); a = a[(a.j >= 1) & (a.j <= 6)]
A_blk = a.groupby("seed")[["dR_e1", "A"]].mean().to_numpy(); rng = np.random.default_rng(20260920); out = {}
for s, d in b.groupby("stream"):
    X = d.groupby("seed")[["dR", "A"]].mean().to_numpy(); n, m = len(X), len(A_blk)
    diff = np.array([X[rng.integers(0, n, n)].mean(0) - A_blk[rng.integers(0, m, m)].mean(0) for _ in range(20000)])
    out[s] = dict(
        dR_minus_asrs=float(X[:, 0].mean() - A_blk[:, 0].mean()),
        ci=[float(np.percentile(diff[:, 0], 2.5)), float(np.percentile(diff[:, 0], 97.5))],
        A_minus_asrs=float(X[:, 1].mean() - A_blk[:, 1].mean()),
        ci_A=[float(np.percentile(diff[:, 1], 2.5)), float(np.percentile(diff[:, 1], 97.5))],
    )
    print(
        f"{s:10s} dR - ASRS {out[s]['dR_minus_asrs']:+.4f} [{out[s]['ci'][0]:+.4f}, {out[s]['ci'][1]:+.4f}]   A - "
        f"ASRS {out[s]['A_minus_asrs']:+.4f} [{out[s]['ci_A'][0]:+.4f}, {out[s]['ci_A'][1]:+.4f}]"
    )
json.dump(out, open("results/asrs_supplementary/benchmark_vs_asrs.json", "w"), indent=1)
