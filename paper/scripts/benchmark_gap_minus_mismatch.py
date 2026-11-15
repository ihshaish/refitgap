#!/usr/bin/env python3
"""Each benchmark stream's contrast A - dR on the restricted range (periods 2 to 7), with a
percentile bootstrap over initialisations, and whether its dR interval lies below the ASRS 0.0113.
Reads results/benchmarks/external_bridge.parquet; writes
results/asrs_supplementary/benchmark_AminusdR.json. Backs the last column of Table D.2.
Needs nothing beyond the repository."""
import json, numpy as np, pandas as pd
t = pd.read_parquet("results/benchmarks/external_bridge.parquet"); t = t[t.j >= 1]
rng = np.random.default_rng(20260920); out = {}
for s, d in t.groupby("stream"):
    blk = d.groupby("seed")[["A", "dR", "B"]].mean(); n = len(blk); X = blk.to_numpy()
    draws = np.array([X[rng.integers(0, n, n)].mean(0) for _ in range(20000)])
    diff = draws[:, 0] - draws[:, 1]
    out[s] = dict(n=n, A=float(X[:, 0].mean()), dR=float(X[:, 1].mean()), A_minus_dR=float((X[:, 0] - X[:, 1]).mean()),
                  ci_A_minus_dR=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                  ci_dR=[float(np.percentile(draws[:, 1], 2.5)), float(np.percentile(draws[:, 1], 97.5))],
                  dR_ci_excludes_asrs_0p0113=bool(np.percentile(draws[:, 1], 97.5) < 0.0113))
    print(
        f"{s:10s} A-dR {out[s]['A_minus_dR']:+.4f} [{out[s]['ci_A_minus_dR'][0]:+.4f}, "
        f"{out[s]['ci_A_minus_dR'][1]:+.4f}]  dR CI [{out[s]['ci_dR'][0]:+.4f}, {out[s]['ci_dR'][1]:+.4f}]  excludes "
        f"0.0113: {out[s]['dR_ci_excludes_asrs_0p0113']}"
    )
json.dump(out, open("results/asrs_supplementary/benchmark_AminusdR.json", "w"), indent=1)
