#!/usr/bin/env python3
"""MultiEURLEX BiGRU: B, A and dR per historical period with bootstrap intervals over the ten
initialisations, and the share Delta / G_U of replay's lead that is threshold placement.
Reads results/multieurlex_bigru/eurlex_bridge.parquet and results/paired_refit/paired_replay_eurlex.json;
writes results/multieurlex_bigru/eurlex_per_period_ci.json. Supports Table D.1 and Table C.1 (MultiEURLEX).
Needs nothing beyond the repository."""
import json
import numpy as np, pandas as pd

BOOT, BSEED = 20000, 20260920
t = pd.read_parquet("results/multieurlex_bigru/eurlex_bridge.parquet")
pairs = json.load(open("results/paired_refit/paired_replay_eurlex.json"))["pairs"]


def ci(x, fn=np.mean):
    x = np.asarray(x); rng = np.random.default_rng(BSEED); n = len(x)
    d = np.array([fn(x[rng.integers(0, n, n)]) for _ in range(BOOT)])
    return [round(float(fn(x)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


out = {"per_period": {}}
for j in range(1, 7):
    s = t[t.j == j].sort_values("seed")
    out["per_period"][str(j)] = {q: ci(s[q].to_numpy()) for q in ("B", "A", "dR")}
    print(
        f"j={j}  B {out['per_period'][str(j)]['B']}  A {out['per_period'][str(j)]['A']}  dR "
        f"{out['per_period'][str(j)]['dR']}"
    )
X = np.array([[pairs[k]["Delta"], pairs[k]["G_U"]] for k in sorted(pairs, key=int)])
rng = np.random.default_rng(BSEED); n = len(X)
d = np.array([X[idx, 0].mean() / X[idx, 1].mean() for idx in (rng.integers(0, n, n) for _ in range(BOOT))])
out["share_delta_over_GU"] = [
    float(X[:, 0].mean() / X[:, 1].mean()),
    float(np.percentile(d, 2.5)),
    float(np.percentile(d, 97.5)),
]
print("share Delta/G_U", [round(v, 3) for v in out["share_delta_over_GU"]])
json.dump(out, open("results/multieurlex_bigru/eurlex_per_period_ci.json", "w"), indent=1)
