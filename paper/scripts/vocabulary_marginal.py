#!/usr/bin/env python3
"""Jensen-Shannon divergence (bits) of the unigram distribution of each vocabulary stream from
the unmodified stream, per period, with the control's adjacent-period divergence as yardstick.
Reads streams/stream_{primary,asrsdrift50,asrsperm100}.parquet; writes results/asrs_supplementary/vocab_marginal.json.
Backs the token-marginal sentence after Table E.1. Needs the stream files (not shipped)."""
import json, numpy as np, pandas as pd
from collections import Counter
def dist(texts):
    c = Counter(w for t in texts for w in t.split()); n = sum(c.values()); return c, n
def jsd(c1, n1, c2, n2):
    keys = set(c1) | set(c2)
    p = np.array([c1.get(k, 0) / n1 for k in keys])
    q = np.array([c2.get(k, 0) / n2 for k in keys])
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(a / b), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
ctrl = pd.read_parquet("streams/stream_primary.parquet"); out = {}
cd = {p: dist(ctrl[ctrl.period == p].text) for p in sorted(ctrl.period.unique())}
out["control_adjacent_period_jsd"] = {int(p): jsd(*cd[p], *cd[p + 1]) for p in range(ctrl.period.max())}
for name in ["asrsdrift50", "asrsperm100"]:
    st = pd.read_parquet(f"streams/stream_{name}.parquet")
    per = {}
    for p in sorted(st.period.unique()):
        c, n = dist(st[st.period == p].text); per[int(p)] = jsd(c, n, *cd[p])
    out[name] = per; print(name, {k: round(v, 4) for k, v in per.items()})
print("control adjacent", {k: round(v, 4) for k, v in out["control_adjacent_period_jsd"].items()})
out["mapping"] = dict(
    pairing="involution on the paired token lists (each mapped token swaps with exactly one partner, "
    "applied per occurrence with probability rho)",
    permutation="a new bijection of the pooled vocabulary per period, applied to every occurrence",
)
json.dump(out, open("results/asrs_supplementary/vocab_marginal.json", "w"), indent=1)
