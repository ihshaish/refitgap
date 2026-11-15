#!/usr/bin/env python3
"""Build the reweighted streams: label priors tilted from period 3 on, text and labels untouched.

Reads streams/stream_primary.parquet; writes streams/stream_asrsprior{00,40,100}.parquet (00 is the
size-matched uniform control). Builds the reweighting streams of the constructed drifts (Table E.1,
Appendix E). Needs the primary stream, built from the ASRS export (not shipped)."""
import numpy as np
import pandas as pd

# Keeping a fraction KEEP of records can concentrate a label by at most 1/KEEP;
# 0.35 is the strongest shift that keeps every label above 75 test positives.
SEED, PSTAR, KEEP, TILT = 20260802, 3, 0.35, 5.0
st = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in st.labels for l in s.split(";")})
Y = np.array([[l in s.split(";") for l in labels] for s in st.labels])

# Direction from period-0 prevalences only: towards the rarer half of the labels.
p0 = Y[(st.period == 0).values].mean(0)
direction = np.where(p0 <= np.median(p0), 1.0, -1.0)
print("label prevalences at period 0, and shift direction:")
for l, p, d in zip(labels, p0, direction):
    print(f"  {l[:34]:34s} {p:.3f}  {'up' if d > 0 else 'down'}")


def draw(idx, n, rho, rng):
    """Take n of idx without replacement: uniformly, or tilted towards the up-labels."""
    if rho == 0:
        return rng.choice(idx, size=n, replace=False)
    w = np.exp(rho * TILT * (Y[idx] @ direction))
    return rng.choice(idx, size=n, replace=False, p=w / w.sum())


def build(rho, tag):
    rng = np.random.default_rng(SEED + int(rho * 100))
    keep_idx = []
    for period in sorted(st.period.unique()):
        # Disjoint halves before tilting, so train and test both carry the shift.
        allp = np.flatnonzero((st.period == period).values)
        shuf = rng.permutation(allp)
        half = len(shuf) // 2
        A, B = shuf[:half], shuf[half:]
        r = 0.0 if period < PSTAR else rho
        n = int(round(KEEP * half))
        train = draw(A, n, r, rng)
        test = draw(B, n, r, rng)
        tr_df = st.iloc[train].copy(); tr_df["split"] = "train_pool"
        te_df = st.iloc[test].copy(); te_df["split"] = "test"
        keep_idx += [tr_df, te_df]
    out = pd.concat(keep_idx, ignore_index=True).sort_values(
        ["period", "yyyymm"]).reset_index(drop=True)
    out.to_parquet(f"streams/stream_{tag}.parquet", index=False)
    tr = out[out.split == "train_pool"]
    Yt = np.array([[l in s.split(";") for l in labels] for s in tr.labels])
    pv = np.array([Yt[(tr.period == p).values].mean(0)
                   for p in sorted(tr.period.unique())])
    drift = np.abs(pv[-1] - pv[0]).mean()
    print(f"  {tag:20s} n_train={len(tr):6d}  mean |prior(final) - prior(first)| = {drift:.4f}")
    return drift


print(f"\nbuilding (subsample rate {KEEP}, shift from period {PSTAR}):")
for rho, tag in [(0.0, "asrsprior00"), (0.2, "asrsprior40"), (1.0, "asrsprior100")]:
    build(rho, tag)
