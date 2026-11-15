#!/usr/bin/env python3
"""Threshold update with lagged labels: macro-F1 of the final model on period j with thresholds fitted on n = 200 or
500 pool records of period j (same) or j-1 (lag), against theta_1, frozen conditions, ten seeds.
Writes results/calib_lagged.parquet. Backs Appendix D, "Threshold update with few labels" (one-period lag).
Needs the stream files and checkpoints (not shipped)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

NS = (200, 500)
DRAWS = 20
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in df.labels for l in s.split(";")})
lab_idx = {l: i for i, l in enumerate(labels)}
Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
for r, s in enumerate(df.labels):
    for l in s.split(";"):
        Yall[r, lab_idx[l]] = 1
N = int(df.period.max()) + 1

rows = []
for cond, kind in [("C1", "domain"), ("C2", "glove")]:
    vecs = load_vectors(kind)
    for seed in range(10):
        ck = Path(f"checkpoints/primary_{cond}_s{seed}_v2/period{N-1}.pt")
        if not ck.exists():
            continue
        rng = np.random.default_rng(seed)
        toks = list(vecs)
        mat = np.vstack([np.zeros(KNOBS["emb_dim"]),
                         rng.normal(0, .1, KNOBS["emb_dim"]),
                         np.stack([vecs[t] for t in toks])]).astype(np.float32)
        vocab = Vocab(toks, grow=False)
        model = BiGRU(torch.tensor(mat), freeze=True, n_labels=len(labels)).to(device)
        model.load_state_dict({k: v.float() for k, v in
                               torch.load(ck, map_location="cpu").items()}, strict=False)
        model.eval()

        p1 = df[(df.period == 0) & (df.split == "train_pool")]
        vs = np.random.default_rng(20260802).choice(
            p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
        _, _, Pv, _, _ = evaluate(model, [vocab.encode(df.text[i], False) for i in vs],
                                  Yall[vs], np.full(len(labels), .5), device)
        th_frozen = fit_thresholds(Pv, Yall[vs])

        # cache per-period probabilities on the train pool (source of thresholds)
        # and on the test split (where they are scored)
        Ppool, Ytest, Ptest = {}, {}, {}
        for j in range(N):
            pool = df[(df.period == j) & (df.split == "train_pool")].index.to_numpy()
            _, _, Pp, _, _ = evaluate(model, [vocab.encode(df.text[i], False) for i in pool],
                                      Yall[pool], np.full(len(labels), .5), device)
            Ppool[j] = (Pp, Yall[pool])
            te = df[(df.period == j) & (df.split == "test")]
            Yt = Yall[te.index.to_numpy()]
            _, _, Pt, _, _ = evaluate(model, [vocab.encode(t, False) for t in te.text],
                                      Yt, np.full(len(labels), .5), device)
            Ptest[j], Ytest[j] = Pt, Yt

        for j in range(1, N):          # j=0 has no previous period
            Pt, Yt = Ptest[j], Ytest[j]
            rec = dict(condition=cond, seed=seed, j=j,
                       f1_frozen=f1_score(Yt, (Pt >= th_frozen).astype(int),
                                          average="macro", zero_division=0))
            drng = np.random.default_rng(9000 + 100 * seed + j)
            for n in NS:
                for tag, src in [("same", j), ("lag", j - 1)]:
                    Pp, Yp = Ppool[src]
                    if len(Yp) < n:
                        continue
                    vals = []
                    for _ in range(DRAWS):
                        take = drng.choice(len(Yp), n, replace=False)
                        th = fit_thresholds(Pp[take], Yp[take])
                        vals.append(f1_score(Yt, (Pt >= th).astype(int),
                                             average="macro", zero_division=0))
                    rec[f"{tag}_{n}"] = float(np.mean(vals))
            rows.append(rec)
        print(f"  {cond} s{seed}: done", flush=True)

t = pd.DataFrame(rows)
t.to_parquet("results/calib_lagged.parquet", index=False)
g = t[[c for c in t.columns if c.startswith(("f1_", "same_", "lag_"))]].mean()
print("\nmean macro-F1 over periods 1-7, ten seeds, two frozen conditions")
print(g.round(4).to_string())
for n in NS:
    if f"same_{n}" in g and f"lag_{n}" in g:
        gain_same = g[f"same_{n}"] - g.f1_frozen
        gain_lag = g[f"lag_{n}"] - g.f1_frozen
        print(f"  n={n}: same-period gain {gain_same:+.4f}, lagged gain {gain_lag:+.4f}, "
              f"lag retains {100*gain_lag/gain_same:.0f}% of it")
