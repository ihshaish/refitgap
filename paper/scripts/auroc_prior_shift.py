#!/usr/bin/env python3
"""Macro-AUROC: backward transfer on ASRS (frozen conditions, ten seeds) and the final model's level
on the prior-reweighting and permutation streams, where AUROC is insensitive to the label prior.
Writes results/auroc.parquet and results/auroc_priorshift.csv.
Backs the AUROC sentences of Appendix D (frozen conditions) and Appendix E, "Both drifts combined" (prior doses).
Needs the stream files and checkpoints (not shipped)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, load_vectors

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def macro_auroc(P, Y):
    """Mean over labels that have both classes present in this test set."""
    vals = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        if 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, P[:, k]))
    return float(np.mean(vals)) if vals else np.nan


def load_stream(name):
    df = pd.read_parquet(f"streams/stream_{name}.parquet")
    labels = sorted({l for s in df.labels for l in s.split(";")})
    idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(df), len(labels)), dtype=np.int64)
    for r, s in enumerate(df.labels):
        for l in s.split(";"):
            Y[r, idx[l]] = 1
    return df, labels, Y


def build(kind, seed, n_labels):
    vecs = load_vectors(kind)
    toks = list(vecs)
    rng = np.random.default_rng(seed)
    mat = np.vstack([np.zeros(KNOBS["emb_dim"]),
                     rng.normal(0, .1, KNOBS["emb_dim"]),
                     np.stack([vecs[t] for t in toks])]).astype(np.float32)
    return mat, Vocab(toks, grow=False)


def model_from(mat, path, n_labels):
    m = BiGRU(torch.tensor(mat), freeze=True, n_labels=n_labels).to(device)
    m.load_state_dict({k: v.float() for k, v in
                       torch.load(path, map_location="cpu").items()}, strict=False)
    m.eval()
    return m


# AUROC backward transfer on the primary stream
df, labels, Yall = load_stream("primary")
N = int(df.period.max()) + 1
tests = {}
for j in range(N):
    te = df[(df.period == j) & (df.split == "test")]
    tests[j] = (te.text.tolist(), Yall[te.index.to_numpy()])

rows = []
for cond, kind in [("C1", "domain"), ("C2", "glove")]:
    mat, vocab = build(kind, 0, len(labels))
    codes = {j: [vocab.encode(t, False) for t in tests[j][0]] for j in range(N)}
    for seed in range(10):
        d = Path(f"checkpoints/primary_{cond}_s{seed}_v2")
        if not all((d / f"period{i}.pt").exists() for i in range(N)):
            continue
        mat_s, vocab_s = build(kind, seed, len(labels))
        codes_s = {j: [vocab_s.encode(t, False) for t in tests[j][0]] for j in range(N)}
        for i in range(N):
            m = model_from(mat_s, d / f"period{i}.pt", len(labels))
            for j in range(N):
                Yt = tests[j][1]
                _, _, P, _, _ = evaluate(m, codes_s[j], Yt,
                                         np.full(len(labels), .5), device)
                rows.append(dict(condition=cond, seed=seed, i=i, j=j,
                                 auroc=macro_auroc(P, Yt)))
        print(f"  primary {cond} s{seed}: done", flush=True)

t = pd.DataFrame(rows)
t.to_parquet("results/auroc.parquet", index=False)
bw = []
for (c, s), g in t.groupby(["condition", "seed"]):
    R = np.full((N, N), np.nan)
    for _, r in g.iterrows():
        R[int(r.i), int(r.j)] = r.auroc
    if not np.isnan(R).any():
        bw.append(np.mean([R[N-1, j] - R[j, j] for j in range(N-1)]))
print(f"\nmacro-AUROC backward transfer, primary: {np.mean(bw):+.4f} "
      f"(sd {np.std(bw):.4f}, n={len(bw)} condition-seed blocks)")

# final-model AUROC across the prior-reweighting doses (only final checkpoints exist for these)
out = []
for arm in ["asrsprior00", "asrsprior40", "asrsprior100", "asrsperm100"]:
    dfa, laba, Ya = load_stream(arm)
    Na = int(dfa.period.max()) + 1
    mat, vocab = build("glove", 0, len(laba))
    for seed in range(5):
        ck = Path(f"checkpoints/{arm}_C2_s{seed}_v2/period{Na-1}.pt")
        if not ck.exists():
            continue
        mat_s, vocab_s = build("glove", seed, len(laba))
        m = model_from(mat_s, ck, len(laba))
        aus, aps = [], []
        for j in range(Na):
            te = dfa[(dfa.period == j) & (dfa.split == "test")]
            Yt = Ya[te.index.to_numpy()]
            _, ap, P, _, _ = evaluate(m, [vocab_s.encode(x, False) for x in te.text],
                                      Yt, np.full(len(laba), .5), device)
            aus.append(macro_auroc(P, Yt)); aps.append(ap)
        out.append(dict(arm=arm, seed=seed, auroc=np.mean(aus), auprc=np.mean(aps)))
    print(f"  {arm}: done", flush=True)

o = pd.DataFrame(out)
o.to_csv("results/auroc_priorshift.csv", index=False)
print("\nfinal-model means across arms (level comparison, not backward transfer)")
print(o.groupby("arm")[["auroc", "auprc"]].agg(["mean", "std"]).round(4).to_string())
