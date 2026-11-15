#!/usr/bin/env python3
"""Attainable gap with trainable embeddings (C3, C4, seeds 1 and 2): final against contemporary model, both refitted by
split-half (grid fitter), and the final model at theta_1, periods 1 to 7.
Writes results/oracle_trainable.parquet. Backs the four trainable-embedding observations of Appendix D.
Needs the stream files and checkpoints with their stored embedding tables (not shipped)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

SPLITS = 20
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in df.labels for l in s.split(";")})
lab_idx = {l: i for i, l in enumerate(labels)}
Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
for r, s in enumerate(df.labels):
    for l in s.split(";"):
        Yall[r, lab_idx[l]] = 1
N = int(df.period.max()) + 1


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def split_half_oracle(P, Y, rng):
    vals = []
    for _ in range(SPLITS):
        perm = rng.permutation(len(Y)); h = len(Y) // 2
        A, B = perm[:h], perm[h:]
        vals.append(0.5 * (mf1(P[B], Y[B], fit_thresholds(P[A], Y[A])) +
                           mf1(P[A], Y[A], fit_thresholds(P[B], Y[B]))))
    return float(np.mean(vals))


def load_ckpt(path, base_mat, n_labels):
    """Load a checkpoint, taking the embedding table from it when stored (trainable conditions grow it)."""
    sd = torch.load(path, map_location="cpu")
    emb_key = next((k for k in sd if k.endswith("emb.weight")), None)
    mat = sd[emb_key].float().numpy() if emb_key is not None else base_mat
    m = BiGRU(torch.tensor(mat), freeze=True, n_labels=n_labels).to(device)
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    m.eval()
    return m, mat.shape[0]


rows = []
for cond, kind in [("C3", "domain"), ("C4", "glove")]:
    vecs = load_vectors(kind)
    toks = list(vecs)
    for seed in (1, 2):
        d = Path(f"checkpoints/primary_{cond}_s{seed}_v2")
        if not all((d / f"period{i}.pt").exists() for i in range(N)):
            print(f"  {cond} s{seed}: incomplete, skipped", flush=True)
            continue
        rng = np.random.default_rng(seed)
        base = np.vstack([np.zeros(KNOBS["emb_dim"]),
                          rng.normal(0, .1, KNOBS["emb_dim"]),
                          np.stack([vecs[t] for t in toks])]).astype(np.float32)
        # base tokens keep their indices; rows beyond the base table were added in training
        vocab = Vocab(toks, grow=False)

        final, n_final = load_ckpt(d / f"period{N-1}.pt", base, len(labels))
        p1 = df[(df.period == 0) & (df.split == "train_pool")]
        vs = np.random.default_rng(20260802).choice(
            p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
        _, _, Pv, _, _ = evaluate(final, [vocab.encode(df.text[i], False) for i in vs],
                                  Yall[vs], np.full(len(labels), .5), device)
        th_frozen = fit_thresholds(Pv, Yall[vs])

        for j in range(N - 1):
            te = df[(df.period == j) & (df.split == "test")]
            codes = [vocab.encode(t, False) for t in te.text]
            Yt = Yall[te.index.to_numpy()]
            _, _, Pf, _, _ = evaluate(final, codes, Yt, np.full(len(labels), .5), device)
            contemp, _ = load_ckpt(d / f"period{j}.pt", base, len(labels))
            _, _, Pc, _, _ = evaluate(contemp, codes, Yt, np.full(len(labels), .5), device)
            srng = np.random.default_rng(5000 + 100 * seed + j)
            rows.append(dict(
                condition=cond, seed=seed, j=j,
                f1_final_frozen=mf1(Pf, Yt, th_frozen),
                f1_final_oracle=split_half_oracle(Pf, Yt, srng),
                f1_contemp_oracle=split_half_oracle(
                    Pc, Yt, np.random.default_rng(5000 + 100 * seed + j))))
        print(f"  {cond} s{seed}: done (embedding rows {n_final})", flush=True)

t = pd.DataFrame(rows)
t.to_parquet("results/oracle_trainable.parquet", index=False)

sub = t[t.j > 0]                      # period 1 excluded, as in the frozen analysis
blocks = sub.groupby(["condition", "seed"]).apply(
    lambda x: (x.f1_final_oracle - x.f1_contemp_oracle).mean())
print("\nfinal minus contemporary, both threshold-optimised, periods 2-7")
print(f"  trainable embeddings: {blocks.mean():+.4f}  "
      f"(per block: {', '.join(f'{v:+.4f}' for v in blocks)})")
print(f"  final model below its own optimum at inherited thresholds: "
      f"{(sub.f1_final_frozen - sub.f1_final_oracle).mean():+.4f}")
