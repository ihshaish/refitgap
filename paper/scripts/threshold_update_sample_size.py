#!/usr/bin/env python3
"""Threshold update with n = 50, 100, 200, 500 labelled pool records (20 draws each), against theta_1 and the
in-sample refit, for the final model of every condition and seed with a usable checkpoint.
Writes results/calib_experiment_v2.parquet. Backs Appendix D, "Threshold update with few labels".
Needs the stream files and checkpoints (not shipped); trainable conditions also need the stored embeddings."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

DRAWS = 20
NS = (50, 100, 200, 500)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in df.labels for l in s.split(";")})
lab_idx = {l: i for i, l in enumerate(labels)}
Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
for r, s in enumerate(df.labels):
    for l in s.split(";"):
        Yall[r, lab_idx[l]] = 1
N = int(df.period.max()) + 1

VEC_CACHE = {}
def vectors(kind):
    if kind not in VEC_CACHE:
        VEC_CACHE[kind] = load_vectors(kind)
    return VEC_CACHE[kind]

def build(condition, seed):
    """Rebuild the vocabulary (replaying its growth for trainable conditions) and the final model."""
    frozen = condition in ("C1", "C2")
    kind = {"C1": "domain", "C2": "glove", "C3": "domain", "C4": "glove"}[condition]
    vecs = vectors(kind)
    rng = np.random.default_rng(seed)
    tokens = list(vecs)
    mat = np.vstack([np.zeros(KNOBS["emb_dim"]), rng.normal(0, .1, KNOBS["emb_dim"]),
                     np.stack([vecs[t] for t in tokens])]).astype(np.float32)
    vocab = Vocab(tokens, grow=not frozen)
    p1 = df[(df.period == 0) & (df.split == "train_pool")]
    val_n = int(len(p1) * KNOBS["val_frac_p1"])
    val_sel = np.random.default_rng(20260802).choice(p1.index.to_numpy(), val_n, False)
    tr_sel0 = np.setdiff1d(p1.index.to_numpy(), val_sel)
    if not frozen:  # replay growth to match the stored embedding rows
        for i in range(N):
            tri = df[(df.period == i) & (df.split == "train_pool")]
            sel = tri.index.to_numpy() if i else tr_sel0
            for idx in sel:
                vocab.encode(df.text[idx], train_mode=True)
    ckpt = Path(f"checkpoints/primary_{condition}_s{seed}_v2/period{N-1}.pt")
    if not ckpt.exists():
        return None
    sd = torch.load(ckpt, map_location="cpu")
    if frozen:
        model = BiGRU(torch.tensor(mat), freeze=True, n_labels=len(labels)).to(device)
        model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    else:
        if "emb.weight" not in sd:
            return None
        if sd["emb.weight"].shape[0] != len(vocab.idx) + 2:
            print(f"  vocab replay mismatch {condition} s{seed}; skip"); return None
        model = BiGRU(sd["emb.weight"].float(), freeze=False, n_labels=len(labels)).to(device)
        model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    return model, vocab, val_sel

rows = []
for condition in ("C1", "C2", "C3", "C4"):
    for seed in range(10):
        built = build(condition, seed)
        if built is None:
            continue
        model, vocab, val_sel = built
        vcodes = [vocab.encode(df.text[i], False) for i in val_sel]
        _, _, Pv, _, _ = evaluate(model, vcodes, Yall[val_sel],
                                  np.full(len(labels), .5), device)
        th_frozen = fit_thresholds(Pv, Yall[val_sel])
        for j in range(N):
            te = df[(df.period == j) & (df.split == "test")]
            tcodes = [vocab.encode(t, False) for t in te.text]
            Yt = Yall[te.index.to_numpy()]
            _, _, Pt, _, _ = evaluate(model, tcodes, Yt,
                                      np.full(len(labels), .5), device)
            pool = df[(df.period == j) & (df.split == "train_pool")].index.to_numpy()
            pcodes = [vocab.encode(df.text[i], False) for i in pool]
            _, _, Pp, _, _ = evaluate(model, pcodes, Yall[pool],
                                      np.full(len(labels), .5), device)
            f1_frozen = f1_score(Yt, (Pt >= th_frozen).astype(int),
                                 average="macro", zero_division=0)
            th_or = fit_thresholds(Pt, Yt)
            f1_oracle = f1_score(Yt, (Pt >= th_or).astype(int),
                                 average="macro", zero_division=0)
            rec = dict(condition=condition, seed=seed, j=j,
                       f1_frozen=f1_frozen, f1_oracle=f1_oracle)
            for n in NS:
                vals = []
                for d in range(DRAWS):
                    srng = np.random.default_rng(100000 + seed*1000 + j*40 + d)
                    sel = srng.choice(len(pool), min(n, len(pool)), replace=False)
                    th = fit_thresholds(Pp[sel], Yall[pool[sel]])
                    vals.append(f1_score(Yt, (Pt >= th).astype(int),
                                         average="macro", zero_division=0))
                rec[f"recal_{n}_mean"] = float(np.mean(vals))
                rec[f"recal_{n}_sd"] = float(np.std(vals))
            rows.append(rec)
        print(f"{condition} s{seed} done ({len(rows)} rows)", flush=True)
        pd.DataFrame(rows).to_parquet("results/calib_experiment_v2.parquet", index=False)

out = pd.DataFrame(rows)
m = out.groupby("j")[["f1_frozen"] + [f"recal_{n}_mean" for n in NS] + ["f1_oracle"]].mean()
print(m.round(4).to_string())
print("runs:", out.groupby("condition").seed.nunique().to_dict())
