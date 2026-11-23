#!/usr/bin/env python3
"""Rescore the frozen-embedding runs of the primary stream (C1, C2, seeds 0-9) from their period checkpoints.

Reads streams/stream_primary.parquet and checkpoints/primary_{cond}_s{seed}_v2/; writes P_i_j, Y_j, Pval_i, Yval,
theta_dep and theta_final to results/asrs_supplementary/scores/primary_{cond}_s{seed}.npz. Trains nothing.
Backs the ASRS decomposition (Tables D.2 to D.4, Figure 4). Needs the checkpoints (not distributed), the
embeddings and the primary stream. Usage: python scripts/store_asrs_scores.py [seeds] [conditions]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(10))
CONDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["C1", "C2"]
OUT = Path("results/asrs_supplementary/scores"); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in df.labels for l in s.split(";")}); lab_idx = {l: i for i, l in enumerate(labels)}
Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
for r, s in enumerate(df.labels):
    for l in s.split(";"): Yall[r, lab_idx[l]] = 1
N = int(df.period.max()) + 1; K = len(labels)
def load_ckpt(path, base_mat):
    sd = torch.load(path, map_location="cpu")
    emb_key = next((k for k in sd if k.endswith("emb.weight")), None)
    mat = sd[emb_key].float().numpy() if emb_key is not None else base_mat
    m = BiGRU(torch.tensor(mat), freeze=True, n_labels=K).to(device)
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); m.eval(); return m
p1 = df[(df.period == 0) & (df.split == "train_pool")]
vs = np.random.default_rng(20260802).choice(p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
tests = {j: df[(df.period == j) & (df.split == "test")] for j in range(N)}
t0 = time.time()
for cond in CONDS:
    kind = {"C1": "domain", "C2": "glove"}[cond]
    vecs = load_vectors(kind); toks = list(vecs); vocab = Vocab(toks, grow=False)
    vcodes = [vocab.encode(df.text[i], False) for i in vs]
    tcodes = {j: [vocab.encode(t, False) for t in tests[j].text] for j in range(N)}
    tY = {j: Yall[tests[j].index.to_numpy()] for j in range(N)}
    for seed in SEEDS:
        out = OUT / f"primary_{cond}_s{seed}.npz"
        if out.exists(): print("skip", out); continue
        d = Path(f"checkpoints/primary_{cond}_s{seed}_v2")
        assert all((d / f"period{i}.pt").exists() for i in range(N)), d
        rng = np.random.default_rng(seed)
        base = np.vstack(
            [np.zeros(KNOBS["emb_dim"]), rng.normal(0, 0.1, KNOBS["emb_dim"]), np.stack([vecs[t] for t in toks])]
        ).astype(np.float32)
        z = dict(labels=np.array(labels), Yval=Yall[vs].astype(np.int8))
        for i in range(N):
            m = load_ckpt(d / f"period{i}.pt", base)
            _, _, Pv, _, _ = evaluate(m, vcodes, Yall[vs], np.full(K, 0.5), device)
            z[f"Pval_{i}"] = Pv.astype(np.float32)
            for j in range(N):
                _, _, P, _, _ = evaluate(m, tcodes[j], tY[j], np.full(K, 0.5), device)
                z[f"P_{i}_{j}"] = P.astype(np.float32)
        for j in range(N): z[f"Y_{j}"] = tY[j].astype(np.int8)
        z["theta_dep"] = fit_thresholds(z["Pval_0"], Yall[vs]).astype(np.float32)
        z["theta_final"] = fit_thresholds(z[f"Pval_{N-1}"], Yall[vs]).astype(np.float32)
        np.savez_compressed(out, **z); print(f"{cond} s{seed} stored ({time.time()-t0:.0f}s)", flush=True)
print("done")
