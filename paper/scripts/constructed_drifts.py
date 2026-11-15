#!/usr/bin/env python3
"""Rescore the constructed-stream checkpoints with both threshold fitters (grid and exact) on the same
split-half partitions; theta_1 stays the grid fit. Defines fit_exact, used across the scripts.
Writes results/constructed_drift_rescore.parquet (merged by stream) and results/constructed_drift_cells.csv,
the per-cell comparison of the grid arm with results/calib_splits200*.parquet (exit status 2 if it differs).
Backs Table E.1 and Appendix A, "How thresholds are fitted". Needs the stream files and checkpoints (not shipped)."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import constructed_drift_levels as C
from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

SPLITS = 200
STORED = {
    "results/calib_splits200.parquet": ["asrsdrift50", "asrsperm100", "primary"],
    "results/calib_splits200_priors.parquet": ["asrsprior00", "asrsprior100"],
    "results/calib_splits200_prior40.parquet": ["asrsprior40"],
}


def fit_exact(P, Y):
    """Per-label F1 maximiser over all score cutpoints (float64, positive if score >= theta,
    the highest of tied maximisers)."""
    P = np.asarray(P, dtype=np.float64)
    out = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        s, y = P[:, k], Y[:, k].astype(bool)
        o = np.argsort(-s, kind="mergesort")
        ss, ys = s[o], y[o]
        npos = int(ys.sum())
        if npos == 0:
            out[k] = float(ss[0]) + 1e-9
            continue
        f1 = 2 * np.cumsum(ys) / (np.arange(1, len(ss) + 1) + npos)
        ok = np.ones(len(ss), bool)
        ok[:-1] = ss[:-1] > ss[1:]
        out[k] = float(ss[int(np.argmax(np.where(ok, f1, -1.0)))])
    return out


def run_stream(stream, vecs, rows):
    df = pd.read_parquet(f"streams/stream_{stream}.parquet")
    labels = sorted({l for s in df.labels for l in s.split(";")})
    li = {l: i for i, l in enumerate(labels)}
    Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
    for r, s in enumerate(df.labels):
        for l in s.split(";"):
            Yall[r, li[l]] = 1
    N = int(df.period.max()) + 1
    tokens = list(vecs)

    for seed in C.SEEDS:
        ckpt = Path(f"checkpoints/{stream}_C2_s{seed}_v2/period{N-1}.pt")
        if not ckpt.exists():
            continue
        rng = np.random.default_rng(seed)
        mat = np.vstack([np.zeros(KNOBS["emb_dim"]),
                         rng.normal(0, .1, KNOBS["emb_dim"]),
                         np.stack([vecs[t] for t in tokens])]).astype(np.float32)
        vocab = Vocab(tokens, grow=False)
        model = BiGRU(torch.tensor(mat), freeze=True,
                      n_labels=len(labels)).to(C.device)
        model.load_state_dict({k: v.float() for k, v in
                               torch.load(ckpt, map_location="cpu").items()},
                              strict=False)
        model.eval()

        # inherited thresholds: grid fit, as the runs used
        p1 = df[(df.period == 0) & (df.split == "train_pool")]
        val_sel = np.random.default_rng(20260802).choice(
            p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
        _, _, Pv, _, _ = evaluate(model, [vocab.encode(df.text[i], False)
                                          for i in val_sel],
                                  Yall[val_sel], np.full(len(labels), .5), C.device)
        th_frozen = fit_thresholds(Pv, Yall[val_sel])

        for j in range(N):
            te = df[(df.period == j) & (df.split == "test")]
            Yt = Yall[te.index.to_numpy()]
            _, _, Pt, _, _ = evaluate(model, [vocab.encode(t, False) for t in te.text],
                                      Yt, np.full(len(labels), .5), C.device)
            rec = dict(stream=stream, seed=seed, j=j,
                       f1_frozen=C.macro_f1(Pt, Yt, th_frozen))
            # the same partitions for both fitters
            for tag, fitter in (("grid", fit_thresholds), ("exact", fit_exact)):
                srng = np.random.default_rng(7000 + 100 * seed + j)
                vals = []
                for _ in range(SPLITS):
                    perm = srng.permutation(len(Yt))
                    h = len(Yt) // 2
                    A, B = perm[:h], perm[h:]
                    vals.append(0.5 * (C.macro_f1(Pt[B], Yt[B], fitter(Pt[A], Yt[A])) +
                                       C.macro_f1(Pt[A], Yt[A], fitter(Pt[B], Yt[B]))))
                rec[f"f1_oracle_splithalf_{tag}"] = float(np.mean(vals))
            rows.append(rec)
        print(f"    {stream} seed {seed} done", flush=True)


def check_reproduction(t):
    """Compare the grid arm with the stored split-half values per (stream, seed, period)."""
    pub = pd.concat([pd.read_parquet(f) for f in STORED], ignore_index=True)
    m = t.merge(pub[["stream", "seed", "j", "f1_frozen", "f1_oracle_splithalf"]],
                on=["stream", "seed", "j"], suffixes=("", "_pub"))
    if m.empty:
        print("Reproduction check failed: no overlapping (stream, seed, period) rows")
        return False, m
    m["d_frozen"] = m.f1_frozen - m.f1_frozen_pub
    m["d_oracle"] = m.f1_oracle_splithalf_grid - m.f1_oracle_splithalf
    print(f"\nReproduction check over {len(m)} (stream, seed, period) cells")
    for c, nm in (("d_frozen", "f1_frozen"),
                  ("d_oracle", "f1_oracle_splithalf (grid arm)")):
        print(f"  {nm:34s} max|delta| {m[c].abs().max():.2e}   "
              f"mean|delta| {m[c].abs().mean():.2e}")
    ok = bool(m.d_frozen.abs().max() < 1e-6 and m.d_oracle.abs().max() < 1e-6)
    by = m.groupby("stream")[["d_frozen", "d_oracle"]].apply(
        lambda g: pd.Series({"max_abs": float(np.abs(g.values).max())}))
    print(by.to_string())
    return ok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=",".join(
        s for v in STORED.values() for s in v))
    a = ap.parse_args()
    print(f"SPLITS={SPLITS}, seeds {list(C.SEEDS)}, device {C.device}\n")
    vecs = load_vectors("glove")
    rows = []
    for s in a.streams.split(","):
        print(f"  {s}:", flush=True)
        run_stream(s, vecs, rows)
    t = pd.DataFrame(rows)
    # merge by stream, so that partial runs accumulate
    if os.path.exists("results/constructed_drift_rescore.parquet"):
        old = pd.read_parquet("results/constructed_drift_rescore.parquet")
        old = old[~old.stream.isin(t.stream.unique())]
        t = pd.concat([old, t], ignore_index=True)
    t.to_parquet("results/constructed_drift_rescore.parquet", index=False)
    print(f"\nwrote results/constructed_drift_rescore.parquet ({len(t)} rows)")

    ok, m = check_reproduction(t)
    m.to_csv("results/constructed_drift_cells.csv", index=False)
    if not ok:
        print("\nThe grid arm does not reproduce results/calib_splits200*.parquet; the rescored")
        print("checkpoints are not the ones behind the stored values.")
        return 2
    print("\nThe grid arm reproduces the stored values per (stream, seed, period).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
