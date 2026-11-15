#!/usr/bin/env python3
"""Final-model levels on the constructed streams: macro-F1 at theta_1 (fitted on the period-1
validation split), the in-sample and split-half refits, and threshold refits on n labelled records
(argmax and prevalence-matching). python scripts/constructed_drift_levels.py [STREAM ...]
Writes results/calib_intervention_v2.parquet; imported by the other constructed_drift_* scripts.
Backs Table E.1. Needs the stream files and checkpoints (not shipped)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

DRAWS, SPLITS = 20, 20
NS = (100, 200, 500)
# main() overwrites the parquet, so SEEDS must span every seed in it; missing checkpoints are skipped.
SEEDS = range(10)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def macro_f1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def quantile_thresholds(P_test, prev):
    """Threshold per label so the predicted positive rate matches the estimated
    prevalence. Uses the labelled sample only through `prev` (one scalar each)."""
    th = np.empty(P_test.shape[1])
    for k in range(P_test.shape[1]):
        q = np.clip(1.0 - prev[k], 0.0, 1.0)
        th[k] = np.quantile(P_test[:, k], q) if 0 < q < 1 else (1.1 if q >= 1 else 0.0)
    return th


def run_stream(stream, rows):
    df = pd.read_parquet(f"streams/stream_{stream}.parquet")
    labels = sorted({l for s in df.labels for l in s.split(";")})
    lab_idx = {l: i for i, l in enumerate(labels)}
    Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
    for r, s in enumerate(df.labels):
        for l in s.split(";"):
            Yall[r, lab_idx[l]] = 1
    N = int(df.period.max()) + 1
    vecs = load_vectors("glove")

    for seed in SEEDS:
        ckpt = Path(f"checkpoints/{stream}_C2_s{seed}_v2/period{N-1}.pt")
        if not ckpt.exists():
            continue
        rng = np.random.default_rng(seed)
        tokens = list(vecs)
        mat = np.vstack([np.zeros(KNOBS["emb_dim"]),
                         rng.normal(0, .1, KNOBS["emb_dim"]),
                         np.stack([vecs[t] for t in tokens])]).astype(np.float32)
        vocab = Vocab(tokens, grow=False)
        model = BiGRU(torch.tensor(mat), freeze=True, n_labels=len(labels)).to(device)
        model.load_state_dict({k: v.float() for k, v in
                               torch.load(ckpt, map_location="cpu").items()},
                              strict=False)
        model.eval()

        p1 = df[(df.period == 0) & (df.split == "train_pool")]
        val_sel = np.random.default_rng(20260802).choice(
            p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
        vcodes = [vocab.encode(df.text[i], False) for i in val_sel]
        _, _, Pv, _, _ = evaluate(model, vcodes, Yall[val_sel],
                                  np.full(len(labels), .5), device)
        th_frozen = fit_thresholds(Pv, Yall[val_sel])

        for j in range(N):
            te = df[(df.period == j) & (df.split == "test")]
            Yt = Yall[te.index.to_numpy()]
            _, _, Pt, _, _ = evaluate(model, [vocab.encode(t, False) for t in te.text],
                                      Yt, np.full(len(labels), .5), device)
            pool = df[(df.period == j) & (df.split == "train_pool")].index.to_numpy()
            _, _, Pp, _, _ = evaluate(model, [vocab.encode(df.text[i], False) for i in pool],
                                      Yall[pool], np.full(len(labels), .5), device)
            Yp = Yall[pool]

            rec = dict(stream=stream, seed=seed, j=j,
                       f1_frozen=macro_f1(Pt, Yt, th_frozen),
                       f1_oracle_insample=macro_f1(Pt, Yt, fit_thresholds(Pt, Yt)))

            # split-half oracle: fit on half the test set, score on the other
            srng = np.random.default_rng(7000 + 100 * seed + j)
            vals = []
            for _ in range(SPLITS):
                perm = srng.permutation(len(Yt)); h = len(Yt) // 2
                A, B = perm[:h], perm[h:]
                vals.append(0.5 * (macro_f1(Pt[B], Yt[B], fit_thresholds(Pt[A], Yt[A])) +
                                   macro_f1(Pt[A], Yt[A], fit_thresholds(Pt[B], Yt[B]))))
            rec["f1_oracle_splithalf"] = float(np.mean(vals))

            drng = np.random.default_rng(1000 * seed + j)
            for n in NS:
                if len(pool) < n:
                    continue
                f1_max, f1_qt = [], []
                for _ in range(DRAWS):
                    take = drng.choice(len(pool), n, replace=False)
                    f1_max.append(macro_f1(Pt, Yt, fit_thresholds(Pp[take], Yp[take])))
                    f1_qt.append(macro_f1(Pt, Yt,
                                          quantile_thresholds(Pt, Yp[take].mean(0))))
                rec[f"recal_{n}_argmax"] = float(np.mean(f1_max))
                rec[f"recal_{n}_quantile"] = float(np.mean(f1_qt))
            rows.append(rec)
        print(f"  {stream} s{seed}: done", flush=True)


def main():
    streams = sys.argv[1:] or ["asrsprior00", "asrsprior40", "asrsprior100",
                               "asrsdrift50", "asrsperm100"]
    rows = []
    for s in streams:
        print(f"{s}:", flush=True)
        run_stream(s, rows)
    t = pd.DataFrame(rows)

    # missing checkpoints would shrink the file without notice
    found = t.groupby("stream")["seed"].nunique()
    short = found[found < len(SEEDS)]
    if len(short):
        print("\nWARNING: fewer seeds than requested, checkpoints missing for:")
        print(short.to_string())
        print(f"expected {len(SEEDS)} per stream. Writing anyway; rerun when "
              f"training finishes.")

    t.to_parquet("results/calib_intervention_v2.parquet", index=False)
    g = t.groupby("stream")[["f1_frozen", "f1_oracle_insample",
                             "f1_oracle_splithalf", "recal_500_argmax",
                             "recal_500_quantile"]].mean()
    g["oracle_optimism"] = g.f1_oracle_insample - g.f1_oracle_splithalf
    print("\n" + g.round(4).to_string())


if __name__ == "__main__":
    main()
