#!/usr/bin/env python3
"""Sequential BiGRU training with replay, EWC, a bounded window or a scaled embedding learning rate.

Reads streams/stream_<stream>.parquet and the embeddings in ../embeddings; writes <out-dir>/<run>.parquet and
<run>_scores.npz (default results/methods), no checkpoints. Builds the runs used by the paired refit (Figure 2,
Table C.1), the benchmark decompositions (Table D.1) and the training-regime factorial (Table E.2).
Needs the streams built from the corpora, GloVe or the domain word2vec, and training."""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_sequential import (KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors,
                              batches)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fisher_diag(model, codes, Y, seed, n_batches=40):
    """Empirical diagonal Fisher (mean squared gradient) over at most n_batches batches."""
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if p.requires_grad}
    model.eval()
    for b, (x, lens, y) in enumerate(batches(codes, Y, rng)):
        if b >= n_batches:
            break
        model.zero_grad()
        lossf(model(x.to(device), lens), y.float().to(device)).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= max(1, min(b + 1, n_batches))
    model.zero_grad()
    return fisher


def train_period(model, codes, Y, seed, emb_lr_mult, ewc=None, lam=0.0):
    """As train_sequential.train_period, with its own embedding learning rate and an
    optional EWC penalty; ewc is (fisher, anchor parameters)."""
    emb = [p for n, p in model.named_parameters() if n == "emb.weight" and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if n != "emb.weight" and p.requires_grad]
    groups = [{"params": rest, "lr": KNOBS["lr"]}]
    if emb:
        groups.append({"params": emb, "lr": KNOBS["lr"] * emb_lr_mult})
    opt = torch.optim.Adam(groups)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(KNOBS["epochs"]):
        for x, lens, y in batches(codes, Y, rng):
            opt.zero_grad()
            loss = lossf(model(x.to(device), lens), y.float().to(device))
            if ewc is not None and lam > 0:
                fisher, anchor = ewc
                pen = sum((fisher[n] * (p - anchor[n]) ** 2).sum()
                          for n, p in model.named_parameters()
                          if p.requires_grad and n in fisher
                          and p.shape == anchor[n].shape)
                loss = loss + 0.5 * lam * pen
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="primary")
    ap.add_argument("--condition", default="C2", choices=["C1", "C2", "C3", "C4"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--strategy", default="sequential", choices=["sequential", "replay", "ewc"],
                    type=lambda v: "sequential" if v == "naive" else v)   # stored configs may say naive
    ap.add_argument("--ewc-lambda", type=float, default=1000.0)
    ap.add_argument("--window", type=int, default=0,
                    help="train on the last k periods only; 0 keeps the full history")
    ap.add_argument("--buffer-frac", type=float, default=0.05,
                    help="reservoir budget as a fraction of the stream train pool")
    ap.add_argument("--emb-lr-mult", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=KNOBS["epochs"],
                    help="optimisation epochs per period (the factorial uses 3 and 10)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/methods")
    args = ap.parse_args()
    if args.strategy == "sequential":
        args.strategy = "naive"   # the name stored in result files
    KNOBS["epochs"] = args.epochs

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"{args.strategy}" +
                       (f"_ewc{args.ewc_lambda:g}" if args.strategy == "ewc" else "") +
                       (f"_win{args.window}" if args.window else "") +
                       f"_lam{args.emb_lr_mult:g}" +
                       (f"_ep{args.epochs}" if args.epochs != 3 else ""))
    run_id = f"{args.stream}_{args.condition}_{tag}_s{args.seed}"
    if (out_dir / f"{run_id}.parquet").exists():
        print(f"{run_id}: already done, skipping"); return

    torch.manual_seed(args.seed)

    df = pd.read_parquet(f"streams/stream_{args.stream}.parquet")
    labels = sorted({l for s in df.labels for l in s.split(";")})
    lab_idx = {l: i for i, l in enumerate(labels)}
    Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
    for r, s in enumerate(df.labels):
        for l in s.split(";"):
            Yall[r, lab_idx[l]] = 1
    N = int(df.period.max()) + 1

    frozen = args.condition in ("C1", "C2")
    grow = args.condition in ("C3", "C4")
    rng = np.random.default_rng(args.seed)
    vecs = load_vectors("domain" if args.condition in ("C1", "C3") else "glove")
    tokens = list(vecs)
    mat = np.vstack([np.zeros(KNOBS["emb_dim"]), rng.normal(0, .1, KNOBS["emb_dim"]),
                     np.stack([vecs[t] for t in tokens])]).astype(np.float32)
    vocab = Vocab(tokens, grow=grow)
    model = BiGRU(torch.tensor(mat), freeze=frozen, n_labels=len(labels)).to(device)

    test_codes, test_Y = {}, {}
    for k in range(N):
        te = df[(df.period == k) & (df.split == "test")]
        test_codes[k] = [vocab.encode(t, False) for t in te.text]
        test_Y[k] = Yall[te.index.to_numpy()]

    p1 = df[(df.period == 0) & (df.split == "train_pool")]
    val_sel = np.random.default_rng(20260802).choice(
        p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
    tr_sel0 = np.setdiff1d(p1.index.to_numpy(), val_sel)

    pool_total = int((df.split == "train_pool").sum())
    budget = int(round(args.buffer_frac * pool_total))
    buf, seen = [], 0                      # reservoir over records already trained on
    brng = np.random.default_rng(10_000 + args.seed)

    ewc_state, per_period_ids = None, {}
    rows, scores, thresholds, t0 = [], {}, None, time.time()
    for i in range(N):
        tri = df[(df.period == i) & (df.split == "train_pool")]
        sel = list(tri.index.to_numpy() if i else tr_sel0)
        per_period_ids[i] = sel

        train_ids = list(sel)
        if args.window:
            # bounded history: the most recent k periods, this one included
            for q in range(max(0, i - args.window + 1), i):
                train_ids += list(per_period_ids[q])
        if args.strategy == "replay" and buf:
            train_ids += list(buf)         # the buffer is replayed in full each period

        codes = [vocab.encode(df.text[k_], True) for k_ in train_ids]
        model.grow_rows(len(vocab.idx) + 2 - model.emb.weight.shape[0])
        train_period(model, codes, Yall[np.array(train_ids)], args.seed + i,
                     args.emb_lr_mult,
                     ewc=ewc_state if args.strategy == "ewc" else None,
                     lam=args.ewc_lambda)

        if args.strategy == "ewc":
            f = fisher_diag(model, codes, Yall[np.array(train_ids)], args.seed + i)
            if ewc_state is None:
                ewc_state = (f, {n: p.detach().clone()
                                 for n, p in model.named_parameters() if p.requires_grad})
            else:
                old, _ = ewc_state
                acc = {n: old[n] + f[n] if n in old and old[n].shape == f[n].shape
                       else f[n] for n in f}
                ewc_state = (acc, {n: p.detach().clone()
                                   for n, p in model.named_parameters() if p.requires_grad})

        if args.strategy == "replay":      # Vitter's algorithm R, after training
            for rid in sel:
                seen += 1
                if len(buf) < budget:
                    buf.append(rid)
                else:
                    j_ = int(brng.integers(0, seen))
                    if j_ < budget:
                        buf[j_] = rid

        if i == 0:
            vcodes = [vocab.encode(df.text[k_], False) for k_ in val_sel]
            _, _, P, _, _ = evaluate(model, vcodes, Yall[val_sel],
                                     np.full(len(labels), .5), device)
            thresholds = fit_thresholds(P, Yall[val_sel])
        if grow:
            for k in range(N):
                te = df[(df.period == k) & (df.split == "test")]
                test_codes[k] = [vocab.encode(t, False) for t in te.text]

        for j in range(N):
            f1, auprc, P, pf1, ppr = evaluate(model, test_codes[j], test_Y[j],
                                              thresholds, device)
            scores[f"P_{i}_{j}"] = P.astype(np.float16)
            rows.append(dict(stream=args.stream, condition=args.condition,
                             strategy=args.strategy, buffer_frac=args.buffer_frac,
                             window=args.window, ewc_lambda=args.ewc_lambda,
                             emb_lr_mult=args.emb_lr_mult, seed=args.seed, i=i, j=j,
                             macro_f1=f1, macro_auprc=auprc,
                             per_label_f1=json.dumps(pf1.round(4).tolist()),
                             per_label_auprc=json.dumps(ppr.round(4).tolist()),
                             labels=json.dumps(labels), protocol="e2"))
        d = next(r for r in rows if r["i"] == i and r["j"] == i)
        print(f"  period {i}: R[{i},{i}]={d['macro_f1']:.3f} "
              f"buf={len(buf)} ({time.time()-t0:.0f}s)", flush=True)

    for j in range(N):
        scores[f"Y_{j}"] = test_Y[j].astype(np.int8)
    scores["thresholds"] = thresholds.astype(np.float32)
    np.savez_compressed(out_dir / f"{run_id}_scores.npz", **scores)
    pd.DataFrame(rows).to_parquet(out_dir / f"{run_id}.parquet", index=False)
    mb = (out_dir / f"{run_id}_scores.npz").stat().st_size / 1e6
    print(f"{run_id}: done in {time.time()-t0:.0f}s, scores {mb:.1f} MB")


if __name__ == "__main__":
    main()
