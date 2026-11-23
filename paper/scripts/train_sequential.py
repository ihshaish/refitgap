#!/usr/bin/env python3
"""Sequential BiGRU training over one stream, one embedding condition and one seed.

Reads streams/stream_<stream>.parquet and the embeddings in ../embeddings; appends to results/e1_results.parquet,
writes checkpoints/<run>/period<k>.pt and, with --save-scores, the score matrices. Builds the BiGRU runs used by
the ASRS decomposition (Tables D.2 to D.4, Figure 4) and the constructed drifts (Table E.1). Needs the streams
built from the corpora, GloVe or the domain word2vec, and training; streams/stream_sample.parquet is a smoke test."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score

# training settings, the same for every condition and stored with every result row
KNOBS = dict(protocol="v2", max_len=256, batch=64, lr=1e-3, epochs=3,
             hidden=96, emb_dim=200, optimizer="adam", lr_reset="per_period",
             val_frac_p1=0.10)

EMB_PATHS = {
    "domain": os.environ.get("DOMAIN_EMB", "../embeddings/asrs_w2v_pre2010_200d.kv"),
    "glove": "../embeddings/glove.6B.200d.txt",
}


def load_vectors(kind):
    if kind == "domain":
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(EMB_PATHS["domain"])
        return {w: kv[w] for w in kv.index_to_key}
    vecs = {}
    with open(EMB_PATHS["glove"], encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vecs


class Vocab:
    """Index 0 = pad, 1 = OOV. With grow, unseen training tokens get new rows on first sight."""

    def __init__(self, initial_tokens, grow):
        self.grow = grow
        self.idx = {t: i + 2 for i, t in enumerate(initial_tokens)}

    def encode(self, text, train_mode):
        out = []
        for t in text.split()[:KNOBS["max_len"]]:
            i = self.idx.get(t)
            if i is None and self.grow and train_mode:
                i = len(self.idx) + 2
                self.idx[t] = i
            out.append(i if i is not None else 1)
        return out or [1]


class BiGRU(nn.Module):
    def __init__(self, emb_matrix, freeze, n_labels):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb_matrix, freeze=freeze, padding_idx=0)
        self.gru = nn.GRU(KNOBS["emb_dim"], KNOBS["hidden"], batch_first=True,
                          bidirectional=True)
        self.head = nn.Linear(2 * KNOBS["hidden"], n_labels)

    def grow_rows(self, n_new):
        if n_new <= 0:
            return
        old = self.emb.weight.data
        new = torch.randn(n_new, old.shape[1], device=old.device) * 0.1
        self.emb = nn.Embedding.from_pretrained(
            torch.cat([old, new]), freeze=False, padding_idx=0)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                     # terminal-hidden-state pooling
        return self.head(torch.cat([h[0], h[1]], dim=1))


def batches(codes, Y, rng=None):
    order = rng.permutation(len(codes)) if rng is not None else np.arange(len(codes))
    for s in range(0, len(order), KNOBS["batch"]):
        sel = order[s:s + KNOBS["batch"]]
        chunk = [codes[i] for i in sel]
        L = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for r, c in enumerate(chunk):
            x[r, :len(c)] = torch.tensor(c)
        yield x, torch.tensor([len(c) for c in chunk]), torch.tensor(Y[sel])


def evaluate(model, codes, Y, thresholds, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, lens, _ in batches(codes, Y):
            probs.append(torch.sigmoid(model(x.to(device), lens)).cpu().numpy())
    P = np.vstack(probs)
    per_f1 = f1_score(Y, (P >= thresholds).astype(int), average=None, zero_division=0)
    per_pr = np.array([average_precision_score(Y[:, k], P[:, k])
                       for k in range(Y.shape[1])])
    return float(per_f1.mean()), float(per_pr.mean()), P, per_f1, per_pr


def fit_thresholds(P, Y):
    """Per-label F1-optimal threshold on a 19-point grid."""
    ths = []
    for k in range(Y.shape[1]):
        grid = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(Y[:, k], (P[:, k] >= g).astype(int), zero_division=0) for g in grid]
        ths.append(grid[int(np.argmax(f1s))])
    return np.asarray(ths)


def train_period(model, codes, Y, device, seed):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=KNOBS["lr"])
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(KNOBS["epochs"]):
        for x, lens, y in batches(codes, Y, rng):
            opt.zero_grad()
            loss = lossf(model(x.to(device), lens), y.float().to(device))
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="primary")
    ap.add_argument("--condition", required=True,
                    choices=["C1", "C2", "C3", "C4", "C5", "C6", "scratch"],
                    help="C1/C2 frozen domain/GloVe embeddings, C3/C4 the same trainable, "
                         "C5 random trainable, C6 frozen random, scratch one model per period")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override for smoke tests only")
    ap.add_argument("--out", default="results/e1_results.parquet")
    ap.add_argument("--save-scores", default=None,
                    help="directory for per-run score matrices")
    args = ap.parse_args()
    if args.epochs:
        KNOBS["epochs"] = args.epochs
        KNOBS["protocol"] = f"v2-e{args.epochs}"
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_parquet(f"streams/stream_{args.stream}.parquet")
    labels = sorted({l for s in df.labels for l in s.split(";")})
    lab_idx = {l: i for i, l in enumerate(labels)}
    Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
    for r, s in enumerate(df.labels):
        for l in s.split(";"):
            Yall[r, lab_idx[l]] = 1
    N = int(df.period.max()) + 1

    # ---- vocabulary + embedding matrix per condition
    frozen = args.condition in ("C1", "C2", "C6")
    grow = args.condition in ("C3", "C4", "C5")
    rng = np.random.default_rng(args.seed)
    if args.condition in ("C1", "C3"):
        vecs = load_vectors("domain")
    elif args.condition in ("C2", "C4"):
        vecs = load_vectors("glove")
    else:
        vecs = None
    if vecs is not None:
        tokens = list(vecs)
        mat = np.vstack([np.zeros(KNOBS["emb_dim"]), rng.normal(0, .1, KNOBS["emb_dim"]),
                         np.stack([vecs[t] for t in tokens])]).astype(np.float32)
    else:  # C5, C6 and scratch: random vectors over the period-0 vocabulary
        p0 = df[(df.period == 0) & (df.split == "train_pool")]
        counts = pd.Series(" ".join(p0.text).split()).value_counts()
        tokens = list(counts[counts >= 5].index)
        mat = rng.normal(0, .1, (len(tokens) + 2, KNOBS["emb_dim"])).astype(np.float32)
        mat[0] = 0
    vocab = Vocab(tokens, grow=grow)

    def make_model():
        return BiGRU(torch.tensor(mat), freeze=frozen, n_labels=len(labels)).to(device)

    # ---- encode all splits (trainable-vocab growth happens period by period)
    test_codes, test_Y = {}, {}
    for k in range(N):
        te = df[(df.period == k) & (df.split == "test")]
        test_codes[k] = [vocab.encode(t, train_mode=False) for t in te.text]
        test_Y[k] = Yall[te.index.to_numpy()]

    run_id = f"{args.stream}_{args.condition}_s{args.seed}_{KNOBS['protocol']}"
    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    # period-1 validation split for the frozen thresholds
    p1 = df[(df.period == 0) & (df.split == "train_pool")]
    val_n = int(len(p1) * KNOBS["val_frac_p1"])
    val_sel = np.random.default_rng(20260802).choice(p1.index.to_numpy(), val_n, False)
    tr_sel0 = np.setdiff1d(p1.index.to_numpy(), val_sel)

    def encode_train(sel):
        return [vocab.encode(df.text[i], train_mode=True) for i in sel]

    def encode_tests(only=None):
        """Re-encode test sets with the current vocabulary."""
        for k in (range(N) if only is None else [only]):
            te = df[(df.period == k) & (df.split == "test")]
            test_codes[k] = [vocab.encode(t, train_mode=False) for t in te.text]

    if args.condition == "scratch":
        for k in range(N):
            # vocabulary from the period's own train pool
            trk = df[(df.period == k) & (df.split == "train_pool")]
            sel = trk.index.to_numpy() if k else tr_sel0
            counts = pd.Series(" ".join(df.text[i] for i in sel).split()).value_counts()
            ptoks = list(counts[counts >= 5].index)
            pmat = np.random.default_rng(args.seed + k).normal(
                0, .1, (len(ptoks) + 2, KNOBS["emb_dim"])).astype(np.float32)
            pmat[0] = 0
            vocab = Vocab(ptoks, grow=False)
            model = BiGRU(torch.tensor(pmat), freeze=False,
                          n_labels=len(labels)).to(device)
            codes = [vocab.encode(df.text[i], True) for i in sel]
            train_period(model, codes, Yall[sel], device, args.seed + k)
            vcodes = [vocab.encode(df.text[i], False) for i in val_sel]
            _, _, P, _, _ = evaluate(model, vcodes, Yall[val_sel], np.full(len(labels), .5), device)
            thresholds = fit_thresholds(P, Yall[val_sel])
            encode_tests(only=k)
            f1, auprc, _, pf1, ppr = evaluate(model, test_codes[k], test_Y[k], thresholds, device)
            rows.append(dict(stream=args.stream, condition="scratch", strategy="scratch",
                             seed=args.seed, i=k, j=k, macro_f1=f1, macro_auprc=auprc,
                             per_label_f1=json.dumps(pf1.round(4).tolist()),
                             per_label_auprc=json.dumps(ppr.round(4).tolist()),
                             labels=json.dumps(labels),
                             protocol=KNOBS["protocol"]))
            print(f"scratch period {k}: F1={f1:.3f} AUPRC={auprc:.3f}")
    else:
        model = make_model()
        thresholds = None
        _scores = {}
        for i in range(N):
            tri = df[(df.period == i) & (df.split == "train_pool")]
            sel = tri.index.to_numpy() if i else tr_sel0
            codes = encode_train(sel)
            model.grow_rows(len(vocab.idx) + 2 - model.emb.weight.shape[0])
            train_period(model, codes, Yall[sel], device, args.seed + i)
            if i == 0:
                vcodes = [vocab.encode(df.text[i_], False) for i_ in val_sel]
                _, _, P, _, _ = evaluate(model, vcodes, Yall[val_sel], np.full(len(labels), .5), device)
                thresholds = fit_thresholds(P, Yall[val_sel])
            # the embedding matrix is stored only for trainable conditions on seeds 0-2;
            # frozen embeddings are rebuilt from the embedding files
            sd = {k_: v.half() for k_, v in model.state_dict().items()
                  if not (k_ == "emb.weight" and (frozen or args.seed > 2))}
            # constructed-drift streams keep only the final checkpoint
            _probe = args.stream in ("primary", "secondary", "crossbreak")
            _interv = args.stream.startswith(("asrsperm", "asrsdrift", "asrsprior"))
            if _probe or (_interv and i == N - 1):
                torch.save(sd, ckpt_dir / f"period{i}.pt")
            if grow:
                encode_tests()   # newly learned tokens must be visible at evaluation
            for j in range(N):
                f1, auprc, P_ij, pf1, ppr = evaluate(model, test_codes[j], test_Y[j], thresholds, device)
                if args.save_scores:
                    _scores[f"P_{i}_{j}"] = P_ij.astype(np.float16)
                rows.append(dict(stream=args.stream, condition=args.condition,
                                 strategy="naive", seed=args.seed, i=i, j=j,
                                 macro_f1=f1, macro_auprc=auprc,
                                 per_label_f1=json.dumps(pf1.round(4).tolist()),
                                 per_label_auprc=json.dumps(ppr.round(4).tolist()),
                                 labels=json.dumps(labels),
                                 protocol=KNOBS["protocol"]))
            diag = next(r for r in rows if r["i"] == i and r["j"] == i)
            back = [r["macro_f1"] for r in rows if r["i"] == i and r["j"] < i]
            print(f"period {i}: R[{i},{i}]={diag['macro_f1']:.3f}"
                  + (f"  mean-back={np.mean(back):.3f}" if back else "")
                  + f"  ({time.time() - t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    new["knobs"] = json.dumps(KNOBS)
    if out.exists():
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True)
    new.to_parquet(out, index=False)
    print(f"appended {len(rows)} rows -> {out} ({time.time() - t0:.0f}s total)")
    if args.save_scores and args.condition != "scratch":
        sdir = Path(args.save_scores); sdir.mkdir(parents=True, exist_ok=True)
        for j in range(N):
            _scores[f"Y_{j}"] = test_Y[j].astype(np.int8)
        _scores["thresholds"] = np.asarray(thresholds, dtype=np.float32)
        spath = sdir / f"{args.stream}_{args.condition}_s{args.seed}_scores.npz"
        np.savez_compressed(spath, **_scores)
        assert spath.exists() and args.stream in spath.name
        print(f"saved score matrices -> {spath}")


if __name__ == "__main__":
    main()
