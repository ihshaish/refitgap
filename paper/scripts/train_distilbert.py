#!/usr/bin/env python3
"""Sequential DistilBERT training over the eight-period ASRS primary stream, keeping every score matrix.

Reads --data (streams/stream_primary.parquet); writes to --out, per seed, e4_results_seed<k>.parquet and
primary_distilbert_s<k>_scores.npz, resuming from _ckpt_s<k>.pt after an interruption (--stage: local copy).
Builds the DistilBERT runs of the DistilBERT decomposition. Needs the primary stream (ASRS export), the
distilbert-base-uncased weights and a GPU. Usage: --data ... --out DIR --seeds 0 1 2 [--smoke]"""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL = "distilbert-base-uncased"
MAXLEN, BS, EVAL_BS, EPOCHS, LR, WARMUP = 256, 32, 128, 3, 2e-5, 0.10
VAL_RNG_SEED = 20260802
GRID = np.linspace(0.05, 0.95, 19)


def _atomic(path, writer, suffix):
    """Write with `writer(tmp_path)` into the target directory, then rename."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=suffix)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_parquet(rows, path):
    _atomic(path, lambda p: pd.DataFrame(rows).to_parquet(p, index=False), ".parquet")


def save_npz(arrays, path):
    def w(p):
        with open(p, "wb") as fh:
            np.savez_compressed(fh, **arrays)
    _atomic(path, w, ".npz")


def save_ckpt(state, path):
    _atomic(path, lambda p: torch.save(state, p), ".pt")


class Store:
    """Files live in `stage`; each save is copied to `out` with retries."""

    def __init__(self, stage, out):
        self.stage, self.out = stage, out
        os.makedirs(stage, exist_ok=True)
        os.makedirs(out, exist_ok=True)

    def path(self, name):
        return os.path.join(self.stage, name)

    def pull(self, name):
        src, dst = os.path.join(self.out, name), self.path(name)
        if self.stage != self.out and os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)

    def publish(self, name):
        if self.stage == self.out:
            return
        src, dst = self.path(name), os.path.join(self.out, name)
        err = None
        for attempt in range(4):
            try:
                tmp = dst + ".part"
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
                if os.path.getsize(dst) == os.path.getsize(src):
                    return
                err = RuntimeError("size mismatch after copy")
            except Exception as e:  # network mounts fail transiently
                err = e
            time.sleep(10 * (attempt + 1))
        print(f"WARNING: could not publish {dst}: {err}", flush=True)

    def remove(self, name):
        for d in {self.stage, self.out}:
            p = os.path.join(d, name)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"WARNING: could not remove {p}: {e}", flush=True)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default=None, help="local work dir; outputs are copied from here to --out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="tiny subsample, plumbing check only")
    args = ap.parse_args()

    device = pick_device(args.device)
    store = Store(args.stage or args.out, args.out)
    print(f"device {device}; torch {torch.__version__}; stage {store.stage}; out {store.out}", flush=True)
    if device.type == "cpu" and not args.smoke:
        raise SystemExit("no GPU available: refusing to run the full protocol on CPU")

    df = pd.read_parquet(args.data)
    assert isinstance(df.index, pd.RangeIndex), "the stream must carry a RangeIndex"
    labels = sorted({l for s in df.labels for l in s.split(";")})
    lab_idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(df), len(labels)), dtype=np.float32)
    for r, s in enumerate(df.labels):
        for l in s.split(";"):
            Y[r, lab_idx[l]] = 1
    N = int(df.period.max()) + 1
    K = len(labels)
    print(f"{len(df)} records, {K} labels, {N} periods", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")

    def batches(idx, bs, rng=None):
        order = rng.permutation(len(idx)) if rng is not None else np.arange(len(idx))
        for s in range(0, len(order), bs):
            yield idx[order[s:s + bs]]

    def evaluate(model, idx, thresholds):
        model.eval()
        probs = []
        with torch.no_grad():
            for b in batches(idx, EVAL_BS):
                enc = encode(df.text.iloc[b]).to(device)
                probs.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
        P = np.vstack(probs)
        Yt = Y[idx]
        per_f1 = f1_score(Yt, (P >= thresholds).astype(int), average=None, zero_division=0)
        per_pr = np.array([average_precision_score(Yt[:, k], P[:, k]) for k in range(K)])
        return float(per_f1.mean()), float(per_pr.mean()), P, per_f1, per_pr

    def fit_thresholds(P, Yv):
        ths = []
        for k in range(Yv.shape[1]):
            f = [f1_score(Yv[:, k], (P[:, k] >= g).astype(int), zero_division=0) for g in GRID]
            ths.append(GRID[int(np.argmax(f))])
        return np.asarray(ths)

    # fixed partitions, identical for every seed
    p0 = df[(df.period == 0) & (df.split == "train_pool")].index.to_numpy()
    val = np.random.default_rng(VAL_RNG_SEED).choice(p0, int(len(p0) * 0.10), False)
    tr0 = np.setdiff1d(p0, val)
    train_idx = {i: (tr0 if i == 0 else df[(df.period == i) & (df.split == "train_pool")].index.to_numpy())
                 for i in range(N)}
    test_idx = {j: df[(df.period == j) & (df.split == "test")].index.to_numpy() for j in range(N)}
    if args.smoke:
        train_idx = {i: v[:64] for i, v in train_idx.items()}
        test_idx = {j: v[:64] for j, v in test_idx.items()}
        val = val[:64]

    lossf = torch.nn.BCEWithLogitsLoss()

    for seed in args.seeds:
        f_parquet = f"e4_results_seed{seed}.parquet"
        f_npz = f"primary_distilbert_s{seed}_scores.npz"
        f_ckpt = f"_ckpt_s{seed}.pt"
        for f in (f_parquet, f_npz, f_ckpt):
            store.pull(f)
        if os.path.exists(store.path(f_parquet)) and os.path.exists(store.path(f_npz)):
            try:
                done = (len(pd.read_parquet(store.path(f_parquet))) >= N * N
                        and "thresholds" in np.load(store.path(f_npz)).files)
            except Exception:
                done = False
            if done:
                print(f"seed {seed}: complete, skipping", flush=True)
                continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=K, problem_type="multi_label_classification").to(device)
        rows, scores, thresholds, start = [], {}, None, 0

        if os.path.exists(store.path(f_ckpt)):
            ck = torch.load(store.path(f_ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            rows, scores, thresholds, start = ck["rows"], ck["scores"], ck["thresholds"], ck["next_period"]
            torch.set_rng_state(ck["torch_rng"])
            if device.type == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"])
            print(f"seed {seed}: resumed at period {start} from checkpoint", flush=True)

        for i in range(start, N):
            t0 = time.time()
            tri = train_idx[i]
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            steps = (len(tri) // BS + 1) * EPOCHS
            sched = get_linear_schedule_with_warmup(opt, int(steps * WARMUP), steps)
            rng = np.random.default_rng(seed + i)
            model.train()
            for ep in range(EPOCHS):
                for b in batches(tri, BS, rng):
                    enc = encode(df.text.iloc[b]).to(device)
                    loss = lossf(model(**enc).logits, torch.tensor(Y[b]).to(device))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            _, _, Pv, _, _ = evaluate(model, val, np.full(K, 0.5))
            scores[f"Pval_{i}"] = Pv.astype(np.float16)
            if i == 0:
                thresholds = fit_thresholds(Pv, Y[val])
            for j in range(N):
                f1, pr, P_ij, pf1, ppr = evaluate(model, test_idx[j], thresholds)
                scores[f"P_{i}_{j}"] = P_ij.astype(np.float16)
                rows.append(dict(stream="primary", condition="distilbert", strategy="naive",
                                 seed=seed, i=i, j=j, macro_f1=f1, macro_auprc=pr,
                                 per_label_f1=json.dumps(pf1.round(4).tolist()),
                                 per_label_auprc=json.dumps(ppr.round(4).tolist()),
                                 labels=json.dumps(labels), protocol="v2-distilbert"))
            diag = [r for r in rows if r["i"] == i and r["j"] == i][0]["macro_f1"]
            save_parquet(rows, store.path(f_parquet))
            save_ckpt(dict(model=model.state_dict(), rows=rows, scores=scores, thresholds=thresholds,
                           next_period=i + 1, torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None),
                      store.path(f_ckpt))
            store.publish(f_parquet)
            store.publish(f_ckpt)
            print(f"seed {seed} period {i}: R[{i},{i}]={diag:.3f}  ({(time.time() - t0) / 60:.1f} min) saved",
                  flush=True)

        for j in range(N):
            scores[f"Y_{j}"] = Y[test_idx[j]].astype(np.int8)
        scores["thresholds"] = np.asarray(thresholds, dtype=np.float32)
        scores["Yval"] = Y[val].astype(np.int8)
        save_npz(scores, store.path(f_npz))
        store.publish(f_npz)
        store.remove(f_ckpt)
        print(f"seed {seed}: complete, score matrices saved", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    print("all seeds complete", flush=True)


if __name__ == "__main__":
    main()
