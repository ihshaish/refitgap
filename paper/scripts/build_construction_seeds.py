#!/usr/bin/env python3
"""Rebuild the constructed drifts under another construction seed S (same logic as build_vocabulary.py,
build_vocabulary_perm.py and build_prior_shift.py). Reads streams/stream_primary.parquet; writes
streams/stream_{asrsdrift{S}_r50, asrsdrift{S}_r100, asrsperm{S}_r100, asrsprior{S}_00, asrsprior{S}_100}.parquet.
Backs the repeated constructions of the constructed drifts (Table E.1, Appendix E).
Needs the primary stream, built from the ASRS export (not shipped). Usage: --seed 20260921"""
import argparse, os
from collections import Counter
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--force", action="store_true")
ap.add_argument(
    "--streams", default="full", choices=["full", "matched"], help="matched = prior100, prior00 and drift r50 only"
)
args = ap.parse_args()
S = args.seed

st = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in st.labels for l in s.split(";")})
p0 = st[(st.period == 0) & (st.split == "train_pool")]
Y0 = np.array([[l in s.split(";") for l in labels] for s in p0.labels])
docs = [set(t.split()) for t in p0.text]
cnt = Counter(w for d in docs for w in d)


def write(out, tag):
    # names start asrsdrift/asrsperm/asrsprior so that train_sequential.py keeps the final checkpoint
    path = f"streams/stream_{tag}.parquet"
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"refusing to overwrite {path}")
    out.to_parquet(path, index=False)
    print(f"  wrote {path}")


def mi_lists(topk, min_count):
    per_label = {}
    for k, lab in enumerate(labels):
        sc = {}
        for w, c in cnt.items():
            if c < min_count:
                continue
            x = np.array([w in d for d in docs])
            pw, py = x.mean(), Y0[:, k].mean()
            pwy = (x & (Y0[:, k] == 1)).mean()
            if pwy > 0 and pw > 0 and py > 0:
                sc[w] = pwy * np.log2(pwy / (pw * py))
        per_label[lab] = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:topk]]
    return per_label


# fixed label pairing
per_label = mi_lists(200, 50)
rng = np.random.default_rng(S)
order = list(labels); rng.shuffle(order)
swap = {}
for a, b in zip(order[::2], order[1::2]):
    for wa, wb in zip(per_label[a], per_label[b]):
        if wa != wb:
            swap[wa] = wb; swap[wb] = wa
print(f"pairing seed {S}: swap map covers {len(swap)} tokens across {len(order)//2} label pairs")
PSTAR_A = 3
for rho in ((0.5,) if args.streams == "matched" else (0.5, 1.0)):
    out = st.copy()
    r2 = np.random.default_rng(S + int(rho * 100))
    txt = []
    for row in out.itertuples():
        if row.period < PSTAR_A:
            txt.append(row.text); continue
        txt.append(" ".join(swap[t] if (t in swap and r2.random() < rho) else t for t in row.text.split()))
    out["text"] = txt
    write(out, f"asrsdrift{S}_r{int(rho*100)}")

# per-period permutation
per_label_b = mi_lists(400, 40) if args.streams != "matched" else None
if per_label_b is not None:
    pool = sorted({w for v in per_label_b.values() for w in v})
    print(f"permutation pool: {len(pool)} tokens")
    PSTAR_B = 2
    rho = 1.0
    out = st.copy(); r2 = np.random.default_rng(S + 100)
    perms = {p: dict(zip(pool, r2.permutation(pool))) for p in range(PSTAR_B, out.period.max() + 1)}
    txt = []
    for row in out.itertuples():
        if row.period < PSTAR_B:
            txt.append(row.text); continue
        m = perms[row.period]
        txt.append(" ".join(m[t] if (t in m and r2.random() < rho) else t for t in row.text.split()))
    out["text"] = txt
    write(out, f"asrsperm{S}_r100")

# prior tilt and its uniform control
PSTAR_P, KEEP, TILT = 3, 0.35, 5.0
Y = np.array([[l in s.split(";") for l in labels] for s in st.labels])
pv0 = Y[(st.period == 0).values].mean(0)
direction = np.where(pv0 <= np.median(pv0), 1.0, -1.0)


def draw(idx, n, rho, rng):
    if rho == 0:
        return rng.choice(idx, size=n, replace=False)
    w = np.exp(rho * TILT * (Y[idx] @ direction))
    return rng.choice(idx, size=n, replace=False, p=w / w.sum())


for rho, tag in ((0.0, f"asrsprior{S}_00"), (1.0, f"asrsprior{S}_100")):
    rng = np.random.default_rng(S + int(rho * 100))
    keep = []
    for period in sorted(st.period.unique()):
        allp = np.flatnonzero((st.period == period).values)
        shuf = rng.permutation(allp); half = len(shuf) // 2
        A, B = shuf[:half], shuf[half:]
        r = 0.0 if period < PSTAR_P else rho
        n = int(round(KEEP * half))
        tr_df = st.iloc[draw(A, n, r, rng)].copy(); tr_df["split"] = "train_pool"
        te_df = st.iloc[draw(B, n, r, rng)].copy(); te_df["split"] = "test"
        keep += [tr_df, te_df]
    out = pd.concat(keep, ignore_index=True).sort_values(["period", "yyyymm"]).reset_index(drop=True)
    tr = out[out.split == "train_pool"]
    Yt = np.array([[l in s.split(";") for l in labels] for s in tr.labels])
    pv = np.array([Yt[(tr.period == p).values].mean(0) for p in sorted(tr.period.unique())])
    print(
        f"  {tag}: n_train={len(tr)} mean |prior(final) - prior(first)| = {np.abs(pv[-1] - pv[0]).mean():.4f}; "
        f"min positives per period in test = "
        f"{int(min(out[out.split == 'test'].groupby('period').apply(lambda g: np.array([[l in s.split(';') for l in labels] for s in g.labels]).sum(0).min())))}"
    )
    write(out, tag)
print("done")
