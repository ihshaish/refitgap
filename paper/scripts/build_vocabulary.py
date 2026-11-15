#!/usr/bin/env python3
"""Build the vocabulary-reassignment streams: paired labels swap their signature tokens from period 3 on.

Reads streams/stream_primary.parquet; writes streams/stream_asrsdrift{25,50,75}.parquet (rho, the swap
probability). Builds the fixed-pairing vocabulary streams of the constructed drifts (Table E.1,
Appendix E). Needs the primary stream, built from the ASRS export (not shipped)."""
import numpy as np, pandas as pd
from collections import Counter

SEED, TOPK, PSTAR = 20260802, 200, 3
st = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in st.labels for l in s.split(";")})
p0 = st[(st.period == 0) & (st.split == "train_pool")]
Y0 = np.array([[l in s.split(";") for l in labels] for s in p0.labels])
docs = [set(t.split()) for t in p0.text]
cnt = Counter(w for d in docs for w in d)

# top discriminative tokens per label by pointwise MI, period 0 only
per_label = {}
for k, lab in enumerate(labels):
    scores = {}
    for w, c in cnt.items():
        if c < 50: continue
        x = np.array([w in d for d in docs])
        pw, py = x.mean(), Y0[:, k].mean()
        pwy = ((x) & (Y0[:, k] == 1)).mean()
        if pwy > 0 and pw > 0 and py > 0:
            scores[w] = pwy * np.log2(pwy / (pw * py))
    per_label[lab] = [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:TOPK]]

# pair up labels and exchange their signature vocabularies rank for rank
rng = np.random.default_rng(SEED)
order = list(labels); rng.shuffle(order)
swap = {}
for a, b in zip(order[::2], order[1::2]):
    for wa, wb in zip(per_label[a], per_label[b]):
        if wa != wb:
            swap[wa] = wb; swap[wb] = wa
print(f"swap map covers {len(swap)} tokens across {len(order)//2} label pairs")

for rho in (0.25, 0.50, 0.75):
    out = st.copy()
    r2 = np.random.default_rng(SEED + int(rho * 100))
    def corrupt(row):
        if row.period < PSTAR: return row.text
        toks = row.text.split()
        return " ".join(swap[t] if (t in swap and r2.random() < rho) else t for t in toks)
    out["text"] = [corrupt(r) for r in out.itertuples()]
    tag = f"asrsdrift{int(rho*100)}"
    out.to_parquet(f"streams/stream_{tag}.parquet", index=False)
    print(f"  wrote stream_{tag}.parquet (rho={rho}, from period {PSTAR})")
