#!/usr/bin/env python3
"""Rebuild the fixed-pairing token map of the asrsdrift streams (build_vocabulary.py, same seed)
and check it: distinct images, images within the selected vocabulary, f(f(w)) = w, and tokens on
more than one label's list. Reads streams/stream_primary.parquet and scripts/build_vocabulary.py;
writes results/asrs_supplementary/pairing_check.json. Backs Appendix E, "How the streams were built".
Needs the stream files (not shipped)."""
import json, re, numpy as np, pandas as pd
from collections import Counter
from pathlib import Path
src = open(Path(__file__).with_name("build_vocabulary.py")).read()
seed = int(re.search(r"SEED, TOPK, PSTAR = (\d+)", src).group(1))
st = pd.read_parquet("streams/stream_primary.parquet"); labels = sorted({l for s in st.labels for l in s.split(";")})
p0 = st[(st.period == 0) & (st.split == "train_pool")]
Y0 = np.array([[l in s.split(";") for l in labels] for s in p0.labels]); docs = [set(t.split()) for t in p0.text]
cnt = Counter(w for d in docs for w in d); per_label = {}
for k, lab in enumerate(labels):
    sc = {}
    for w, c in cnt.items():
        if c < 50: continue
        x = np.array([w in d for d in docs]); pw, py = x.mean(), Y0[:, k].mean(); pwy = (x & (Y0[:, k] == 1)).mean()
        if pwy > 0 and pw > 0 and py > 0: sc[w] = pwy * np.log2(pwy / (pw * py))
    per_label[lab] = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:200]]
rng = np.random.default_rng(seed); order = list(labels); rng.shuffle(order)
swap, writes = {}, Counter()
for a, b in zip(order[::2], order[1::2]):
    for wa, wb in zip(per_label[a], per_label[b]):
        if wa != wb: swap[wa] = wb; swap[wb] = wa; writes[wa] += 1; writes[wb] += 1
selected = set(w for v in per_label.values() for w in v)
out = dict(
    seed_used=seed,
    mapped_tokens=len(swap),
    outputs_unique=len(set(swap.values())) == len(swap),
    n_distinct_outputs=len(set(swap.values())),
    outputs_within_selected_vocabulary=all(v in selected for v in swap.values()),
    involution_share=float(np.mean([swap.get(swap[w]) == w for w in swap])),
    tokens_assigned_more_than_once=int(sum(1 for w, c in writes.items() if c > 1)),
    tokens_in_more_than_one_label_list=int(
        sum(1 for w, c in Counter(w for v in per_label.values() for w in v).items() if c > 1)
    ),
    pairs=[[a, b] for a, b in zip(order[::2], order[1::2])],
    unpaired_label=order[-1] if len(order) % 2 else None,
)
print(json.dumps(out, indent=1)); json.dump(out, open("results/asrs_supplementary/pairing_check.json", "w"), indent=1)
