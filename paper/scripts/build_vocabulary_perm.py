#!/usr/bin/env python3
"""Build the per-period permutation streams: from period 2 on, a new permutation of the pooled signature tokens.

Reads streams/stream_primary.parquet; writes streams/stream_asrsperm{50,100}<suffix>.parquet, refusing
to overwrite without --force. Builds the permutation vocabulary streams of the constructed drifts
(Table E.1, Appendix E). Needs the primary stream, built from the ASRS export (not shipped)."""
import argparse, os
import numpy as np, pandas as pd
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="", help="appended to the stream tag, e.g. _recon")
ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
ap.add_argument("--streams", default="streams", help="streams directory")
args = ap.parse_args()

SEED, TOPK, PSTAR = 20260802, 400, 2
st = pd.read_parquet(f"{args.streams}/stream_primary.parquet")
labels = sorted({l for s in st.labels for l in s.split(";")})
p0 = st[(st.period==0)&(st.split=="train_pool")]
Y0 = np.array([[l in s.split(";") for l in labels] for s in p0.labels])
docs = [set(t.split()) for t in p0.text]; cnt = Counter(w for d in docs for w in d)
per_label={}
for k,lab in enumerate(labels):
    sc={}
    for w,c in cnt.items():
        if c<40: continue
        x=np.array([w in d for d in docs]); pw,py=x.mean(),Y0[:,k].mean()
        pwy=((x)&(Y0[:,k]==1)).mean()
        if pwy>0 and pw>0 and py>0: sc[w]=pwy*np.log2(pwy/(pw*py))
    per_label[lab]=[w for w,_ in sorted(sc.items(), key=lambda kv:-kv[1])[:TOPK]]
pool = sorted({w for v in per_label.values() for w in v})
print("intervention vocabulary:", len(pool), "tokens")
for rho in (0.5, 1.0):
    out = st.copy(); r2 = np.random.default_rng(SEED+int(rho*100))
    # permutations are drawn first, then one uniform per pooled-token occurrence in stream order
    perms = {p: dict(zip(pool, r2.permutation(pool))) for p in range(PSTAR, out.period.max()+1)}
    txt=[]
    for row in out.itertuples():
        if row.period < PSTAR: txt.append(row.text); continue
        m = perms[row.period]
        txt.append(" ".join(m[t] if (t in m and r2.random()<rho) else t for t in row.text.split()))
    out["text"]=txt
    tag=f"asrsperm{int(rho*100)}{args.suffix}"
    path = f"{args.streams}/stream_{tag}.parquet"
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"refusing to overwrite {path}; pass --force or --suffix")
    out.to_parquet(path, index=False)
    print(f"  wrote stream_{tag}.parquet (rho={rho}, from period index {PSTAR})")
