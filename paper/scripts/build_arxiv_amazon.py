#!/usr/bin/env python3
"""Build the arXiv (2012-2023) and Amazon CDs and Vinyl (2012-2022) streams; prints a label-schema check first.

Reads ../data_external/prospective/{arxiv/*.parquet, amazon/CDs_and_Vinyl.jsonl}; writes
streams/stream_arxiv.parquet and streams/stream_amazon.parquet, used in streams and settings
(Tables B.1, B.2). Needs both corpora (scripts/fetch_data.py), which the repository does not ship."""
import datetime as dt
import glob
import json
import re

import numpy as np
import pandas as pd

DATA_SEED = 20260802; N_PERIODS = 8; TEST_FRAC = 0.20; MIN_POS = 75
TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")
def norm(t): return " ".join(TOKEN_RE.findall(t.lower()))

def finalise(df, name, min_pos=MIN_POS):
    df = df.sort_values(["yyyymm", "acn"]).reset_index(drop=True)
    bounds = np.linspace(0, len(df), N_PERIODS + 1).astype(int)
    df["period"] = 0
    for k in range(N_PERIODS):
        df.loc[bounds[k]:bounds[k+1]-1, "period"] = k
    rng = np.random.default_rng(DATA_SEED)
    df["split"] = "train_pool"
    for k in range(N_PERIODS):
        idx = df.index[df.period == k]
        df.loc[rng.choice(idx, int(len(idx)*TEST_FRAC), replace=False), "split"] = "test"
    all_lab = sorted({l for s in df.labels for l in s.split(";")})
    keep = [l for l in all_lab
            if min(((df.period==k)&(df.split=="test")&
                    df.labels.str.split(";").apply(lambda x: l in x)).sum()
                   for k in range(N_PERIODS)) >= min_pos]
    df["labels"] = df.labels.map(lambda s: ";".join(l for l in s.split(";") if l in keep))
    df = df[df.labels.str.len() > 0].reset_index(drop=True)
    df.to_parquet(f"streams/stream_{name}.parquet", index=False)
    print(f"  -> {name}: {len(df)} records, {df.split.eq('test').sum()} test, "
          f"{len(keep)} labels, {df.yyyymm.min()}-{df.yyyymm.max()}, "
          f"{df.labels.str.count(';').add(1).mean():.2f} labels/rec")
    print("  ", df.groupby("period").yyyymm.agg(['min','max','count']).to_string().replace("\n","\n   "))
    return keep

# arXiv
print("=== arXiv: schema check ===", flush=True)
rows = []
for f in sorted(glob.glob("../data_external/prospective/arxiv/*.parquet")):
    d = pd.read_parquet(f, columns=["id","title","abstract","categories","versions"])
    for r in d.itertuples(index=False):
        try:
            v1 = r.versions[0]["created"]          # 'Mon, 2 Apr 2007 19:18:42 GMT'
            parts = v1.split()
            y, mon = int(parts[3]), parts[2]
            m = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index(mon)+1
        except Exception:
            continue
        prim = [c.split(".")[0] for c in r.categories.split()]
        rows.append((y*100+m, norm((r.title or "") + " " + (r.abstract or "")), ";".join(sorted(set(prim)))))
ax = pd.DataFrame(rows, columns=["yyyymm","text","labels"])
ax["acn"] = np.arange(len(ax)).astype(str)
ax = ax[ax.text.str.split().str.len() >= 20]
yr = ax.assign(y=ax.yyyymm//100)
per_year = yr.groupby("y").size()
print(per_year.tail(20).to_string())
# archive availability per year (schema check)
arch = {}
for a in ["cs","math","physics","stat","eess","q-bio","econ","astro-ph","cond-mat"]:
    s = yr[yr.labels.str.contains(rf"(^|;){a}($|;)", regex=True)].groupby("y").size()
    arch[a] = s
av = pd.DataFrame(arch).fillna(0).astype(int)
print("\narchive counts by year (tail):"); print(av.tail(12).to_string())
# window: archives present every year with >=2000/yr; recent-but-not-partial
WIN = (2012, 2023)
stable = [a for a in av.columns if av.loc[WIN[0]:WIN[1], a].min() >= 2000]
print(f"\narXiv window {WIN}, archives stable throughout: {stable}")
sub = ax[(ax.yyyymm >= WIN[0]*100) & (ax.yyyymm <= WIN[1]*100+12)].copy()
sub["labels"] = sub.labels.map(lambda s: ";".join(l for l in s.split(";") if l in stable))
sub = sub[sub.labels.str.len() > 0]
# subsample for tractability, preserving chronology
if len(sub) > 120000:
    sub = sub.sample(120000, random_state=DATA_SEED)
print(f"arXiv candidate records: {len(sub)}")
finalise(sub[["acn","yyyymm","text","labels"]], "arxiv")

# Amazon
print("\n=== Amazon CDs&Vinyl: schema check ===", flush=True)
rows = []
with open("../data_external/prospective/amazon/CDs_and_Vinyl.jsonl") as f:
    for line in f:
        try: r = json.loads(line)
        except Exception: continue
        ts = r.get("timestamp")
        if not ts: continue
        d0 = dt.datetime.utcfromtimestamp(ts/1000 if ts > 1e11 else ts)
        txt = norm((r.get("title") or "") + " " + (r.get("text") or ""))
        if len(txt.split()) < 15: continue
        rt = r.get("rating")
        if rt is None: continue
        rows.append((d0.year*100+d0.month, txt, f"star{int(rt)}"))
        if len(rows) >= 400000: break
am = pd.DataFrame(rows, columns=["yyyymm","text","labels"])
am["acn"] = np.arange(len(am)).astype(str)
ay = am.assign(y=am.yyyymm//100)
print(ay.groupby("y").size().tail(18).to_string())
print("\nrating distribution by year (share):")
print(
    ay.groupby(["y", "labels"])
    .size()
    .unstack(fill_value=0)
    .pipe(lambda x: (x.T / x.sum(1)).T)
    .tail(10)
    .round(3)
    .to_string()
)
WIN_A = (2012, 2022)
sub = am[(am.yyyymm >= WIN_A[0]*100) & (am.yyyymm <= WIN_A[1]*100+12)]
if len(sub) > 120000:
    sub = sub.sample(120000, random_state=DATA_SEED)
print(f"\nAmazon window {WIN_A}, candidate records: {len(sub)}")
finalise(sub[["acn","yyyymm","text","labels"]], "amazon")
print("\nprospective streams built")
