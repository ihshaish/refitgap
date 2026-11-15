#!/usr/bin/env python3
"""Build the two vocabulary cells of the 2x2 prior-by-vocabulary factorial and check the manipulations.

Reads streams/stream_{primary,asrsprior00,asrsprior100}.parquet; writes streams/stream_asrsp0v50.parquet
and stream_asrsp1v50.parquet (rho = 0.5 swaps of build_vocabulary.py on the two prior cells); exits
nonzero if a check fails. Backs the factorial of the constructed drifts (Table E.1, Appendix E).
Needs the primary and reweighted streams, built from the ASRS export (not shipped)."""
import sys
import numpy as np
import pandas as pd
from collections import Counter

SEED, TOPK, PSTAR, RHO = 20260802, 200, 3, 0.50

prim = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in prim.labels for l in s.split(";")})
p0 = prim[(prim.period == 0) & (prim.split == "train_pool")]
Y0 = np.array([[l in s.split(";") for l in labels] for s in p0.labels])
docs = [set(t.split()) for t in p0.text]
cnt = Counter(w for d in docs for w in d)

per_label = {}
for k, lab in enumerate(labels):
    scores = {}
    for w, c in cnt.items():
        if c < 50:
            continue
        x = np.array([w in d for d in docs])
        pw, py = x.mean(), Y0[:, k].mean()
        pwy = (x & (Y0[:, k] == 1)).mean()
        if pwy > 0 and pw > 0 and py > 0:
            scores[w] = pwy * np.log2(pwy / (pw * py))
    per_label[lab] = [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:TOPK]]

rng = np.random.default_rng(SEED)
order = list(labels); rng.shuffle(order)
swap = {}
for a, b in zip(order[::2], order[1::2]):
    for wa, wb in zip(per_label[a], per_label[b]):
        if wa != wb:
            swap[wa] = wb; swap[wb] = wa
print(f"swap map covers {len(swap)} tokens across {len(order)//2} label pairs")


def priors(df):
    tr = df[df.split == "train_pool"]
    Yt = np.array([[l in s.split(";") for l in labels] for s in tr.labels])
    return np.array([Yt[(tr.period == p).values].mean(0)
                     for p in sorted(tr.period.unique())])


def drift(df):
    pv = priors(df)
    return np.abs(pv[-1] - pv[0]).mean()


cells = {}
for src, v0_key, v1_key, v1_tag, salt in [
        ("asrsprior00", "P0V0", "P0V1", "asrsp0v50", 0),
        ("asrsprior100", "P1V0", "P1V1", "asrsp1v50", 1)]:
    base = pd.read_parquet(f"streams/stream_{src}.parquet")
    out = base.copy()
    r2 = np.random.default_rng(SEED + int(RHO * 100) + salt)
    def corrupt(row):
        if row.period < PSTAR:
            return row.text
        return " ".join(swap[t] if (t in swap and r2.random() < RHO) else t
                        for t in row.text.split())
    out["text"] = [corrupt(r) for r in out.itertuples()]
    out.to_parquet(f"streams/stream_{v1_tag}.parquet", index=False)
    cells[v0_key] = base
    cells[v1_key] = out
    print(f"  wrote stream_{v1_tag}.parquet from {src} (rho={RHO}, from period {PSTAR})")

# P0 cells inherit the corpus's natural prior drift, so they are checked against
# the primary stream's own drift, not against zero.
nat = drift(prim)
checks, ok = [], True
for tag in ("P0V0", "P0V1"):
    d = drift(cells[tag])
    good = abs(d - nat) <= 0.02
    ok &= good
    checks.append(f"prior drift {tag} = {d:.4f} vs natural {nat:.4f}  "
                  f"({'PASS' if good else 'FAIL'})")
for tag in ("P1V0", "P1V1"):
    d = drift(cells[tag])
    good = d >= 0.12
    ok &= good
    checks.append(f"prior drift {tag} = {d:.4f}  ({'PASS' if good else 'FAIL'})")
for p_tag in ("P0", "P1"):
    dv = np.abs(priors(cells[p_tag + "V1"]) - priors(cells[p_tag + "V0"])).max()
    good = dv < 0.01
    ok &= good
    checks.append(f"prior movement from V-corruption {p_tag} = {dv:.5f}  "
                  f"({'PASS' if good else 'FAIL'})")

# Association stability: per-token correlation of label associations between the
# first and last period, the same instrument for all four cells.
def stability(df, k=300, floor=3):
    first, last = df.period.min(), df.period.max()
    a = df[(df.period == first) & (df.split == "train_pool")]
    b = df[(df.period == last) & (df.split == "train_pool")]
    Ya = np.array([[l in s.split(";") for l in labels] for s in a.labels])
    Yb = np.array([[l in s.split(";") for l in labels] for s in b.labels])
    da = [set(t.split()) for t in a.text]
    db = [set(t.split()) for t in b.text]
    ca, cb = Counter(w for d in da for w in d), Counter(w for d in db for w in d)
    pool = [w for w in ca if ca[w] >= floor and cb.get(w, 0) >= floor]
    assert len(pool) >= 3 * k, f"pool {len(pool)} < 3x top_k {k}"
    mi = {}
    for w in pool:
        x = np.array([w in d for d in da])
        pw = x.mean()
        s = 0.0
        for kk in range(len(labels)):
            pwy, py = (x & (Ya[:, kk] == 1)).mean(), Ya[:, kk].mean()
            if pwy > 0 and pw > 0 and py > 0:
                s += pwy * np.log2(pwy / (pw * py))
        mi[w] = s
    top = sorted(mi, key=mi.get, reverse=True)[:k]
    vals = []
    for w in top:
        xa = np.array([w in d for d in da]); xb = np.array([w in d for d in db])
        aa = (Ya[xa].sum(0) + 1) / (xa.sum() + 2)
        ab = (Yb[xb].sum(0) + 1) / (xb.sum() + 2)
        if aa.std() > 0 and ab.std() > 0:
            vals.append(np.corrcoef(aa, ab)[0, 1])
    return float(np.mean(vals))

# On the primary stream the rho = 0.5 swap gives a stability ratio of 0.935
# (0.844/0.903); the check is that ratio +/- 0.085, within the same P level.
stab = {tag: stability(cells[tag]) for tag in cells}
for p_tag in ("P0", "P1"):
    s0, s1 = stab[p_tag + "V0"], stab[p_tag + "V1"]
    ratio = s1 / s0
    good = 0.85 <= ratio <= 0.995
    ok &= good
    checks.append(f"stability {p_tag}: V0 {s0:.3f} -> V1 {s1:.3f} "
                  f"(ratio {ratio:.3f}, expected band [0.85, 0.995] around "
                  f"the rho = 0.5 ratio 0.935, {'PASS' if good else 'FAIL'})")

print("\nmanipulation checks:")
for c in checks:
    print("  " + c)
print("ALL PASS" if ok else "CHECK FAILED — do not train on the V1 cells")
sys.exit(0 if ok else 1)
