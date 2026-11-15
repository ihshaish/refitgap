#!/usr/bin/env python3
"""Predictions 8 to 11 of the matched-magnitude pair (stronger prior reweighting against the
half-dose pairing) over the repeated constructions; see plans/repeated-constructions.txt.
Reads results/constructions/construction_seeds_summary.json; prints only.
Backs Appendix E, "Repeated constructions". Needs nothing beyond the repository."""
import json, numpy as np
S = json.load(open("results/constructions/construction_seeds_summary.json"))["per_construction"]
prior = {v["construction_seed"]: v for v in S.values() if v["kind"] == "prior rho=1.0"}
vocab = {v["construction_seed"]: v for v in S.values() if v["kind"] == "A rho=0.5"}
seeds = sorted(set(prior) & set(vocab)); n = len(seeds)
print(f"constructions with both arms: {n}")
print("seed      D_prior  A_prior  R_prior  R-A   | D_vocab  A_vocab  R_vocab  A-R")
RA = []; AR = []
for s in seeds:
    p, v = prior[s], vocab[s]; RA.append(p["R"] - p["A"]); AR.append(v["A"] - v["R"])
    print(
        f"{s}  {p['D']:+.4f}  {p['A']:+.4f}  {p['R']:+.4f}  {RA[-1]:+.4f} | {v['D']:+.4f}  {v['A']:+.4f}  "
        f"{v['R']:+.4f}  {AR[-1]:+.4f}"
    )
RA, AR = np.array(RA), np.array(AR)
rng = np.random.default_rng(20260920)
def ci(x):
    b = np.array([x[rng.integers(0, n, n)].mean() for _ in range(20000)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
print(f"\nP8  prior R-A > 0 in {int((RA > 0).sum())}/{n}; vocab A-R > 0 in {int((AR > 0).sum())}/{n}")
print(
    f"P9  mean R-A prior {RA.mean():+.4f} [{ci(RA)[0]:+.4f}, {ci(RA)[1]:+.4f}]; mean A-R vocab {AR.mean():+.4f} "
    f"[{ci(AR)[0]:+.4f}, {ci(AR)[1]:+.4f}]"
)
print(
    f"P10 falsifier fires: prior A>R in {[s for s in seeds if prior[s]['A'] > prior[s]['R']]}; vocab R>A in "
    f"{[s for s in seeds if vocab[s]['R'] > vocab[s]['A']]}"
)
for name, d in (("prior", prior), ("vocab", vocab)):
    A = np.array([d[s]["A"] for s in seeds])
    R = np.array([d[s]["R"] for s in seeds])
    D = np.array([d[s]["D"] for s in seeds])
    print(
        f"P11 {name}: sd A {A.std():.4f} sd R {R.std():.4f} sd A/D {(A/D).std():.3f} sd R/D {(R/D).std():.3f}; D "
        f"mean {D.mean():.4f} range {D.min():.4f}..{D.max():.4f}; mean A/D {np.mean(A/D):.3f} mean R/D "
        f"{np.mean(R/D):.3f}"
    )
