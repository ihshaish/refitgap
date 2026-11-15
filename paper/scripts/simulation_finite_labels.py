#!/usr/bin/env python3
"""Simulation with known optima: nine labels with ASRS prevalences and Beta scores. The finite-budget level V_n against
the population V, the split-half estimate at 2000 test records, the estimated gap to a scorer with separations
reduced by 5%, and the gap under an order-preserving rescoring (population gap zero).
Writes results/asrs_supplementary/simulation_Vn.json. Backs Appendix A, "Finite labelled budgets" (first simulation).
Needs nothing beyond the repository."""
import json, numpy as np
from sklearn.metrics import f1_score
from constructed_drifts import fit_exact
rng = np.random.default_rng(20261013)
prev = np.array([0.05, 0.07, 0.08, 0.14, 0.16, 0.24, 0.25, 0.48, 0.51]); K = len(prev)
sep = np.array([1.2, 1.0, 1.4, 1.1, 1.3, 0.9, 1.0, 1.2, 1.1])   # separation of the two score distributions
def draw(n):
    Y = (rng.random((n, K)) < prev).astype(int); P = np.empty((n, K))
    for k in range(K):
        a1, b1 = 2 + sep[k], 2; a0, b0 = 2, 2 + sep[k]
        P[:, k] = np.where(Y[:, k] == 1, rng.beta(a1, b1, n), rng.beta(a0, b0, n))
    return P, Y
def mf1(P, Y, th): return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)
def pf1(P, Y, th): return f1_score(Y, (P >= th).astype(int), average=None, zero_division=0)
Pbig, Ybig = draw(400_000); th_pop = fit_exact(Pbig, Ybig)
Ptest, Ytest = draw(100_000); V_pop = mf1(Ptest, Ytest, th_pop); pl_pop = pf1(Ptest, Ytest, th_pop)
out = dict(V_pop=float(V_pop), prev=prev.tolist(), per_label_V_pop=pl_pop.tolist(), budgets={})
for n in [50, 100, 200, 500, 1000, 2000, 5000]:
    vals, pl = [], []
    for _ in range(200):
        Pc, Yc = draw(n); th = fit_exact(Pc, Yc); vals.append(mf1(Ptest, Ytest, th)); pl.append(pf1(Ptest, Ytest, th))
    vals = np.array(vals); pl = np.array(pl)
    out["budgets"][n] = dict(
        V_n=float(vals.mean()),
        bias=float(vals.mean() - V_pop),
        sd=float(vals.std()),
        bias_rarest=float(pl[:, 0].mean() - pl_pop[0]),
        bias_commonest=float(pl[:, -1].mean() - pl_pop[-1]),
    )
    print(
        f"n={n:5d}  V_n {vals.mean():.4f}  bias {vals.mean()-V_pop:+.4f} (sd {vals.std():.4f})  rarest "
        f"{pl[:,0].mean()-pl_pop[0]:+.4f}  commonest {pl[:,-1].mean()-pl_pop[-1]:+.4f}"
    )
# split-half at the paper's test-set size
vals = []
for _ in range(200):
    P, Y = draw(2000); perm = rng.permutation(2000); a, b = perm[:1000], perm[1000:]
    vals.append(0.5 * (mf1(P[b], Y[b], fit_exact(P[a], Y[a])) + mf1(P[a], Y[a], fit_exact(P[b], Y[b]))))
out["split_half_2000"] = dict(V_hat=float(np.mean(vals)), bias=float(np.mean(vals) - V_pop), sd=float(np.std(vals)))
print(
    f"split-half (test 2000, calibration 1000): {np.mean(vals):.4f}  bias {np.mean(vals)-V_pop:+.4f}  (V_pop "
    f"{V_pop:.4f})"
)
json.dump(out, open("results/asrs_supplementary/simulation_Vn.json", "w"), indent=1)

# the gap between two scorers: a second scorer with separations reduced by 5%
sep2 = sep * 0.95
def draw2(n):
    Y = (rng.random((n, K)) < prev).astype(int); P = np.empty((n, K)); P2 = np.empty((n, K))
    for k in range(K):
        u = rng.random(n)  # shared latent draw so the two scorers are paired per record
        from scipy.stats import beta as B
        P[:, k] = np.where(Y[:, k] == 1, B.ppf(u, 2 + sep[k], 2), B.ppf(u, 2, 2 + sep[k]))
        P2[:, k] = np.where(Y[:, k] == 1, B.ppf(u, 2 + sep2[k], 2), B.ppf(u, 2, 2 + sep2[k]))
    return P, P2, Y
Pb, P2b, Yb = draw2(400_000); thA, thB = fit_exact(Pb, Yb), fit_exact(P2b, Yb)
Pt, P2t, Yt = draw2(100_000); A_pop = mf1(Pt, Yt, thA) - mf1(P2t, Yt, thB)
est = []
for _ in range(200):
    P, P2, Y = draw2(2000); perm = rng.permutation(2000); a, b = perm[:1000], perm[1000:]
    vA = 0.5 * (mf1(P[b], Y[b], fit_exact(P[a], Y[a])) + mf1(P[a], Y[a], fit_exact(P[b], Y[b])))
    vB = 0.5 * (mf1(P2[b], Y[b], fit_exact(P2[a], Y[a])) + mf1(P2[a], Y[a], fit_exact(P2[b], Y[b])))
    est.append(vA - vB)
out["A_difference"] = dict(
    A_pop=float(A_pop), A_hat_mean=float(np.mean(est)), bias=float(np.mean(est) - A_pop), sd=float(np.std(est))
)
print(
    f"A: population {A_pop:+.4f}, split-half estimate {np.mean(est):+.4f} (bias {np.mean(est)-A_pop:+.4f}, sd "
    f"{np.std(est):.4f})"
)
json.dump(out, open("results/asrs_supplementary/simulation_Vn.json", "w"), indent=1)

# a second scorer with the same ranking, scores squeezed towards 0.5 by an order-preserving map
def squeeze(P, c=0.35): return 0.5 + np.sign(P - 0.5) * (np.abs(P - 0.5) ** (1 / c)) * (0.5 ** (1 - 1 / c))
vals = []
for _ in range(200):
    P, Y = draw(2000); P2 = squeeze(P); perm = rng.permutation(2000); a, b = perm[:1000], perm[1000:]
    vA = 0.5 * (mf1(P[b], Y[b], fit_exact(P[a], Y[a])) + mf1(P[a], Y[a], fit_exact(P[b], Y[b])))
    vB = 0.5 * (mf1(P2[b], Y[b], fit_exact(P2[a], Y[a])) + mf1(P2[a], Y[a], fit_exact(P2[b], Y[b])))
    vals.append(vA - vB)
out["A_difference_order_preserving_rescoring"] = dict(
    A_pop=0.0,
    A_hat_mean=float(np.mean(vals)),
    sd=float(np.std(vals)),
    note="population A is exactly zero; the exact cutpoint search is invariant to the map, "
    "so any nonzero value is tie/precision effect",
)
print(
    f"A under an order-preserving rescoring of the same scorer: population 0, estimate {np.mean(vals):+.5f} (sd "
    f"{np.std(vals):.5f})"
)
json.dump(out, open("results/asrs_supplementary/simulation_Vn.json", "w"), indent=1)
