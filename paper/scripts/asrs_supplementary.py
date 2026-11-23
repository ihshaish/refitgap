#!/usr/bin/env python3
"""Analyses on the stored ASRS score matrices (frozen embeddings C1 and C2, seeds 0-9, periods 2 to 7):
python scripts/asrs_supplementary.py {recovery,budget,loo,origin,invariance,shared,twolevel}.
Reads results/asrs_supplementary/scores/; each writes results/asrs_supplementary/<name>.json.
Backs Appendix A ("Estimator on known cases", "Finite labelled budgets", "Resampling of records")
and the Appendix D sensitivity. Needs the score files written by store_asrs_scores.py (not shipped)."""
import argparse, glob, json, re, time
import numpy as np
from sklearn.metrics import f1_score
from constructed_drifts import fit_exact
from train_sequential import fit_thresholds
OUT = "results/asrs_supplementary"; J = range(1, 7)
def mf1(P, Y, th): return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)
def pl1(P, Y, th): return f1_score(Y, (P >= th).astype(int), average=None, zero_division=0)
def load():
    runs = []
    for f in sorted(glob.glob(f"{OUT}/scores/primary_C*_s*.npz")):
        m = re.search(r"primary_(C\d)_s(\d+)", f); runs.append((m.group(1), int(m.group(2)), np.load(f)))
    assert runs, "no stored scores"; return runs
def halves(n, rng): perm = rng.permutation(n); h = n // 2; return perm[:h], perm[h:]
def Vsplit(PN, Pj, Y, rng, M=200, frac=1.0, per_label=False):
    vN, vj = [], []
    for _ in range(M):
        a, b = halves(len(Y), rng)
        ca, cb = (a[: max(10, int(len(a) * frac))], b[: max(10, int(len(b) * frac))]) if frac < 1 else (a, b)
        f = pl1 if per_label else mf1
        vN.append(0.5 * (f(PN[b], Y[b], fit_exact(PN[ca], Y[ca])) + f(PN[a], Y[a], fit_exact(PN[cb], Y[cb]))))
        vj.append(0.5 * (f(Pj[b], Y[b], fit_exact(Pj[ca], Y[ca])) + f(Pj[a], Y[a], fit_exact(Pj[cb], Y[cb]))))
    return np.mean(vN, axis=0), np.mean(vj, axis=0)
def boot(blk, rng=None, B=20000):
    rng = rng or np.random.default_rng(20260920)
    n = len(blk)
    d = np.array([blk[rng.integers(0, n, n)].mean(0) for _ in range(B)])
    return np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)
def cells(z, j):
    N = 8
    return (
        z[f"P_{N-1}_{j}"].astype(np.float64),
        z[f"P_{j}_{j}"].astype(np.float64),
        z[f"Y_{j}"].astype(int),
        z["theta_dep"].astype(np.float64),
    )

def recovery():
    """L = max(0, U_ref - V_N): what is left after the final model refits, against the reference at theta_1."""
    rows = []
    for cond, seed, z in load():
        for j in J:
            PN, Pj, Y, th = cells(z, j); vN, vj = Vsplit(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j))
            Uref, UN = mf1(Pj, Y, th), mf1(PN, Y, th)
            rows.append(
                dict(
                    cond=cond,
                    seed=seed,
                    j=j,
                    U_ref=Uref,
                    U_N=UN,
                    V_N=float(vN),
                    V_ref=float(vj),
                    B=Uref - UN,
                    A=float(vj - vN),
                    R_N=float(vN - UN),
                    R_ref=float(vj - Uref),
                    L=max(0.0, Uref - float(vN)),
                    VN_minus_Uref=float(vN) - Uref,
                )
            )
    import pandas as pd

    t = pd.DataFrame(rows)
    blk = t.groupby("seed")[["B", "A", "R_N", "R_ref", "L", "VN_minus_Uref"]].mean()
    lo, hi = boot(blk.to_numpy())
    per_j = t.groupby("j")[["U_ref", "V_N", "L", "VN_minus_Uref"]].mean().round(5).to_dict(orient="index")
    out = dict(
        n_obs=int(t.groupby(["cond", "seed"]).ngroups),
        means={c: float(blk[c].mean()) for c in blk},
        ci={c: [float(lo[i]), float(hi[i])] for i, c in enumerate(blk)},
        cells_with_L_positive=int((t.L > 0).sum()),
        cells=len(t),
        per_period=per_j,
    )
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/recovery_target.json", "w"), indent=1)
    t.to_parquet(f"{OUT}/recovery_target.parquet", index=False)

def budget():
    """Attainable levels and A, R_N, R_ref with the calibration half subsampled."""
    fr = [0.1, 0.25, 0.5, 0.75, 1.0]; acc = {f: [] for f in fr}
    for cond, seed, z in load():
        for j in J:
            PN, Pj, Y, th = cells(z, j); UN, Uref = mf1(PN, Y, th), mf1(Pj, Y, th)
            for f in fr:
                vN, vj = Vsplit(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j), M=50, frac=f)
                acc[f].append(
                    dict(
                        seed=seed,
                        V_N=float(vN),
                        V_ref=float(vj),
                        A=float(vj - vN),
                        R_N=float(vN - UN),
                        R_ref=float(vj - Uref),
                    )
                )
    import pandas as pd; out = {}
    for f in fr:
        t = pd.DataFrame(acc[f])
        blk = t.groupby("seed")[["V_N", "V_ref", "A", "R_N", "R_ref"]].mean()
        lo, hi = boot(blk.to_numpy())
        out[f] = dict(
            calibration_records_per_period=f"about {int(1000*f)} (half of a test period x fraction)",
            means={c: float(blk[c].mean()) for c in blk},
            ci_A=[float(lo[2]), float(hi[2])],
        )
        print(
            f"frac {f}: V_N {blk.V_N.mean():.4f} V_ref {blk.V_ref.mean():.4f} A {blk.A.mean():+.4f} "
            f"[{lo[2]:+.4f},{hi[2]:+.4f}] R_N {blk.R_N.mean():.4f} R_ref {blk.R_ref.mean():.4f}",
            flush=True,
        )
    json.dump(out, open(f"{OUT}/budget_sensitivity.json", "w"), indent=1)

def loo():
    """Leave-one-label-out B, A and dR."""
    import pandas as pd; rows = []
    for cond, seed, z in load():
        labels = [str(l) for l in z["labels"]]
        for j in J:
            PN, Pj, Y, th = cells(z, j)
            vN, vj = Vsplit(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j), per_label=True)
            uN, uref = pl1(PN, Y, th), pl1(Pj, Y, th)
            for k in range(-1, Y.shape[1]):
                keep = [i for i in range(Y.shape[1]) if i != k]
                rows.append(
                    dict(
                        seed=seed,
                        j=j,
                        dropped=("none" if k < 0 else labels[k]),
                        B=float(uref[keep].mean() - uN[keep].mean()),
                        A=float(vj[keep].mean() - vN[keep].mean()),
                        R_N=float(vN[keep].mean() - uN[keep].mean()),
                        R_ref=float(vj[keep].mean() - uref[keep].mean()),
                    )
                )
    t = pd.DataFrame(rows); t["dR"] = t.R_N - t.R_ref; out = {}
    for d, g in t.groupby("dropped"):
        blk = g.groupby("seed")[["B", "A", "dR"]].mean(); lo, hi = boot(blk.to_numpy())
        out[d] = dict(
            B=float(blk.B.mean()),
            A=float(blk.A.mean()),
            dR=float(blk.dR.mean()),
            dR_over_B=float(blk.dR.mean() / blk.B.mean()),
            ci_dR=[float(lo[2]), float(hi[2])],
            ci_A=[float(lo[1]), float(hi[1])],
        )
        print(
            f"drop {d[:28]:28s} B {blk.B.mean():+.4f} A {blk.A.mean():+.4f} dR {blk.dR.mean():+.4f} share "
            f"{blk.dR.mean()/blk.B.mean():.2f}",
            flush=True,
        )
    json.dump(out, open(f"{OUT}/leave_one_label_out.json", "w"), indent=1)

def origin(R=100):
    """theta_1 refitted on resamples of the period-1 validation split, and rules set at later origin periods."""
    import pandas as pd; rows = []
    for cond, seed, z in load():
        Pv, Yv = z["Pval_0"].astype(np.float64), z["Yval"].astype(int); rng = np.random.default_rng(31000 + seed)
        cell = [cells(z, j) for j in J]
        vals = {j: Vsplit(c[0], c[1], c[2], np.random.default_rng(7000 + 100 * seed + j)) for j, c in zip(J, cell)}
        P00, Y00 = z["P_0_0"].astype(np.float64), z["Y_0"].astype(
            int
        )  # period-1 model on period-1 test records: calibration records unused so far
        for kind, src, draws in (
            ("bootstrap", (Pv, Yv), [rng.integers(0, len(Yv), len(Yv)) for _ in range(R)]),
            ("half", (Pv, Yv), [rng.choice(len(Yv), len(Yv) // 2, False) for _ in range(R)]),
            ("quarter", (Pv, Yv), [rng.choice(len(Yv), len(Yv) // 4, False) for _ in range(R)]),
            (
                "fresh_matched_budget",
                (P00, Y00),
                [rng.choice(len(Y00), min(len(Yv), len(Y00)), False) for _ in range(R)],
            ),
        ):
            for r, idx in enumerate(draws):
                th = fit_thresholds(src[0][idx], src[1][idx]); B, RN, Rr = [], [], []
                for j, (PN, Pj, Y, _) in zip(J, cell):
                    vN, vj = vals[j]
                    UN, Uref = mf1(PN, Y, th), mf1(Pj, Y, th)
                    B.append(Uref - UN)
                    RN.append(vN - UN)
                    Rr.append(vj - Uref)
                rows.append(
                    dict(
                        cond=cond,
                        seed=seed,
                        kind=kind,
                        r=r,
                        B=np.mean(B),
                        R_N=np.mean(RN),
                        R_ref=np.mean(Rr),
                        dR=np.mean(RN) - np.mean(Rr),
                    )
                )
        print(cond, seed, "done", flush=True)
    t = pd.DataFrame(rows); out = {}
    for kind, g in t.groupby("kind"):
        m = g.groupby(["cond", "seed"])[["B", "R_N", "R_ref", "dR"]].mean()   # mean over resamples per observation
        sd = g.groupby(["cond", "seed"])[["B", "dR"]].std().mean()  # within-observation spread over theta_1 draws
        out[kind] = dict(
            B=float(m.B.mean()),
            R_N=float(m.R_N.mean()),
            R_ref=float(m.R_ref.mean()),
            dR=float(m.dR.mean()),
            sd_B_over_theta_draws=float(sd.B),
            sd_dR_over_theta_draws=float(sd.dR),
            share_dR_over_B=float(m.dR.mean() / m.B.mean()),
        )
        print(kind, {k: round(v, 4) for k, v in out[kind].items()})
    # origin period: thresholds fitted at period p (model p, grid fitter, half of its test records), carried to j > p
    rows2 = []
    for cond, seed, z in load():
        rng = np.random.default_rng(32000 + seed)
        for p in (0, 1, 2, 3):
            Pp, Yp = z[f"P_{p}_{p}"].astype(np.float64), z[f"Y_{p}"].astype(int)
            a, b = halves(len(Yp), rng)
            th = fit_thresholds(Pp[a], Yp[a])
            for j in range(p + 1, 7):
                PN, Pj, Y, _ = cells(z, j)
                vN, vj = Vsplit(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j))
                UN, Uref = mf1(PN, Y, th), mf1(Pj, Y, th)
                rows2.append(
                    dict(seed=seed, origin=p, j=j, B=Uref - UN, A=float(vj - vN), dR=float((vN - UN) - (vj - Uref)))
                )
    t2 = pd.DataFrame(rows2); out["origin_period"] = {}
    for p, g in t2.groupby("origin"):
        blk = g.groupby("seed")[["B", "A", "dR"]].mean()
        out["origin_period"][int(p)] = dict(
            periods_evaluated=f"{p+1}..6",
            B=float(blk.B.mean()),
            A=float(blk.A.mean()),
            dR=float(blk.dR.mean()),
            share=float(blk.dR.mean() / blk.B.mean()),
        )
        print(
            "origin period",
            p,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["origin_period"][int(p)].items()},
        )
    json.dump(out, open(f"{OUT}/threshold_origin.json", "w"), indent=1)

def invariance():
    """Known cases: identical scorers, strictly increasing per-label rescoring, shuffled scores."""
    import pandas as pd; rows = []; rng0 = np.random.default_rng(20261013)
    gam = rng0.uniform(
        0.4, 2.5, 9
    )  # per-label strictly increasing maps p -> p**gamma (no clipping, ties preserved exactly)
    def mono(P): return np.power(P, gam)
    def order_preserved(P, Q):   # same ordering and the same tie groups, per label
        return all(
            np.array_equal(np.argsort(P[:, k], kind="mergesort"), np.argsort(Q[:, k], kind="mergesort"))
            and np.array_equal(np.unique(P[:, k], return_inverse=True)[1], np.unique(Q[:, k], return_inverse=True)[1])
            for k in range(P.shape[1])
        )
    checks = []
    for cond, seed, z in load():
        for j in J:
            PN, Pj, Y, th = cells(z, j)
            rng = np.random.default_rng(7000 + 100 * seed + j)
            st = rng.bit_generator.state
            vN, vj = Vsplit(PN, Pj, Y, rng)
            rng.bit_generator.state = st
            vI, _ = Vsplit(PN, PN, Y, rng)  # identity: same scorer twice
            PM = mono(PN); checks.append(order_preserved(PN, PM))
            rng.bit_generator.state = st
            vM, _ = Vsplit(PM, Pj, Y, rng)  # monotone transform of the final model
            rng.bit_generator.state = st
            vR, _ = Vsplit(PN[rng.permutation(len(Y))], Pj, Y, rng)  # randomised: rows shuffled
            rows.append(
                dict(
                    seed=seed,
                    j=j,
                    A=float(vj - vN),
                    B=mf1(Pj, Y, th) - mf1(PN, Y, th),
                    identity_A=float(vN - vI),
                    identity_B=mf1(PN, Y, th) - mf1(PN, Y, th),
                    mono_A=float(vj - vM),
                    mono_B=mf1(Pj, Y, th) - mf1(PM, Y, th),
                    mono_V_change=float(vM - vN),
                    identity_RN_minus_Rref=float((vN - mf1(PN, Y, th)) - (vI - mf1(PN, Y, th))),
                    random_A=float(vj - vR),
                    random_B=mf1(Pj, Y, th) - mf1(PN[np.random.default_rng(1).permutation(len(Y))], Y, th),
                )
            )
    t = pd.DataFrame(rows)
    out = {
        c: dict(mean=float(t[c].mean()), max_abs=float(t[c].abs().max())) for c in t.columns if c not in ("seed", "j")
    }
    out["transform"] = dict(gamma=gam.round(3).tolist(), ordering_and_ties_preserved_in_all_cells=bool(all(checks)))
    out["criteria"] = dict(
        identity="B, A, dR exactly zero (max|.| = 0.0); R_N = R_ref",
        monotone="V change exactly zero so mono_A = A exactly; mono_B and dR differ",
        randomised="random_A large and positive",
    )
    for k, v in out.items():
        if isinstance(v, dict) and "mean" in v: print(f"{k:16s} mean {v['mean']:+.5f} max|.| {v['max_abs']:.5f}")
    json.dump(out, open(f"{OUT}/invariance_checks.json", "w"), indent=1)

def shared(M=50):
    """One threshold vector fitted on the pooled calibration halves of periods 2 to 7."""
    import pandas as pd; rows = []
    for cond, seed, z in load():
        cell = {j: cells(z, j) for j in J}; rng = np.random.default_rng(9000 + seed)
        acc = {j: dict(U_dep=0, U_shared=0, U_shared_mean=0, V_per=0) for j in J}
        for m in range(M):
            H = {j: halves(len(cell[j][2]), rng) for j in J}
            for side in (0, 1):   # fit on side, evaluate on the other
                fitP = np.vstack([cell[j][0][H[j][side]] for j in J])
                fitY = np.vstack([cell[j][2][H[j][side]] for j in J])
                th_sh = fit_exact(fitP, fitY)
                # second objective: the mean over periods of period macro-F1, over 300 quantile cutpoints per label
                th_mean = np.empty(fitP.shape[1])
                for k in range(fitP.shape[1]):
                    cands = np.unique(np.quantile(fitP[:, k], np.linspace(0.001, 0.999, 300)))
                    per = []
                    for j in J:   # F1 at every candidate at once via searchsorted
                        pk = cell[j][0][H[j][side]][:, k]; yk = cell[j][2][H[j][side]][:, k]
                        pos = np.sort(pk[yk == 1]); neg = np.sort(pk[yk == 0]); npos = len(pos)
                        tp = npos - np.searchsorted(pos, cands, side="left")
                        fp = len(neg) - np.searchsorted(neg, cands, side="left")
                        per.append(
                            np.where(
                                2 * tp + fp + (npos - tp) > 0, 2 * tp / np.maximum(2 * tp + fp + (npos - tp), 1), 0.0
                            )
                        )
                    th_mean[k] = cands[int(np.argmax(np.mean(per, axis=0)))]
                for j in J:
                    PN, Pj, Y, th = cell[j]; ev = H[j][1 - side]; fit = H[j][side]
                    acc[j]["U_dep"] += mf1(PN[ev], Y[ev], th) / (2 * M)
                    acc[j]["U_shared"] += mf1(PN[ev], Y[ev], th_sh) / (2 * M)
                    acc[j]["U_shared_mean"] += mf1(PN[ev], Y[ev], th_mean) / (2 * M)
                    acc[j]["V_per"] += mf1(PN[ev], Y[ev], fit_exact(PN[fit], Y[fit])) / (2 * M)
        for j in J: rows.append(dict(cond=cond, seed=seed, j=j, **acc[j]))
        print(cond, seed, "done", flush=True)
    t = pd.DataFrame(rows); blk = t.groupby("seed")[["U_dep", "U_shared", "U_shared_mean", "V_per"]].mean()
    g1, g2, g3 = blk.U_shared - blk.U_dep, blk.V_per - blk.U_shared, blk.U_shared_mean - blk.U_dep
    lo, hi = boot(np.column_stack([g1, g2, g3]))
    out = dict(
        budget="per-period rules: one fitting half (about 1000 records) each; shared rules: the six pooled halves; "
        "evaluation on the other halves, both directions",
        U_inherited=float(blk.U_dep.mean()),
        U_shared_rule_pooled_objective=float(blk.U_shared.mean()),
        U_shared_rule_mean_of_periods_objective=float(blk.U_shared_mean.mean()),
        V_per_period=float(blk.V_per.mean()),
        gain_shared_pooled_over_inherited=float(g1.mean()),
        ci=[float(lo[0]), float(hi[0])],
        gain_shared_mean_over_inherited=float(g3.mean()),
        ci_mean=[float(lo[2]), float(hi[2])],
        gap_per_period_over_shared_pooled=float(g2.mean()),
        ci_gap=[float(lo[1]), float(hi[1])],
    )
    print(json.dumps(out, indent=1)); json.dump(out, open(f"{OUT}/shared_rule.json", "w"), indent=1)

def twolevel(R=200, M=20, pilot=False):
    """Crossed bootstrap over initialisations and, per period, evaluation records; one record draw is
    shared by the selected initialisations and both scorers. Records are split into halves before
    resampling within each half, so no record sits on both sides of a fit."""
    runs = load(); seeds = sorted({s for _, s, _ in runs}); rng = np.random.default_rng(20260920); reps = []
    t0 = time.time(); R = 20 if pilot else R
    for r in range(R):
        sdraw = rng.choice(seeds, len(seeds), True)
        draws = {}
        for j in J:
            n = len(runs[0][2][f"Y_{j}"]); d = []
            for m in range(M):
                a, b = halves(n, rng)
                d.append(
                    (rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                )  # split first, then resample within halves
            draws[j] = d
        B, A, dR = [], [], []
        for s in sdraw:
            for cond, seed, z in runs:
                if seed != s: continue
                for j in J:
                    PN, Pj, Y, th = cells(z, j); vN = vj = 0.0; uN = uref = 0.0
                    for (ia, ib) in draws[j]:
                        vN += (
                            0.5
                            * (
                                mf1(PN[ib], Y[ib], fit_exact(PN[ia], Y[ia]))
                                + mf1(PN[ia], Y[ia], fit_exact(PN[ib], Y[ib]))
                            )
                            / M
                        )
                        vj += (
                            0.5
                            * (
                                mf1(Pj[ib], Y[ib], fit_exact(Pj[ia], Y[ia]))
                                + mf1(Pj[ia], Y[ia], fit_exact(Pj[ib], Y[ib]))
                            )
                            / M
                        )
                        ii = np.concatenate([ia, ib])
                        uN += mf1(PN[ii], Y[ii], th) / M
                        uref += mf1(Pj[ii], Y[ii], th) / M
                    B.append(uref - uN); A.append(vj - vN); dR.append((vN - uN) - (vj - uref))
        reps.append((np.mean(B), np.mean(A), np.mean(dR)))
        if r % 10 == 0:
            print(
                f"rep {r} B {np.mean(B):+.4f} A {np.mean(A):+.4f} dR {np.mean(dR):+.4f} ({time.time()-t0:.0f}s)",
                flush=True,
            )
    reps = np.array(reps)
    out = {
        n: dict(
            mean=float(reps[:, i].mean()),
            ci=[float(np.percentile(reps[:, i], 2.5)), float(np.percentile(reps[:, i], 97.5))],
            sd=float(reps[:, i].std()),
        )
        for i, n in enumerate(["B", "A", "dR"])
    }
    out["share"] = dict(
        mean=float((reps[:, 2] / reps[:, 0]).mean()),
        ci=[float(np.percentile(reps[:, 2] / reps[:, 0], 2.5)), float(np.percentile(reps[:, 2] / reps[:, 0], 97.5))],
    )
    out["settings"] = dict(R=R, M=M, pilot=pilot)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/two_level_bootstrap{'_pilot_M'+str(M) if pilot else ''}.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["recovery", "budget", "loo", "origin", "invariance", "shared", "twolevel"])
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.what == "twolevel": twolevel(M=a.M, pilot=a.pilot)
    else: globals()[a.what]()
