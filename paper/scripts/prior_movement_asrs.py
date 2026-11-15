#!/usr/bin/env python3
"""Split of the refit gain (theta_1 to V) on the 20 frozen-embedding ASRS runs, periods 2 to 7, into a same-scorer
refit on the final model's period-1 test scores, a reweighting of those records to period j's label priors, and the
remainder; seed-block bootstrap, and the same for the three tuned-GloVe runs of results/methods/.
Reads results/asrs_inherited_rule/scores/ and results/exact_fitter/oracle_both.parquet; writes prior_movement_20runs*.
Needs nothing beyond the repository. Its functions from f1_at to split_one are reused by prior_movement_valsplit.py."""
import glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from exact_fitters import exact_th_label, exact_fit_matrix

OUT = Path("results/asrs_inherited_rule"); SC = OUT / "scores"
BOOT, BSEED = 20000, 20260920
N = 8


def f1_at(p, y, th, w=None):
    w = np.ones_like(y, dtype=float) if w is None else w
    pred = p >= th
    tp = (w * (pred & (y == 1))).sum()
    return 2 * tp / ((w * pred).sum() + (w * (y == 1)).sum() + 1e-12)


def half_exact(P, Y, r, S=10):
    def mf1(idx, th):
        pred = P[idx] >= th
        tp = (pred & (Y[idx] == 1)).sum(0)
        return float((2 * tp / (pred.sum(0) + Y[idx].sum(0) + 1e-12)).mean())
    v = []
    for _ in range(S):
        q = r.permutation(len(Y)); h = len(Y) // 2
        A, B = q[:h], q[h:]
        v.append(.5 * (mf1(B, exact_fit_matrix(P[A], Y[A])) + mf1(A, exact_fit_matrix(P[B], Y[B]))))
    return float(np.mean(v))


def split_one(th_dep, P1, Y1, get_PY, V_ext=None):
    """One run: per-period levels on j = 1..6; returns range means."""
    K = Y1.shape[1]
    th_b = np.array([exact_th_label(P1[:, k], Y1[:, k]) for k in range(K)])
    fz, ob, po, rf, per = [], [], [], [], []
    for j in range(1, N - 1):
        Pj, Yj = get_PY(j)
        tme = []
        for k in range(K):
            pi1, pij = Y1[:, k].mean(), Yj[:, k].mean()
            w = np.where(Y1[:, k] == 1, pij / max(pi1, 1e-9), (1 - pij) / max(1 - pi1, 1e-9))
            tme.append(exact_th_label(P1[:, k], Y1[:, k], w))
        fz.append(np.mean([f1_at(Pj[:, k], Yj[:, k], th_dep[k]) for k in range(K)]))
        ob.append(np.mean([f1_at(Pj[:, k], Yj[:, k], th_b[k]) for k in range(K)]))
        po.append(np.mean([f1_at(Pj[:, k], Yj[:, k], tme[k]) for k in range(K)]))
        rf.append(half_exact(Pj, Yj, np.random.default_rng(31 + j)))
        per.append(dict(j=j, frozen_dep=fz[-1], refit_old=ob[-1], prior_only=po[-1], refit_S10=rf[-1],
                        refit_V=(None if V_ext is None else V_ext[j])))
    out = dict(frozen_dep=np.mean(fz), refit_old=np.mean(ob), prior_only=np.mean(po), refit_S10=np.mean(rf),
               refit_V=(None if V_ext is None else float(np.mean([V_ext[j] for j in range(1, N - 1)]))),
               theta_b=th_b.tolist())
    return out, per


if __name__ == "__main__":
    ob = pd.read_parquet("results/exact_fitter/oracle_both.parquet")
    ob = ob[ob.family == "frozen"].set_index(["condition", "seed", "j"])

    # the 20 frozen-embedding runs
    runs, per_rows = [], []
    for f in sorted(SC.glob("*.npz")):
        cond, seed = f.stem.split("_s"); seed = int(seed)
        if cond not in ("C1", "C2"):
            continue
        z = np.load(f)
        P1, Y1 = z["P_final_0"].astype(np.float64), z["Y_0"].astype(np.int64)
        V_ext = {j: float(ob.f1_final_oracle_exact.loc[(cond, seed, j)]) for j in range(N - 1)}
        out, per = split_one(z["theta_dep"], P1, Y1,
                             lambda j: (z[f"P_final_{j}"].astype(np.float64), z[f"Y_{j}"].astype(np.int64)), V_ext)
        runs.append(dict(condition=cond, seed=seed, **out))
        per_rows.append(pd.DataFrame(per).assign(condition=cond, seed=seed))
        print(
            f"  {cond} s{seed}: frozen {out['frozen_dep']:.5f} refit_old {out['refit_old']:.5f} prior_only "
            f"{out['prior_only']:.5f} "
            f"V {out['refit_V']:.5f} S10 {out['refit_S10']:.5f}",
            flush=True,
        )
    a = pd.DataFrame(runs)
    a.drop(columns=["theta_b"]).to_parquet(OUT / "prior_movement_20runs_runs.parquet", index=False)
    pd.concat(per_rows).to_parquet(OUT / "prior_movement_20runs_periods.parquet", index=False)

    def shares(a, den_col):
        a = a.copy()
        a["tot"] = a[den_col] - a.frozen_dep
        a["g_old"] = a.refit_old - a.frozen_dep
        a["g_prior"] = a.prior_only - a.refit_old
        a["g_rem"] = a[den_col] - a.prior_only
        a["g_pub"] = a.prior_only - a.frozen_dep
        return a

    def block_ci(a, cols):
        """Percentile CI of ratio-of-means shares over seed blocks (mean over conditions)."""
        blk = a.groupby("seed")[cols + ["tot"]].mean().to_numpy(); n = len(blk)
        rng = np.random.default_rng(BSEED); dr = np.empty((BOOT, len(cols)))
        for r in range(BOOT):
            k = rng.integers(0, n, n); m = blk[k].mean(0)
            dr[r] = 100 * m[:-1] / m[-1]
        return {
            c: [float(np.percentile(dr[:, i], 2.5)), float(np.percentile(dr[:, i], 97.5))] for i, c in enumerate(cols)
        }, n

    res = dict(
        generator="prior_movement_asrs.py",
        scoring="score_inherited_rule.py",
        range="periods 2-7 (j=1..6)",
        fit_set="final model's period-1 TEST scores (2077 records), exact fitter; "
        "prior weights per label = period-j prior odds",
        bootstrap=dict(
            B=BOOT,
            seed=BSEED,
            blocks="seed (10); block value = mean over C1, C2 of per-run range means; ratio of block means",
        ),
    )
    for den, label in (
        ("refit_V", "V(s_N) from oracle_both (exact split-half B=20)"),
        ("refit_S10", "npz-style split-half S=10, rng 31+j"),
    ):
        s = shares(a, den)
        cols = ["g_old", "g_prior", "g_rem", "g_pub"]
        cis, nb = block_ci(s, cols)
        d = dict(
            denominator=label,
            n_runs=int(len(s)),
            n_blocks=int(nb),
            levels={k: float(s[k].mean()) for k in ("frozen_dep", "refit_old", "prior_only", den)},
            total_gain=float(s.tot.mean()),
            share_same_scorer_refit_pct=float(100 * s.g_old.mean() / s.tot.mean()),
            ci_same_scorer=cis["g_old"],
            share_prior_reweighting_pct=float(100 * s.g_prior.mean() / s.tot.mean()),
            ci_prior=cis["g_prior"],
            share_remainder_pct=float(100 * s.g_rem.mean() / s.tot.mean()),
            ci_remainder=cis["g_rem"],
            share_published_construct_pct=float(100 * s.g_pub.mean() / s.tot.mean()),
            ci_published=cis["g_pub"],
            n_runs_prior_gain_positive=int((s.g_prior > 0).sum()),
            n_runs_old_gain_positive=int((s.g_old > 0).sum()),
            per_condition={
                c: dict(
                    same_scorer=float(100 * g.g_old.mean() / g.tot.mean()),
                    prior=float(100 * g.g_prior.mean() / g.tot.mean()),
                    remainder=float(100 * g.g_rem.mean() / g.tot.mean()),
                    published=float(100 * g.g_pub.mean() / g.tot.mean()),
                )
                for c, g in s.groupby("condition")
            },
        )
        res[f"frozen20_{den}"] = d
        print(f"\n20 runs, denominator {label}:")
        print("  levels", {k: round(v, 6) for k, v in d["levels"].items()}, "total gain", round(d["total_gain"], 6))
        print(
            f"  same-scorer refit {d['share_same_scorer_refit_pct']:.2f}% {np.round(cis['g_old'],1).tolist()} | "
            f"prior reweighting {d['share_prior_reweighting_pct']:.2f}% {np.round(cis['g_prior'],1).tolist()} | "
            f"remainder {d['share_remainder_pct']:.2f}% {np.round(cis['g_rem'],1).tolist()} | prior-only rule "
            f"{d['share_published_construct_pct']:.2f}% {np.round(cis['g_pub'],1).tolist()}"
        )
        print("  per condition:", {c: {k: round(v, 1) for k, v in v_.items()} for c, v_ in d["per_condition"].items()})

    # the three tuned-GloVe runs of results/methods/, same code
    r3 = []
    for fn in sorted(glob.glob("results/methods/*naive_ref*_scores.npz")):
        z = np.load(fn)
        P1, Y1 = z["Y_0"], z[f"P_{N-1}_0"].astype(np.float64)
        out, _ = split_one(z["thresholds"], z[f"P_{N-1}_0"].astype(np.float64), z["Y_0"],
                           lambda j: (z[f"P_{N-1}_{j}"].astype(np.float64), z[f"Y_{j}"]))
        r3.append(
            dict(
                file=fn,
                seed=int(re.search(r"_s(\d+)_", fn).group(1)),
                **{k: v for k, v in out.items() if k not in ("theta_b", "refit_V")},
            )
        )
    b = shares(pd.DataFrame(r3), "refit_S10")
    rng = np.random.default_rng(20260818)
    drs = np.array(
        [
            [
                100 * (q.g_old.mean() / q.tot.mean()),
                100 * (q.g_prior.mean() / q.tot.mean()),
                100 * (q.g_pub.mean() / q.tot.mean()),
            ]
            for q in (b.iloc[rng.integers(0, 3, 3)] for _ in range(BOOT))
        ]
    )
    res["tuned_glove_3runs"] = dict(
        files=[r["file"] for r in r3],
        n_runs=3,
        denominator="split-half S=10, rng 31+j",
        levels={k: float(b[k].mean()) for k in ("frozen_dep", "refit_old", "prior_only", "refit_S10")},
        share_same_scorer_refit_pct=float(100 * b.g_old.mean() / b.tot.mean()),
        ci_same_scorer=[float(np.percentile(drs[:, 0], 2.5)), float(np.percentile(drs[:, 0], 97.5))],
        share_prior_reweighting_pct=float(100 * b.g_prior.mean() / b.tot.mean()),
        ci_prior=[float(np.percentile(drs[:, 1], 2.5)), float(np.percentile(drs[:, 1], 97.5))],
        share_published_construct_pct=float(100 * b.g_pub.mean() / b.tot.mean()),
        ci_published=[float(np.percentile(drs[:, 2], 2.5)), float(np.percentile(drs[:, 2], 97.5))],
        bootstrap="over 3 runs, 20000 resamples, default_rng(20260818)",
        per_run=b[["seed", "frozen_dep", "refit_old", "prior_only", "refit_S10"]].to_dict(orient="records"),
    )
    e = res["tuned_glove_3runs"]
    print(
        f"\n3 tuned-GloVe runs: same-scorer {e['share_same_scorer_refit_pct']:.2f}% "
        f"prior {e['share_prior_reweighting_pct']:.2f}% prior-only rule {e['share_published_construct_pct']:.2f}% "
        f"{np.round(e['ci_published'],1).tolist()}"
    )
    print("   levels:", {k: round(v, 6) for k, v in e["levels"].items()})
    json.dump(res, open(OUT / "prior_movement_20runs.json", "w"), indent=1)
    print("wrote", OUT / "prior_movement_20runs.json")
