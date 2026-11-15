#!/usr/bin/env python3
"""Decisions changed by a threshold update on 500 labelled pool records: records changed and net cells
corrected per 1000 test records, at theta_1 (period-1 model) and at the same-scorer rule (final model),
whose three runs must reproduce results/operational_counts.csv. Reads results/asrs_inherited_rule/scores/
(score_inherited_rule.py); writes operational_counts_deployed{.csv,_summary.json} there.
Backs Appendix D, "Threshold update with few labels". Needs nothing beyond the repository."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from train_sequential import fit_thresholds

OUT = Path("results/asrs_inherited_rule"); SC = OUT / "scores"
BOOT, BSEED = 20000, 20260920
N = 8
orig = pd.read_csv("results/operational_counts.csv")
ORIG3 = sorted({(r.cond, int(r.seed)) for r in orig.itertuples()})


def counts(Pt, Yt, th_frozen, th_new):
    Df, Dn = (Pt >= th_frozen).astype(int), (Pt >= th_new).astype(int)
    ch = Df != Dn
    right = int(((Df != Yt) & (Dn == Yt) & ch).sum()); wrong = int(((Df == Yt) & (Dn != Yt) & ch).sum())
    return dict(n_test=len(Yt), rec_changed_per_1000=1000 * ch.any(1).mean(), cells_changed=int(ch.sum()),
                became_right=right, became_wrong=wrong, net_corrected=right - wrong,
                net_corrected_per_1000=1000 * (right - wrong) / len(Yt))


rows = []
for f in sorted(SC.glob("*.npz")):
    cond, seed = f.stem.split("_s"); seed = int(seed)
    z = np.load(f)
    th_dep, th_fin = z["theta_dep"], z["theta_final"]
    for j in range(N):
        Pt, Yt = z[f"P_final_{j}"].astype(np.float32), z[f"Y_{j}"].astype(np.int64)
        th_new = fit_thresholds(z[f"P_pool500_{j}"].astype(np.float32), z[f"Y_pool500_{j}"].astype(np.int64))
        for foot, th in (("dep", th_dep), ("art", th_fin)):
            rows.append(dict(cond=cond, seed=seed, period=j, footing=foot, **counts(Pt, Yt, th, th_new),
                             n_labels_th_new_ne_frozen=int((th_new != th).sum())))
t = pd.DataFrame(rows)
t.to_csv(OUT / "operational_counts_deployed.csv", index=False)

# the same-scorer rule must reproduce the stored CSV on its three runs, all 8 periods
g = t[(t.footing == "art")].merge(orig, on=["cond", "seed", "period"], suffixes=("", "_csv"))
check = dict(rows=int(len(g)), expected=int(len(orig)),
            max_abs_rec_changed=float((g.rec_changed_per_1000 - g.rec_changed_per_1000_csv).abs().max()),
            max_abs_net=float((g.net_corrected_per_1000 - g.net_corrected_per_1000_csv).abs().max()),
            max_abs_cells=int((g.cells_changed - g.cells_changed_csv).abs().max()))
check["passed"] = bool(
    check["rows"] == check["expected"] and check["max_abs_rec_changed"] < 1e-9 and check["max_abs_net"] < 1e-9
)
print("theta_1~ against operational_counts.csv:", check)


def summarise(sub, label, ci=False):
    p = sub[sub.period >= 1]
    out = dict(
        runs=sorted((r.cond, int(r.seed)) for r in p[["cond", "seed"]].drop_duplicates().itertuples()),
        n_runs=int(p.groupby(["cond", "seed"]).ngroups),
        footing="periods 2-8 (j>=1)",
        rec_changed_per_1000=float(p.rec_changed_per_1000.mean()),
        net_corrected_per_1000=float(p.net_corrected_per_1000.mean()),
        all_periods=dict(
            rec_changed_per_1000=float(sub.rec_changed_per_1000.mean()),
            net_corrected_per_1000=float(sub.net_corrected_per_1000.mean()),
        ),
        by_period={
            int(j): dict(rec=float(v.rec_changed_per_1000), net=float(v.net_corrected_per_1000))
            for j, v in sub.groupby("period")[["rec_changed_per_1000", "net_corrected_per_1000"]].mean().iterrows()
        },
    )
    if ci:
        obs = p.groupby(["cond", "seed"])[["rec_changed_per_1000", "net_corrected_per_1000"]].mean()
        blk = obs.groupby(level="seed").mean().to_numpy(); n = len(blk)
        rng = np.random.default_rng(BSEED); dr = np.empty((BOOT, 2))
        for r in range(BOOT):
            dr[r] = blk[rng.integers(0, n, n)].mean(0)
        out["bootstrap"] = dict(B=BOOT, seed=BSEED, blocks=int(n))
        out["rec_changed_ci"] = [float(np.percentile(dr[:, 0], 2.5)), float(np.percentile(dr[:, 0], 97.5))]
        out["net_corrected_ci"] = [float(np.percentile(dr[:, 1], 2.5)), float(np.percentile(dr[:, 1], 97.5))]
        out["n_runs_net_positive"] = int((obs.net_corrected_per_1000 > 0).sum())
        out["per_run_net"] = {f"{c}_s{s}": float(v) for (c, s), v in obs.net_corrected_per_1000.items()}
    print(
        f"{label:26s} runs {out['n_runs']:2d}: changed {out['rec_changed_per_1000']:.4f}/1000  net "
        f"{out['net_corrected_per_1000']:+.4f}/1000"
        + (
            f"  CI net [{out['net_corrected_ci'][0]:+.3f}, {out['net_corrected_ci'][1]:+.3f}]  CI changed "
            f"[{out['rec_changed_ci'][0]:.2f}, {out['rec_changed_ci'][1]:.2f}]  runs net>0 "
            f"{out['n_runs_net_positive']}/{out['n_runs']}"
            if ci
            else ""
        )
    )
    return out


is3 = t.apply(lambda r: (r.cond, r.seed) in ORIG3, axis=1)
fro = t.cond.isin(["C1", "C2"])
res = dict(generator="operational_counts.py", scoring="score_inherited_rule.py", reproduction_check=check,
           orig3_art=summarise(t[is3 & (t.footing == "art")], "orig3 theta_1~"),
           orig3_dep=summarise(t[is3 & (t.footing == "dep")], "orig3 theta_1^dep"),
           frozen20_dep=summarise(t[fro & (t.footing == "dep")], "frozen20 theta_1^dep", ci=True),
           frozen20_art=summarise(t[fro & (t.footing == "art")], "frozen20 theta_1~", ci=True),
           all24_dep=summarise(t[t.footing == "dep"], "all24 theta_1^dep"),
           trainable4_dep=summarise(t[~fro & (t.footing == "dep")], "trainable4 theta_1^dep"))
json.dump(res, open(OUT / "operational_counts_deployed_summary.json", "w"), indent=1)
print("wrote", OUT / "operational_counts_deployed.csv", "and _summary.json")
