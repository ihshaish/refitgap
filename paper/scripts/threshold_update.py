#!/usr/bin/env python3
"""Threshold-update gains (n = 50 to 500 labelled pool records) and capture rates (same-period and lagged labels, as
shares of V - U) measured from theta_1, recomputed from the checkpoints (ck), from e1_results (e1) and at the
same-scorer rule (art); seed-block bootstrap. Reads results/calib_experiment_v2.parquet, calib_lagged.parquet and
results/asrs_inherited_rule/deployed_levels.parquet; writes threshold_update_deployed*, capture_deployed* there.
Backs Appendix D, "Threshold update with few labels". Needs nothing beyond the repository."""
import json
import numpy as np
import pandas as pd

OUT = "results/asrs_inherited_rule"
BOOT, BSEED = 20000, 20260920
NS = (50, 100, 200, 500)
N = 8

lv = pd.read_parquet(f"{OUT}/deployed_levels.parquet")
c = pd.read_parquet("results/calib_experiment_v2.parquet")
c = c.merge(
    lv[["condition", "seed", "j", "U_dep_ck", "U_final_ck", "e1_final"]], on=["condition", "seed", "j"], how="left"
)
assert c.U_dep_ck.notna().all(), "theta_1 level missing for some cell"
assert (c.f1_frozen - c.U_final_ck).abs().max() < 1e-12, "same-scorer level does not reproduce f1_frozen"
FOOT = {"ck": "U_dep_ck", "e1": "e1_final", "art": "f1_frozen"}
for tag, col in FOOT.items():
    c[f"optim_{tag}"] = c.f1_oracle - c[col]
    for n in NS:
        c[f"gain_n{n}_{tag}"] = c[f"recal_{n}_mean"] - c[col]
c["U_final_minus_U_dep_ck"] = c.f1_frozen - c.U_dep_ck
c["U_dep_ck_minus_e1"] = c.U_dep_ck - c.e1_final
c.to_parquet(f"{OUT}/threshold_update_deployed.parquet", index=False)


def boot_blocks(blk_df, fn):
    rng = np.random.default_rng(BSEED)
    vals = blk_df.to_numpy(); n = len(vals); draws = np.empty(BOOT)
    for r in range(BOOT):
        k = rng.integers(0, n, n)
        draws[r] = fn(pd.DataFrame(vals[k], columns=blk_df.columns))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def blocks(sub, cols):
    obs = sub.groupby(["condition", "seed"])[cols].mean()
    return obs.groupby(level="seed").mean(), obs


summary = dict(
    generator="threshold_update.py",
    scoring="score_inherited_rule.py",
    bootstrap=dict(
        B=BOOT,
        seed=BSEED,
        blocks="seed; block value = mean over the seed's observations of the per-observation range mean",
    ),
    footings=dict(
        ck="U(s_N, theta_1^dep) recomputed from checkpoints (publishable for this analysis)",
        e1="e1_results final row, in-run fp32 (cross-check)",
        art="artefact f1_frozen at theta_1~ (the number that dies)",
    ),
    verification=dict(
        max_abs_U_dep_ck_minus_e1=float(c.U_dep_ck_minus_e1.abs().max()),
        mean_abs_U_dep_ck_minus_e1=float(c.U_dep_ck_minus_e1.abs().mean()),
        U_final_ck_vs_artefact_f1_frozen_maxabs=float((c.f1_frozen - c.U_final_ck).abs().max()),
    ),
    ranges={},
)
for rname, sub in (("periods_2_to_7", c[(c.j >= 1) & (c.j <= 6)]), ("all_periods", c)):
    out = dict(n_obs=int(sub.groupby(["condition", "seed"]).ngroups), n_cells=int(len(sub)),
               seeds=sorted(int(s) for s in sub.seed.unique()),
               U_final_minus_U_dep_ck_mean=float(sub.U_final_minus_U_dep_ck.mean()))
    cols = [f"gain_n{n}_{t}" for n in NS for t in FOOT] + [f"optim_{t}" for t in FOOT]
    pos_cols = []
    for col in cols:
        sub = sub.assign(**{f"pos_{col}": (sub[col] > 0).astype(float)}); pos_cols.append(f"pos_{col}")
    blk, obs = blocks(sub, cols + pos_cols)
    for col in cols:
        out[col] = dict(mean=float(sub[col].mean()), block_mean=float(blk[col].mean()),
                        ci=boot_blocks(blk, lambda d, q=col: d[q].mean()),
                        pct_cells_beneficial=float(100 * (sub[col] > 0).mean()),
                        pct_cells_ci=[100 * x for x in boot_blocks(blk, lambda d, q=f"pos_{col}": d[q].mean())],
                        n_obs_positive=int((obs[col] > 0).sum()), sd_obs=float(obs[col].std(ddof=0)))
    for t in FOOT:
        out[f"gain_n500_{t}_by_period"] = {
            int(j): float(v) for j, v in sub.groupby("j")[f"gain_n500_{t}"].mean().items()
        }
        out[f"gain_n50_{t}_by_period"] = {int(j): float(v) for j, v in sub.groupby("j")[f"gain_n50_{t}"].mean().items()}
    out["optimism_share_ck"] = dict(
        value=float((out["optim_ck"]["mean"] - out["gain_n500_ck"]["mean"]) / out["optim_ck"]["mean"]),
        definition="(optimism - gain_n500) / optimism, cell means, ck footing",
    )
    out["optimism_share_art"] = float(
        (out["optim_art"]["mean"] - out["gain_n500_art"]["mean"]) / out["optim_art"]["mean"]
    )
    summary["ranges"][rname] = out
    print(
        f"\n== {rname}: {out['n_obs']} observations, {out['n_cells']} cells; U_final - U_dep (ck) mean "
        f"{out['U_final_minus_U_dep_ck_mean']:+.6f}"
    )
    for n in NS:
        for t in FOOT:
            g = out[f"gain_n{n}_{t}"]
            print(
                f"  n={n:3d} [{t}] gain {g['mean']:+.6f} (block {g['block_mean']:+.6f}) CI [{g['ci'][0]:+.6f}, "
                f"{g['ci'][1]:+.6f}]  "
                f"cells>0 {g['pct_cells_beneficial']:5.1f}% CI [{g['pct_cells_ci'][0]:.1f}, "
                f"{g['pct_cells_ci'][1]:.1f}]  obs>0 {g['n_obs_positive']}/{out['n_obs']}"
            )
    for t in FOOT:
        g = out[f"optim_{t}"]
        print(f"  optimism [{t}] {g['mean']:+.6f} CI [{g['ci'][0]:+.6f}, {g['ci'][1]:+.6f}]")
    print(f"  optimism share ck {out['optimism_share_ck']['value']:.4f}  art {out['optimism_share_art']:.4f}")
    print("  n=500 gain by period ck:", {k: round(v, 6) for k, v in out["gain_n500_ck_by_period"].items()})
    print("  n=500 gain by period art:", {k: round(v, 6) for k, v in out["gain_n500_art_by_period"].items()})
json.dump(summary, open(f"{OUT}/threshold_update_deployed_summary.json", "w"), indent=1)

# capture rates
l = pd.read_parquet("results/calib_lagged.parquet")
ob = pd.read_parquet("results/exact_fitter/oracle_both.parquet"); ob = ob[ob.family == "frozen"]
l = l.merge(
    ob[["condition", "seed", "j", "f1_final_frozen", "f1_final_oracle_exact"]],
    on=["condition", "seed", "j"],
    how="left",
)
l = l.merge(
    lv[["condition", "seed", "j", "U_dep_ck", "U_final_ck", "e1_final"]], on=["condition", "seed", "j"], how="left"
)
lp = l[(l.j >= 1) & (l.j <= 6)].copy()
assert lp.U_dep_ck.notna().all() and lp.f1_final_oracle_exact.notna().all()
assert (lp.f1_frozen - lp.f1_final_frozen).abs().max() < 1e-9, "calib_lagged frozen != oracle_both frozen"
assert (lp.f1_frozen - lp.U_final_ck).abs().max() < 1e-9, "calib_lagged frozen != recomputed same-scorer level"
KEYS = ("same_500", "same_200", "lag_500", "lag_200")
cap = dict(
    generator="threshold_update.py",
    scoring="score_inherited_rule.py",
    range="periods 2-7 (j=1..6)",
    n_obs=int(lp.groupby(["condition", "seed"]).ngroups),
    n_cells=int(len(lp)),
    observations=sorted((r.condition, int(r.seed)) for r in lp[["condition", "seed"]].drop_duplicates().itertuples()),
    V="oracle_both.f1_final_oracle_exact (exact split-half, B=20)",
    bootstrap=dict(B=BOOT, seed=BSEED, blocks="seed; ratio of block means"),
    footings={},
)
for tag, col in FOOT.items():
    lp[f"bound_{tag}"] = lp.f1_final_oracle_exact - lp[col]
    for key in KEYS:
        lp[f"gain_{key}_{tag}"] = lp[key] - lp[col]
    cols = [f"bound_{tag}"] + [f"gain_{key}_{tag}" for key in KEYS]
    blk, obs = blocks(lp, cols)
    f = dict(
        bound=dict(
            mean=float(lp[f"bound_{tag}"].mean()),
            ci=boot_blocks(blk, lambda d, q=f"bound_{tag}": d[q].mean()),
            n_obs_positive=int((obs[f"bound_{tag}"] > 0).sum()),
        )
    )
    for key in KEYS:
        g = f"gain_{key}_{tag}"; b = f"bound_{tag}"
        f[key] = dict(gain=float(lp[g].mean()), gain_ci=boot_blocks(blk, lambda d, q=g: d[q].mean()),
                      capture_pct=float(100 * lp[g].mean() / lp[b].mean()),
                      capture_ci=[100 * x for x in boot_blocks(blk, lambda d, q=g, bb=b: d[q].mean() / d[bb].mean())],
                      n_obs_positive=int((obs[g] > 0).sum()))
    cap["footings"][tag] = f
    print(f"\ncapture [{tag}] bound {f['bound']['mean']:.6f} CI [{f['bound']['ci'][0]:.6f}, {f['bound']['ci'][1]:.6f}]")
    for key in KEYS:
        v = f[key]
        print(
            f"  {key:9s} gain {v['gain']:+.6f} [{v['gain_ci'][0]:+.6f}, {v['gain_ci'][1]:+.6f}]  capture "
            f"{v['capture_pct']:6.1f}% "
            f"[{v['capture_ci'][0]:.1f}, {v['capture_ci'][1]:.1f}]  obs>0 {v['n_obs_positive']}/{cap['n_obs']}"
        )
lp.to_parquet(f"{OUT}/capture_deployed_cells.parquet", index=False)
json.dump(cap, open(f"{OUT}/capture_deployed.json", "w"), indent=1)
print(
    "wrote",
    f"{OUT}/threshold_update_deployed.parquet, _summary.json, capture_deployed.json, capture_deployed_cells.parquet",
)
