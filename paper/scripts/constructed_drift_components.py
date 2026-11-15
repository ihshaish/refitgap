#!/usr/bin/env python3
"""Table E.1 components with seed-paired 95% intervals: deficit D (at the same-scorer rule), net
attainable gap A (exact split-half) and mismatch R = D - A, each stream against its control, over
periods 1 to 8; plus the fixed-Q diagnostic rows restated from results/exact_fitter/h6_outcome_both.json.
Reads results/constructed_drift_rescore.parquet and results/constructed_drift_summary.csv; writes
results/asrs_inherited_rule/dissoc_components.json. Backs Table E.1. Needs nothing beyond the repository."""
import json
import numpy as np
import pandas as pd

SRC = "results/constructed_drift_rescore.parquet"
OUT = "results/asrs_inherited_rule/dissoc_components.json"
BOOT, BSEED = 20000, 20260920
PAIRS = [("priors drift .109", "asrsprior40", "asrsprior00"),
         ("priors drift .164", "asrsprior100", "asrsprior00"),
         ("vocabulary rho=0.5", "asrsdrift50", "primary"),
         ("vocabulary rho=1.0", "asrsperm100", "primary")]

t = pd.read_parquet(SRC)
led = pd.read_csv("results/constructed_drift_summary.csv").set_index("row")


def per_seed(stream, col):
    return t[t.stream == stream].groupby("seed")[col].mean()


res = dict(source=SRC, generator="constructed_drift_components.py", bootstrap=dict(B=BOOT, seed=BSEED,
           pairing="seed-paired: one index vector per resample applied to stream and control alike"),
           construct="dU at theta_1~ (thresholds fitted to the final model's own scores on the "
                     "period-1 validation split; constructed_drifts.py); dV exact split-half B=200",
           rows={})
for label, arm, ctrl in PAIRS:
    uc, ua = per_seed(ctrl, "f1_frozen"), per_seed(arm, "f1_frozen")
    vc, va = per_seed(ctrl, "f1_oracle_splithalf_exact"), per_seed(arm, "f1_oracle_splithalf_exact")
    seeds = sorted(set(uc.index) & set(ua.index) & set(vc.index) & set(va.index))
    uc, ua, vc, va = (x.loc[seeds].to_numpy() for x in (uc, ua, vc, va))
    D, A = float(uc.mean() - ua.mean()), float(vc.mean() - va.mean())
    R = D - A
    rng = np.random.default_rng(BSEED)
    n = len(seeds)
    dD, dA, dR, dS = (np.empty(BOOT) for _ in range(4))
    for b in range(BOOT):
        k = rng.integers(0, n, n)
        d = uc[k].mean() - ua[k].mean(); a = vc[k].mean() - va[k].mean()
        dD[b], dA[b], dR[b], dS[b] = d, a, d - a, 100 * (d - a) / d
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    row = dict(
        stream=arm,
        control=ctrl,
        n_seeds=n,
        seeds=[int(s) for s in seeds],
        D_delta=D,
        ci_D=ci(dD),
        A_delta=A,
        ci_A=ci(dA),
        R_delta=R,
        ci_R=ci(dR),
        share_R_over_D_pct=100 * R / D,
        ci_share=ci(dS),
        summary_csv=dict(
            frozen=float(led.loc[label, "frozen"]),
            irred_exact=float(led.loc[label, "irred_exact"]),
            share_exact=float(led.loc[label, "share_exact"]),
            ci_lo=float(led.loc[label, "ci_lo"]),
            ci_hi=float(led.loc[label, "ci_hi"]),
        ),
    )
    # the stored CSV holds |dV|; compare with the signed value
    row["summary_check"] = dict(D_matches=bool(abs(row["summary_csv"]["frozen"] - D) < 1e-9),
                               absA_matches=bool(abs(row["summary_csv"]["irred_exact"] - abs(A)) < 1e-9))
    res["rows"][label] = row
    print(
        f"{label:20s} D {D:+.6f} [{row['ci_D'][0]:+.6f}, {row['ci_D'][1]:+.6f}]  "
        f"A {A:+.6f} [{row['ci_A'][0]:+.6f}, {row['ci_A'][1]:+.6f}]  "
        f"R {R:+.6f} [{row['ci_R'][0]:+.6f}, {row['ci_R'][1]:+.6f}]  "
        f"share {100*R/D:.1f} [{row['ci_share'][0]:.1f}, {row['ci_share'][1]:.1f}]  n={n}  summary ok "
        f"{row['summary_check']}"
    )

# fixed-Q diagnostic rows at theta_1, exact fitter
h6 = json.load(open("results/exact_fitter/h6_outcome_both.json"))["exact"]
res["h6_deployed_fixedQ_rows"] = dict(
    source="results/exact_fitter/h6_outcome_both.json (exact arm, deployed theta_1)",
    convention="fixed-Q diagnostic: D = V(control) - U(stream, theta_1); I = attainable; R = D - I",
    P1V0=dict(D=h6["P1V0"]["D"], ci_D=h6["P1V0"]["ci_D"], I=h6["P1V0"]["I"], ci_I=h6["P1V0"]["ci_I"],
              R=h6["P1V0"]["R"], ci_R=h6["P1V0"]["ci_R"]),
    P0V1=dict(D=h6["P0V1"]["D"], ci_D=h6["P0V1"]["ci_D"], I=h6["P0V1"]["I"], ci_I=h6["P0V1"]["ci_I"],
              R=h6["P0V1"]["R"], ci_R=h6["P0V1"]["ci_R"]),
    P1V1=dict(D=h6["P1V1"]["D"], ci_D=h6["P1V1"]["ci_D"], I=h6["P1V1"]["I"], ci_I=h6["P1V1"]["ci_I"],
              R=h6["P1V1"]["R"], ci_R=h6["P1V1"]["ci_R"]))
for k in ("P1V0", "P0V1", "P1V1"):
    v = res["h6_deployed_fixedQ_rows"][k]
    print(
        f"H6 {k}: D {v['D']:.6f} [{v['ci_D'][0]:.6f}, {v['ci_D'][1]:.6f}]  I {v['I']:+.6f} [{v['ci_I'][0]:+.6f}, "
        f"{v['ci_I'][1]:+.6f}]  "
        f"R {v['R']:.6f} [{v['ci_R'][0]:.6f}, {v['ci_R'][1]:.6f}]"
    )
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
