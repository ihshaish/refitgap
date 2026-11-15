#!/usr/bin/env python3
"""Repeated constructions: rescore each stream as constructed_drifts.run_stream does and difference each
arm against its seed-matched control (D = U_ctrl - U_arm, A = V_ctrl - V_arm, R = D - A), with intervals
and the planned predictions (plans/repeated-constructions.txt). Writes results/constructions/
construction_seeds.parquet and construction_seeds_summary.json. Backs Appendix E, "Repeated constructions".
Needs the construction streams and checkpoints (not shipped); rescored streams are reused from the parquet."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from train_sequential import load_vectors
import constructed_drifts as F

OUT = Path("results/constructions"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS_C = [20260921, 20260922, 20260923, 20260924, 20260925, 20260926, 20260927, 20260928]
BOOT, BSEED = 20000, 20260920
F.SPLITS = 200

ARMS = []
for S in SEEDS_C:
    ARMS += [
        (f"asrsdrift{S}_r50", "primary", "A rho=0.5", S),
        (f"asrsdrift{S}_r100", "primary", "A rho=1.0", S),
        (f"asrsperm{S}_r100", "primary", "B rho=1.0", S),
        (f"asrsprior{S}_100", f"asrsprior{S}_00", "prior rho=1.0", S),
    ]
STREAMS = sorted({a for a, c, _, _ in ARMS} | {c for a, c, _, _ in ARMS})

vecs = load_vectors("glove")
# cache: a stream already rescored with every finished checkpoint is reused
cache_path = OUT / "construction_seeds.parquet"
cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["stream", "seed"])
cache = cache[cache.stream != "primary"]
rows = []
kept = []
for s in STREAMS:
    if s == "primary":
        continue
    if not Path(f"streams/stream_{s}.parquet").exists():
        print(f"  no stream {s}", flush=True); continue
    have = set(cache[cache.stream == s].seed.unique())
    done = {
        int(p.name.split("_s")[1].split("_")[0])
        for p in Path("checkpoints").glob(f"{s}_C2_s*_v2")
        if (p / "period7.pt").exists()
    }
    if done and done <= have:
        print(f"  cached {s} seeds {sorted(have)}", flush=True); kept.append(cache[cache.stream == s]); continue
    print(f"  rescoring {s}", flush=True)
    F.run_stream(s, vecs, rows)
t = pd.concat([pd.DataFrame(rows)] + kept, ignore_index=True)
# primary C2 seeds 0-2 from results/constructed_drift_rescore.parquet (same procedure, same splits)
pub = pd.read_parquet("results/constructed_drift_rescore.parquet")
t = pd.concat([t, pub[(pub.stream == "primary") & (pub.seed.isin([0, 1, 2]))]], ignore_index=True)
t.to_parquet(OUT / "construction_seeds.parquet", index=False)


def per_seed(stream, col):
    d = t[t.stream == stream]
    return d.groupby("seed")[col].mean()


summary = {}
for arm, ctrl, kind, S in ARMS:
    uc, ua = per_seed(ctrl, "f1_frozen"), per_seed(arm, "f1_frozen")
    vc, va = per_seed(ctrl, "f1_oracle_splithalf_exact"), per_seed(arm, "f1_oracle_splithalf_exact")
    seeds = sorted(set(uc.index) & set(ua.index) & set(vc.index) & set(va.index))
    if not seeds:
        print("  no runs for", arm); continue
    U = np.array([[uc[s], ua[s], vc[s], va[s]] for s in seeds])
    def terms(M):
        D = M[:, 0].mean() - M[:, 1].mean(); A = M[:, 2].mean() - M[:, 3].mean()
        return D, A, D - A, A / D, (D - A) / D
    pt = terms(U)
    rng = np.random.default_rng(BSEED)
    draws = np.array([terms(U[rng.integers(0, len(seeds), len(seeds))]) for _ in range(BOOT)])
    ci = [[float(np.percentile(draws[:, i], 2.5)), float(np.percentile(draws[:, i], 97.5))] for i in range(5)]
    summary[arm] = dict(
        kind=kind,
        construction_seed=S,
        control=ctrl,
        n_seeds=len(seeds),
        D=pt[0],
        A=pt[1],
        R=pt[2],
        A_over_D=pt[3],
        R_over_D=pt[4],
        ci_D=ci[0],
        ci_A=ci[1],
        ci_R=ci[2],
        ci_A_over_D=ci[3],
        ci_R_over_D=ci[4],
    )
    print(
        f"{arm:24s} {kind:14s} D {pt[0]:+.4f} A {pt[1]:+.4f} R {pt[2]:+.4f} A/D {pt[3]:.3f} [{ci[3][0]:.3f}, "
        f"{ci[3][1]:.3f}] R/D {pt[4]:.3f}"
    )

between = {}
for kind in sorted({k["kind"] for k in summary.values()}):
    vals = [k for k in summary.values() if k["kind"] == kind]
    between[kind] = dict(
        n_constructions=len(vals),
        sd_A_over_D=float(np.std([v["A_over_D"] for v in vals], ddof=0)),
        sd_R_over_D=float(np.std([v["R_over_D"] for v in vals], ddof=0)),
        mean_A_over_D=float(np.mean([v["A_over_D"] for v in vals])),
        mean_R_over_D=float(np.mean([v["R_over_D"] for v in vals])),
    )
pred = dict(
    p1_vocab_AD_above_075_at_rho1=all(
        v["A_over_D"] > 0.75 for v in summary.values() if v["kind"] in ("A rho=1.0", "B rho=1.0")
    ),
    p1_vocab_AD_above_060_at_rho05=all(v["A_over_D"] > 0.60 for v in summary.values() if v["kind"] == "A rho=0.5"),
    p2_vocab_AD_above_014=all(v["A_over_D"] > 0.14 for v in summary.values() if v["kind"].startswith(("A", "B"))),
    p3_falsifier_vocab_AD_below_050_at_rho1=any(
        v["A_over_D"] < 0.50 for v in summary.values() if v["kind"] in ("A rho=1.0", "B rho=1.0")
    ),
    p5_prior_RD_above_080_and_AD_below_015=all(
        v["R_over_D"] > 0.80 and v["A_over_D"] < 0.15 for v in summary.values() if v["kind"] == "prior rho=1.0"
    ),
    p6_falsifier_prior_AD_above_030=any(v["A_over_D"] > 0.30 for v in summary.values() if v["kind"] == "prior rho=1.0"),
)
(OUT / "construction_seeds_summary.json").write_text(
    json.dumps(
        dict(
            per_construction=summary,
            between_construction=between,
            predictions=pred,
            bootstrap=dict(B=BOOT, seed=BSEED, unit="model seed"),
        ),
        indent=1,
    )
)
print(json.dumps(between, indent=1)); print(json.dumps(pred, indent=1))
