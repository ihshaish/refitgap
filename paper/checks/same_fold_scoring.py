#!/usr/bin/env python3
"""Same-fold inherited scoring for the frozen-embedding ASRS decomposition (supplement Table D.2;
20 observations = C1, C2 x seeds 0-9; period indices j = 1..6, i.e. periods 2 to 7).
Does scoring the inherited thresholds on the held-out halves, as the refit is scored, change
the decomposition? No: the deficit and the change in mismatch move by less than 1e-5.

Reuses, by import from scripts/asrs_supplementary.py: mf1 (macro-F1, P >= theta,
zero_division=0), halves (first n//2 of a permutation), cells (P_7_j final, P_j_j reference,
Y_j, theta_dep), load, boot (seed-block percentile bootstrap, 20000 resamples, rng 20260920),
J = range(1, 7). Partitions: default_rng(7000 + 100*seed + j), 200 successive calls to halves,
exactly as in asrs_supplementary.Vsplit (fit_exact consumes no random numbers, so the stream of
partitions is identical). No threshold is fitted here; A, V_N, V_ref are read from the stored
recovery_target.parquet.

Input: the stored score matrices results/asrs_supplementary/scores/ (written by
scripts/store_asrs_scores.py from the checkpoints). Writes same_fold_cells.csv, same_fold_summary.json.
Run from paper/:  python checks/same_fold_scoring.py
Cannot rerun from this repository alone (the score matrices are not shipped); its stored outputs are kept here.
"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from asrs_supplementary import mf1, halves, cells, load, boot, J

OUT = "checks"
t0 = time.time()
stored = pd.read_parquet("results/asrs_supplementary/recovery_target.parquet").set_index(["cond", "seed", "j"])
rows = []
for cond, seed, z in load():
    for j in J:
        PN, Pj, Y, th = cells(z, j)
        U_ref, U_N = mf1(Pj, Y, th), mf1(PN, Y, th)
        rng = np.random.default_rng(7000 + 100 * seed + j)
        hr, hn = [], []
        for _ in range(200):
            a, b = halves(len(Y), rng)
            hr.append(0.5 * (mf1(Pj[b], Y[b], th) + mf1(Pj[a], Y[a], th)))
            hn.append(0.5 * (mf1(PN[b], Y[b], th) + mf1(PN[a], Y[a], th)))
        s = stored.loc[(cond, seed, j)]
        rows.append(dict(cond=cond, seed=seed, j=j, period=j + 1, n=len(Y), U_ref=U_ref, U_N=U_N,
                         Uh_ref=float(np.mean(hr)), Uh_N=float(np.mean(hn)),
                         V_ref=float(s.V_ref), V_N=float(s.V_N), A=float(s.A),
                         dev_Uref=abs(U_ref - s.U_ref), dev_UN=abs(U_N - s.U_N)))
t = pd.DataFrame(rows)
maxdev = float(max(t.dev_Uref.max(), t.dev_UN.max()))
print(f"reproduction of stored U: max |dev| = {maxdev:.3e}")
if maxdev > 1e-9:
    print("STOP: whole-set U does not reproduce stored values to 1e-9")
    t.to_csv(f"{OUT}/same_fold_cells.csv", index=False)
    sys.exit(1)

t["h_ref"] = t.Uh_ref - t.U_ref; t["h_final"] = t.Uh_N - t.U_N; t["h_diff"] = t.h_ref - t.h_final
t["B_whole"] = t.U_ref - t.U_N; t["B_same"] = t.Uh_ref - t.Uh_N
t["dR_whole"] = t.B_whole - t.A; t["dR_same"] = t.B_same - t.A
t.to_csv(f"{OUT}/same_fold_cells.csv", index=False)

cols = ["B_whole", "B_same", "A", "dR_whole", "dR_same", "h_ref", "h_final", "h_diff"]
blk = t.groupby("seed")[cols].mean()
lo, hi = boot(blk.to_numpy())
summary = {c: dict(mean=float(blk[c].mean()), ci=[float(lo[i]), float(hi[i])]) for i, c in enumerate(cols)}
for k in ("whole", "same"):
    summary[f"share_{k}"] = summary[f"dR_{k}"]["mean"] / summary[f"B_{k}"]["mean"]
per_period = t.groupby("period")[["h_ref", "h_final", "h_diff"]].mean()
out = dict(n_obs=int(t.groupby(["cond", "seed"]).ngroups), cells=len(t), max_dev_U_vs_stored=maxdev,
           summary=summary, per_period=per_period.to_dict(orient="index"),
           max_abs_h=float(np.abs(t[["h_ref", "h_final"]].to_numpy()).max()),
           max_abs_h_diff=float(t.h_diff.abs().max()), runtime_s=time.time() - t0)
json.dump(out, open(f"{OUT}/same_fold_summary.json", "w"), indent=1)
pd.options.display.float_format = "{:+.6f}".format
print(per_period)
for c in cols: print(f"{c:9s} {summary[c]['mean']:+.6f} [{summary[c]['ci'][0]:+.6f}, {summary[c]['ci'][1]:+.6f}]")
print("share", {k: round(summary[f'share_{k}'], 4) for k in ("whole", "same")})
print("max |h|", out["max_abs_h"], "max |h_diff|", out["max_abs_h_diff"], f"runtime {out['runtime_s']:.1f}s")
