#!/usr/bin/env python3
"""Mismatch terms under whole-set and same-fold inherited scoring, and the F1 convention for empty labels.

Reads: same_fold_cells.csv (written by same_fold_scoring.py; no rescoring here) and, for the last
line, the stored score matrices of results/asrs_supplementary/scores/ (not shipped).
Computes: R_N and R_ref with the whole-set and the same-fold inherited score, each with the
seed-block bootstrap of scripts/asrs_supplementary.py; then the macro-F1 of the analysis scripts
(sklearn f1_score, zero_division=0) against refitgap.fitters.macro_f1 on a label with no true and
no predicted positive, on a score tied with the threshold, and on the 120 stored final cells.
Found: the same-fold terms differ from the whole-set ones by less than 0.0004; both F1 functions
give 0.5 and 1.0 on the two toy cases and agree exactly on the 120 cells
(same_fold_mismatch_terms_summary.json, same_fold_mismatch_terms_log.txt).
    python checks/same_fold_mismatch_terms.py          (from paper/)
The first part reruns from this repository; the 120-cell comparison needs the stored score matrices.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from asrs_supplementary import boot, mf1, cells, load
from refitgap.fitters import macro_f1, f1_vec
import sklearn
t = pd.read_csv("checks/same_fold_cells.csv")
t["R_N_whole"] = t.V_N - t.U_N; t["R_N_same"] = t.V_N - t.Uh_N
t["R_ref_whole"] = t.V_ref - t.U_ref; t["R_ref_same"] = t.V_ref - t.Uh_ref
cols = ["R_N_whole", "R_N_same", "R_ref_whole", "R_ref_same"]
blk = t.groupby("seed")[cols].mean(); lo, hi = boot(blk.to_numpy())
out = {c: dict(mean=float(blk[c].mean()), ci=[float(lo[i]), float(hi[i])]) for i, c in enumerate(cols)}
for c in cols: print(f"{c:12s} {out[c]['mean']:.6f} [{out[c]['ci'][0]:.6f}, {out[c]['ci'][1]:.6f}]")
# zero-denominator label: no true positive, no predicted positive in the scored set
Y = np.array([[0, 1], [0, 0], [0, 1]]); P = np.array([[0.1, 0.9], [0.2, 0.1], [0.3, 0.8]]); th = np.array([0.5, 0.5])
zd = dict(sklearn_version=sklearn.__version__, analysis_mf1=mf1(P, Y, th), refitgap_macro_f1=macro_f1(P, Y, th),
          refitgap_f1_vec=f1_vec(P, Y, th).tolist())
# tie at threshold: score exactly equal to theta counts as predicted positive under >=
Pt = np.array([[0.5, 0.9], [0.2, 0.1], [0.3, 0.8]]); Yt = np.array([[1, 1], [0, 0], [0, 1]])
zd["tie_analysis"] = mf1(Pt, Yt, th); zd["tie_refitgap"] = macro_f1(Pt, Yt, th)
# agreement on real stored cells (all 120 whole-set U values)
dev = max(abs(mf1(P_, Y_, th_) - macro_f1(P_, Y_, th_)) for _, _, z in load() for j in range(1, 7)
          for P_, Y_, th_ in [cells(z, j)[0::2][:1] + (cells(z, j)[2], cells(z, j)[3])] )
zd["max_dev_analysis_vs_refitgap_on_120_final_cells"] = dev
print(zd); out["f1_convention"] = zd
json.dump(out, open("checks/same_fold_mismatch_terms_summary.json", "w"), indent=1)
