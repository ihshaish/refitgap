#!/usr/bin/env python3
"""Both drifts combined: the four factorial cells (prior reweighting x half-dose pairing) from stored per-seed
values, deficits measured against the control's attainable level and differenced against the control, with
interactions that include the control cell and a seed-paired bootstrap.
Reads results/exact_fitter/h6_perseed_both.parquet; writes results/paired_refit/factorial_recompute.json.
Backs Appendix E, "Both drifts combined". Needs nothing beyond the repository."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = {"P0V0": "asrsprior00", "P1V0": "asrsprior100", "P0V1": "asrsp0v50", "P1V1": "asrsp1v50"}
SRC = "results/exact_fitter/h6_perseed_both.parquet"
OUT = Path("results/paired_refit/factorial_recompute.json")
BOOT, BSEED = 20000, 20260920

df = pd.read_parquet(SRC)
U, V = {}, {}
for cell, stream in CELLS.items():
    d = df[df.stream == stream].sort_values("seed")
    assert list(d.seed) == list(range(10)), (cell, list(d.seed))
    U[cell] = d.f1_frozen.to_numpy(float)
    V[cell] = d.f1_splithalf_exact.to_numpy(float)


def contrasts(idx):
    u = {c: U[c][idx].mean() for c in CELLS}
    v = {c: V[c][idx].mean() for c in CELLS}
    r = {c: v[c] - u[c] for c in CELLS}
    out = {}
    for c in CELLS:
        out[f"D_att_{c}"] = v["P0V0"] - u[c]          # against the control's attainable level
        out[f"I_{c}"] = v["P0V0"] - v[c]
        out[f"R_{c}"] = r[c]
        out[f"D_diff_{c}"] = u["P0V0"] - u[c]         # differenced against the control
        out[f"Radd_{c}"] = r[c] - r["P0V0"]
        out[f"U_{c}"] = u[c]
        out[f"V_{c}"] = v[c]
    inter = lambda q: q["P1V1"] - q["P1V0"] - q["P0V1"] + q["P0V0"]
    out["int_D_att_published"] = out["D_att_P1V1"] - out["D_att_P1V0"] - out["D_att_P0V1"]      # control treated as 0
    out["int_R_published"] = out["R_P1V1"] - out["R_P1V0"] - out["R_P0V1"]                      # control treated as 0
    out["int_D_att_full"] = inter({c: out[f"D_att_{c}"] for c in CELLS})
    out["int_R_full"] = inter({c: out[f"R_{c}"] for c in CELLS})
    out["int_D_diff"] = inter({c: out[f"D_diff_{c}"] for c in CELLS})
    out["int_Radd"] = inter({c: out[f"Radd_{c}"] for c in CELLS})
    out["int_I"] = inter({c: out[f"I_{c}"] for c in CELLS})
    out["int_logU"] = inter({c: np.log(u[c]) for c in CELLS})
    out["int_logV"] = inter({c: np.log(v[c]) for c in CELLS})
    for c in ("P1V0", "P0V1", "P1V1"):
        out[f"phi_att_{c}"] = out[f"R_{c}"] / out[f"D_att_{c}"]
        out[f"phi_diff_{c}"] = out[f"Radd_{c}"] / out[f"D_diff_{c}"] if abs(out[f"D_diff_{c}"]) > 1e-9 else float("nan")
    return out


point = contrasts(np.arange(10))
rng = np.random.default_rng(BSEED)
boots = {k: [] for k in point}
for _ in range(BOOT):
    idx = rng.integers(0, 10, 10)
    b = contrasts(idx)
    for k in point:
        boots[k].append(b[k])
res = {k: dict(mean=float(point[k]), ci=[float(np.quantile(boots[k], .025)), float(np.quantile(boots[k], .975))])
       for k in point}
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(source=SRC, boot=BOOT, seed=BSEED, cells=CELLS, results=res), open(OUT, "w"), indent=1)

print("cell      U        V        R        D_att    I        D_diff   Radd")
for c in CELLS:
    print(
        f"{c:6s} {res[f'U_{c}']['mean']:8.4f} {res[f'V_{c}']['mean']:8.4f} {res[f'R_{c}']['mean']:8.4f} "
        f"{res[f'D_att_{c}']['mean']:8.4f} {res[f'I_{c}']['mean']:8.4f} {res[f'D_diff_{c}']['mean']:8.4f} "
        f"{res[f'Radd_{c}']['mean']:8.4f}"
    )
print()
for k in (
    "int_D_att_published",
    "int_R_published",
    "int_D_att_full",
    "int_R_full",
    "int_D_diff",
    "int_Radd",
    "int_I",
    "int_logU",
    "int_logV",
):
    r = res[k]
    print(f"{k:22s} {r['mean']:+.4f}  [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
print()
for c in ("P1V0", "P0V1", "P1V1"):
    print(
        f"phi {c}: against-attainable {res[f'phi_att_{c}']['mean']:.3f}   differenced "
        f"{res[f'phi_diff_{c}']['mean']:.3f} "
        f"[{res[f'phi_diff_{c}']['ci'][0]:.3f}, {res[f'phi_diff_{c}']['ci'][1]:.3f}]"
    )
