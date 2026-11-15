#!/usr/bin/env python3
"""As prior_movement_asrs.py, with the same-scorer and reweighted rules fitted on the final model's scores on the
period-1 validation split (831 records), recomputed from the checkpoints. Env FITTER: exact (default) or grid; the paper
prints grid. Writes results/prior_movement_valsplit_grid/ (grid) or results/prior_movement_valsplit/ (exact).
Backs Appendix D, "Threshold update with few labels" (62% and 17%).
Needs the stream file and the final checkpoints (not shipped)."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from train_sequential import KNOBS, Vocab, BiGRU, evaluate, load_vectors
import prior_movement_asrs
from prior_movement_asrs import split_one
N = 8
FITTER = os.environ.get("FITTER", "exact")
if FITTER == "grid":
    # the 19-point grid fitter of train_sequential.fit_thresholds, with optional record weights
    def grid_th_label(s, y, w=None):
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
        y = y.astype(bool); best, bth = -1.0, 0.5
        for g in np.linspace(0.05, 0.95, 19):
            pred = s >= g
            tp = float((w * (pred & y)).sum()); fp = float((w * (pred & ~y)).sum()); fn = float((w * (~pred & y)).sum())
            f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
            if f1 > best:
                best, bth = f1, float(g)
        return bth
    prior_movement_asrs.exact_th_label = grid_th_label   # split_one looks the fitter up at call time

OUT = Path("results/prior_movement_valsplit_grid" if FITTER == "grid" else "results/prior_movement_valsplit")
OUT.mkdir(parents=True, exist_ok=True)
SC = Path("results/asrs_inherited_rule/scores")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BOOT, BSEED = 20000, 20260920

df = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in df.labels for l in s.split(";")})
lab_idx = {l: i for i, l in enumerate(labels)}
Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
for r, s in enumerate(df.labels):
    for l in s.split(";"):
        Yall[r, lab_idx[l]] = 1
K = len(labels)
p1 = df[(df.period == 0) & (df.split == "train_pool")]
vs = np.random.default_rng(20260802).choice(p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
ob = pd.read_parquet("results/exact_fitter/oracle_both.parquet")
ob = ob[ob.family == "frozen"].set_index(["condition", "seed", "j"])


def load_final(cond, seed, vecs, toks):
    rng = np.random.default_rng(seed)
    base = np.vstack([np.zeros(KNOBS["emb_dim"]), rng.normal(0, .1, KNOBS["emb_dim"]),
                      np.stack([vecs[t] for t in toks])]).astype(np.float32)
    sd = torch.load(f"checkpoints/primary_{cond}_s{seed}_v2/period{N-1}.pt", map_location="cpu")
    m = BiGRU(torch.tensor(base), freeze=True, n_labels=K).to(device)
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); m.eval()
    return m


runs, per_rows = [], []
for cond, kind in (("C1", "domain"), ("C2", "glove")):
    vecs = load_vectors(kind); toks = list(vecs); vocab = Vocab(toks, grow=False)
    vcodes = [vocab.encode(df.text[i], False) for i in vs]
    for seed in range(10):
        z = np.load(SC / f"{cond}_s{seed}.npz")
        m = load_final(cond, seed, vecs, toks)
        _, _, Pv, _, _ = evaluate(m, vcodes, Yall[vs], np.full(K, .5), device)
        V_ext = {j: float(ob.f1_final_oracle_exact.loc[(cond, seed, j)]) for j in range(N - 1)}
        out, per = split_one(z["theta_dep"], Pv.astype(np.float64), Yall[vs],
                             lambda j: (z[f"P_final_{j}"].astype(np.float64), z[f"Y_{j}"].astype(np.int64)), V_ext)
        runs.append(dict(condition=cond, seed=seed, **out))
        per_rows.append(pd.DataFrame(per).assign(condition=cond, seed=seed))
        print(
            f"  {cond} s{seed}: frozen {out['frozen_dep']:.5f} refit_old {out['refit_old']:.5f} prior_only "
            f"{out['prior_only']:.5f} V {out['refit_V']:.5f}",
            flush=True,
        )
        del m
a = pd.DataFrame(runs)
a.drop(columns=["theta_b"]).to_parquet(OUT / "prior_movement_valsplit_runs.parquet", index=False)
pd.concat(per_rows).to_parquet(OUT / "prior_movement_valsplit_periods.parquet", index=False)
blk = a.groupby("seed")[["frozen_dep", "refit_old", "prior_only", "refit_V"]].mean()
def shares(d):
    tot = d.refit_V.mean() - d.frozen_dep.mean()
    return np.array(
        [
            (d.refit_old.mean() - d.frozen_dep.mean()) / tot,
            (d.prior_only.mean() - d.refit_old.mean()) / tot,
            (d.refit_V.mean() - d.prior_only.mean()) / tot,
            tot,
        ]
    )
rng = np.random.default_rng(BSEED); n = len(blk); draws = np.empty((BOOT, 4))
for r in range(BOOT):
    draws[r] = shares(blk.iloc[rng.integers(0, n, n)])
pt = shares(blk)
res = dict(
    fit_set=f"final model's period-1 VALIDATION split (831 records), {FITTER} fitter; prior weights per label = "
            f"period-j prior odds",
    levels=dict(
        frozen_dep=float(blk.frozen_dep.mean()),
        refit_old=float(blk.refit_old.mean()),
        prior_only=float(blk.prior_only.mean()),
        refit_V=float(blk.refit_V.mean()),
    ),
    total_gain=float(pt[3]),
    share_same_scorer_pct=float(100 * pt[0]),
    ci_same_scorer=[float(100 * np.percentile(draws[:, 0], q)) for q in (2.5, 97.5)],
    share_prior_pct=float(100 * pt[1]),
    ci_prior=[float(100 * np.percentile(draws[:, 1], q)) for q in (2.5, 97.5)],
    share_remainder_pct=float(100 * pt[2]),
    ci_remainder=[float(100 * np.percentile(draws[:, 2], q)) for q in (2.5, 97.5)],
    n_runs_prior_gain_positive=int(((a.prior_only - a.refit_old) > 0).sum()),
    n_runs_old_gain_positive=int(((a.refit_old - a.frozen_dep) > 0).sum()),
    R_N_valsplit_same_scorer=float(blk.refit_V.mean() - blk.refit_old.mean()),
)
(OUT / f"prior_movement_valsplit_{FITTER}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
