#!/usr/bin/env python3
"""ASRS decomposition B = A + dR at the inherited thresholds theta_1 (fitted on the period-1 model's
validation scores), for the 20 frozen-embedding observations, from the checkpoints; V from
results/exact_fitter/oracle_both.parquet, checked against results/e1_results.parquet. Seed-block
bootstrap over both ranges. Writes results/asrs_inherited_rule/bwt_bridge{,_raw}.parquet and
bwt_bridge_summary.json. Backs Table D.3. Needs the stream files and checkpoints (not shipped)."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(10))
OUT = Path("results/asrs_inherited_rule"); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BOOT, BSEED = 20000, 20260920

df = pd.read_parquet("streams/stream_primary.parquet")
labels = sorted({l for s in df.labels for l in s.split(";")})
lab_idx = {l: i for i, l in enumerate(labels)}
Yall = np.zeros((len(df), len(labels)), dtype=np.int64)
for r, s in enumerate(df.labels):
    for l in s.split(";"):
        Yall[r, lab_idx[l]] = 1
N = int(df.period.max()) + 1
K = len(labels)


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def load_ckpt(path, base_mat):
    sd = torch.load(path, map_location="cpu")
    emb_key = next((k for k in sd if k.endswith("emb.weight")), None)
    mat = sd[emb_key].float().numpy() if emb_key is not None else base_mat
    m = BiGRU(torch.tensor(mat), freeze=True, n_labels=K).to(device)
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    m.eval()
    return m


# stored results to check against
e1 = pd.read_parquet("results/e1_results.parquet")
e1 = e1[(e1.stream == "primary") & (e1.protocol == "v2") & (e1.strategy == "naive")
        & (e1.condition.isin(["C1", "C2"]))]
e1_final = e1[e1.i == N - 1].set_index(["condition", "seed", "j"]).macro_f1
e1_diag = e1[e1.i == e1.j].set_index(["condition", "seed", "j"]).macro_f1
e1_final_pl = e1[e1.i == N - 1].set_index(["condition", "seed", "j"]).per_label_f1
ob = pd.read_parquet("results/exact_fitter/oracle_both.parquet")
ob = ob[ob.family == "frozen"].set_index(["condition", "seed", "j"])

p1 = df[(df.period == 0) & (df.split == "train_pool")]
vs = np.random.default_rng(20260802).choice(
    p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
tests = {j: df[(df.period == j) & (df.split == "test")] for j in range(N)}

rows, thrs, t0 = [], [], time.time()
for cond, kind in [("C1", "domain"), ("C2", "glove")]:
    vecs = load_vectors(kind); toks = list(vecs)
    vocab = Vocab(toks, grow=False)
    vcodes = [vocab.encode(df.text[i], False) for i in vs]
    tcodes = {j: [vocab.encode(t, False) for t in tests[j].text] for j in range(N)}
    tY = {j: Yall[tests[j].index.to_numpy()] for j in range(N)}
    for seed in SEEDS:
        d = Path(f"checkpoints/primary_{cond}_s{seed}_v2")
        assert all((d / f"period{i}.pt").exists() for i in range(N)), d
        rng = np.random.default_rng(seed)
        base = np.vstack([np.zeros(KNOBS["emb_dim"]),
                          rng.normal(0, .1, KNOBS["emb_dim"]),
                          np.stack([vecs[t] for t in toks])]).astype(np.float32)
        models = {i: load_ckpt(d / f"period{i}.pt", base) for i in range(N)}
        _, _, Pv0, _, _ = evaluate(models[0], vcodes, Yall[vs], np.full(K, .5), device)
        th_dep = fit_thresholds(Pv0, Yall[vs])
        _, _, PvN, _, _ = evaluate(models[N - 1], vcodes, Yall[vs], np.full(K, .5), device)
        th_final = fit_thresholds(PvN, Yall[vs])
        thrs.append(dict(condition=cond, seed=seed,
                         theta_dep=th_dep.tolist(), theta_final=th_final.tolist(),
                         n_labels_differ=int((th_dep != th_final).sum())))
        for j in range(N):
            _, _, Pf, _, _ = evaluate(models[N - 1], tcodes[j], tY[j], np.full(K, .5), device)
            _, _, Pc, _, _ = evaluate(models[j], tcodes[j], tY[j], np.full(K, .5), device)
            pl_dep = f1_score(tY[j], (Pf >= th_dep).astype(int), average=None, zero_division=0)
            rec = dict(condition=cond, seed=seed, j=j,
                       U_N_dep=mf1(Pf, tY[j], th_dep),
                       U_j_dep=mf1(Pc, tY[j], th_dep),
                       U_N_final=mf1(Pf, tY[j], th_final),
                       U_j_final=mf1(Pc, tY[j], th_final),
                       e1_final=float(e1_final.loc[(cond, seed, j)]),
                       e1_diag=float(e1_diag.loc[(cond, seed, j)]),
                       maxabs_perlabel_dev=float(np.max(np.abs(
                           pl_dep - np.array(json.loads(e1_final_pl.loc[(cond, seed, j)]))))))
            if j < N - 1:
                o = ob.loc[(cond, seed, j)]
                rec.update(ob_final_frozen=float(o.f1_final_frozen),
                           V_N=float(o.f1_final_oracle_exact),
                           V_j=float(o.f1_contemp_oracle_exact))
            rows.append(rec)
        del models
        print(f"  {cond} s{seed} done ({time.time()-t0:.0f}s)  "
              f"labels where theta_dep != theta_final: {thrs[-1]['n_labels_differ']}",
              flush=True)
        pd.DataFrame(rows).to_parquet(OUT / "bwt_bridge_raw.parquet", index=False)

t = pd.DataFrame(rows)
# agreement with the stored results
ver = dict(
    U_N_dep_vs_e1_final_maxabs=float((t.U_N_dep - t.e1_final).abs().max()),
    U_N_dep_vs_e1_final_meanabs=float((t.U_N_dep - t.e1_final).abs().mean()),
    U_j_dep_vs_e1_diag_maxabs=float((t.U_j_dep - t.e1_diag).abs().max()),
    U_N_final_vs_oracle_both_maxabs=float((t.U_N_final - t.ob_final_frozen).abs().max()),
    perlabel_maxabs_dev_vs_e1_4dp=float(t.maxabs_perlabel_dev.max()),
    n_rows=int(len(t)), n_obs=int(t.groupby(["condition", "seed"]).ngroups),
    theta_dep_vs_theta_final_labels_differ=[x["n_labels_differ"] for x in thrs],
)
print(json.dumps(ver, indent=1))

# decomposition from the rescored levels (ck) and from e1_results (e1)
b = t[t.j < N - 1].copy()
for tag, uN, uj in (("ck", "U_N_dep", "U_j_dep"), ("e1", "e1_final", "e1_diag")):
    b[f"B_{tag}"] = b[uj] - b[uN]
    b[f"R_N_dep_{tag}"] = b.V_N - b[uN]
    b[f"R_ref_{tag}"] = b.V_j - b[uj]
    b[f"dR_{tag}"] = b[f"R_N_dep_{tag}"] - b[f"R_ref_{tag}"]
b["A"] = b.V_j - b.V_N
b["R_N_final"] = b.V_N - b.U_N_final
b["D_paper"] = b.V_j - b.U_N_final          # D = A + R_final, at the same-scorer rule
for tag in ("ck", "e1"):
    resid = (b[f"B_{tag}"] - (b.A + b[f"dR_{tag}"])).abs().max()
    assert resid < 1e-9, (tag, resid)
    ver[f"identity_B_eq_A_plus_dR_maxresid_{tag}"] = float(resid)
b.to_parquet(OUT / "bwt_bridge.parquet", index=False)

QUANT = ["B_ck", "A", "R_N_dep_ck", "R_ref_ck", "dR_ck", "R_N_final", "D_paper",
         "B_e1", "R_N_dep_e1", "R_ref_e1", "dR_e1"]


def block_means(sub):
    """Per seed block: mean over the two conditions of the per-observation range mean."""
    obs = sub.groupby(["condition", "seed"])[QUANT].mean()
    return obs.groupby(level="seed").mean()


def boot(blk, fn):
    rng = np.random.default_rng(BSEED)
    n = len(blk); vals = blk.to_numpy()
    draws = np.empty(BOOT)
    for r in range(BOOT):
        k = rng.integers(0, n, n)
        draws[r] = fn(pd.DataFrame(vals[k], columns=blk.columns))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


summary = dict(verification=ver, bootstrap=dict(B=BOOT, seed=BSEED, blocks="seed (10)",
               note="block = mean over C1,C2 of per-observation range means; percentile CI"),
               thresholds=thrs, seeds=SEEDS, ranges={})
for rname, sub in (("primary_j1to6", b[b.j >= 1]), ("conventional_j0to6", b)):
    blk = block_means(sub)
    per_obs = sub.groupby(["condition", "seed"])[QUANT].mean()
    out = {}
    for q in QUANT:
        out[q] = dict(mean=float(blk[q].mean()), ci=boot(blk, lambda d, q=q: d[q].mean()),
                      sd_obs=float(per_obs[q].std(ddof=0)),
                      n_obs_positive=int((per_obs[q] > 0).sum()))
    for num, den, name in (("A", "B_ck", "A_over_B_ck"), ("dR_ck", "B_ck", "dR_over_B_ck"),
                           ("A", "B_e1", "A_over_B_e1"), ("dR_e1", "B_e1", "dR_over_B_e1"),
                           ("R_N_final", "D_paper", "Rfinal_over_D_paper")):
        out[name] = dict(ratio_of_means=float(blk[num].mean() / blk[den].mean()),
                         ci=boot(blk, lambda d, n=num, dd=den: d[n].mean() / d[dd].mean()))
    out["per_period_means"] = sub.groupby("j")[QUANT].mean().round(5).to_dict(orient="index")
    summary["ranges"][rname] = out
    print(f"\n== {rname} ({len(blk)} blocks, {len(per_obs)} observations)")
    for q in QUANT:
        print(
            f"  {q:12s} {out[q]['mean']:+.5f}  CI [{out[q]['ci'][0]:+.5f}, {out[q]['ci'][1]:+.5f}]  sd_obs "
            f"{out[q]['sd_obs']:.5f}  pos {out[q]['n_obs_positive']}/{len(per_obs)}"
        )
    for name in ("A_over_B_ck", "dR_over_B_ck", "Rfinal_over_D_paper"):
        print(
            f"  {name:20s} {out[name]['ratio_of_means']:.3f}  CI [{out[name]['ci'][0]:.3f}, {out[name]['ci'][1]:.3f}]"
        )
    print("  per-period means:")
    print(sub.groupby("j")[["B_ck", "A", "R_N_dep_ck", "R_ref_ck", "dR_ck", "R_N_final"]].mean().round(4).to_string())

json.dump(summary, open(OUT / "bwt_bridge_summary.json", "w"), indent=1)
print(f"\nwrote {OUT/'bwt_bridge.parquet'} and bwt_bridge_summary.json ({time.time()-t0:.0f}s)")
