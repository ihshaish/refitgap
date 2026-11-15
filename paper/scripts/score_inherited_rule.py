#!/usr/bin/env python3
"""Rescore the final models of the primary stream (C1, C2 seeds 0-9; C3, C4 seeds 1-2) at the deployed thresholds.

Reads streams/stream_primary.parquet, checkpoints/primary_{cond}_s{seed}_v2/ and stored results to compare against;
writes results/asrs_inherited_rule/scores/{cond}_s{seed}.npz, deployed_levels.parquet and
deployed_scoring_verification.json. Backs the threshold update (Appendix D, "Threshold update with few labels")
and the ASRS decomposition (Tables D.3-D.5). Needs the checkpoints (not distributed) and the embeddings."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from train_sequential import KNOBS, Vocab, BiGRU, evaluate, fit_thresholds, load_vectors

OUT = Path("results/asrs_inherited_rule"); SC = OUT / "scores"; SC.mkdir(parents=True, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RUNS = [("C1", "domain", list(range(10))), ("C2", "glove", list(range(10))),
        ("C3", "domain", [1, 2]), ("C4", "glove", [1, 2])]
if len(sys.argv) > 1:                      # optional restriction, e.g. C1:0,1
    RUNS = [(c, k, [int(s) for s in sys.argv[1].split(":")[1].split(",")])
            for c, k, _ in RUNS if c == sys.argv[1].split(":")[0]]

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


# stored results the rescoring is compared against
e1 = pd.read_parquet("results/e1_results.parquet")
e1 = e1[(e1.stream == "primary") & (e1.protocol == "v2") & (e1.strategy == "naive")
        & (e1.condition.isin(["C1", "C2", "C3", "C4"]))]
e1_final = e1[e1.i == N - 1].set_index(["condition", "seed", "j"]).macro_f1
e1_final_pl = e1[e1.i == N - 1].set_index(["condition", "seed", "j"]).per_label_f1
calib = pd.read_parquet("results/calib_experiment_v2.parquet").set_index(["condition", "seed", "j"])
ob = pd.read_parquet("results/exact_fitter/oracle_both.parquet").set_index(["condition", "seed", "j"])
bridge = json.load(open(OUT / "bwt_bridge_summary.json"))
bridge_th = {(t["condition"], t["seed"]): t for t in bridge["thresholds"]}

p1 = df[(df.period == 0) & (df.split == "train_pool")]
val_sel = np.random.default_rng(20260802).choice(
    p1.index.to_numpy(), int(len(p1) * KNOBS["val_frac_p1"]), False)
tr_sel0 = np.setdiff1d(p1.index.to_numpy(), val_sel)
tests = {j: df[(df.period == j) & (df.split == "test")] for j in range(N)}
pools = {j: df[(df.period == j) & (df.split == "train_pool")].index.to_numpy() for j in range(N)}

VEC = {}
def vectors(kind):
    if kind not in VEC:
        VEC[kind] = load_vectors(kind)
    return VEC[kind]


def load_model(path, base_mat, vocab, frozen):
    sd = torch.load(path, map_location="cpu")
    if frozen:
        m = BiGRU(torch.tensor(base_mat), freeze=True, n_labels=K).to(device)
        m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    else:
        assert "emb.weight" in sd, path
        assert sd["emb.weight"].shape[0] == len(vocab.idx) + 2, \
            (path, sd["emb.weight"].shape[0], len(vocab.idx) + 2)
        m = BiGRU(sd["emb.weight"].float(), freeze=False, n_labels=K).to(device)
        m.load_state_dict({k: v.float() for k, v in sd.items()})
    m.eval()
    return m


rows, thrs, t0 = [], [], time.time()
for cond, kind, seeds in RUNS:
    frozen = cond in ("C1", "C2")
    vecs = vectors(kind); toks = list(vecs)
    for seed in seeds:
        d = Path(f"checkpoints/primary_{cond}_s{seed}_v2")
        assert (d / "period0.pt").exists() and (d / f"period{N-1}.pt").exists(), d
        rng = np.random.default_rng(seed)
        base = np.vstack([np.zeros(KNOBS["emb_dim"]), rng.normal(0, .1, KNOBS["emb_dim"]),
                          np.stack([vecs[t] for t in toks])]).astype(np.float32)
        vocab = Vocab(toks, grow=not frozen)
        if not frozen:                                   # replay period-1 vocabulary growth
            for idx in tr_sel0:
                vocab.encode(df.text[idx], train_mode=True)
        model0 = load_model(d / "period0.pt", base, vocab, frozen)
        vcodes = [vocab.encode(df.text[i], False) for i in val_sel]
        _, _, Pv0, _, _ = evaluate(model0, vcodes, Yall[val_sel], np.full(K, .5), device)
        th_dep = fit_thresholds(Pv0, Yall[val_sel])
        del model0
        if not frozen:                                   # then periods 2 to 8
            for i in range(1, N):
                for idx in pools[i]:
                    vocab.encode(df.text[idx], train_mode=True)
        modelN = load_model(d / f"period{N-1}.pt", base, vocab, frozen)
        vcodes = [vocab.encode(df.text[i], False) for i in val_sel]
        _, _, PvN, _, _ = evaluate(modelN, vcodes, Yall[val_sel], np.full(K, .5), device)
        th_final = fit_thresholds(PvN, Yall[val_sel])
        rec_th = dict(condition=cond, seed=seed, theta_dep=th_dep.tolist(),
                      theta_final=th_final.tolist(),
                      n_labels_differ=int((th_dep != th_final).sum()))
        if (cond, seed) in bridge_th:
            b = bridge_th[(cond, seed)]
            rec_th["theta_dep_eq_bridge"] = bool(np.allclose(th_dep, b["theta_dep"], atol=1e-12))
            rec_th["theta_final_eq_bridge"] = bool(np.allclose(th_final, b["theta_final"], atol=1e-12))
        thrs.append(rec_th)
        store = dict(theta_dep=th_dep, theta_final=th_final, labels=np.array(labels))
        for j in range(N):
            te = tests[j]; Yt = Yall[te.index.to_numpy()]
            tcodes = [vocab.encode(t, False) for t in te.text]
            _, _, Pf, _, _ = evaluate(modelN, tcodes, Yt, np.full(K, .5), device)
            # the same 500-record pool draw as operational_counts.py
            sel = np.random.default_rng(500 + seed * 10 + j).choice(
                pools[j], min(500, len(pools[j])), False)
            _, _, Ps, _, _ = evaluate(modelN, [vocab.encode(df.text[i], False) for i in sel],
                                      Yall[sel], np.full(K, .5), device)
            store[f"P_final_{j}"] = Pf.astype(np.float32); store[f"Y_{j}"] = Yt.astype(np.int8)
            store[f"P_pool500_{j}"] = Ps.astype(np.float32); store[f"Y_pool500_{j}"] = Yall[sel].astype(np.int8)
            store[f"sel_{j}"] = sel
            pl_dep = f1_score(Yt, (Pf >= th_dep).astype(int), average=None, zero_division=0)
            rec = dict(condition=cond, seed=seed, j=j,
                       U_dep_ck=mf1(Pf, Yt, th_dep), U_final_ck=mf1(Pf, Yt, th_final),
                       e1_final=float(e1_final.loc[(cond, seed, j)]),
                       calib_f1_frozen=float(calib.f1_frozen.loc[(cond, seed, j)]),
                       maxabs_perlabel_dev_vs_e1=float(np.max(np.abs(
                           pl_dep - np.array(json.loads(e1_final_pl.loc[(cond, seed, j)]))))))
            if (cond, seed, j) in ob.index:
                rec["ob_final_frozen"] = float(ob.f1_final_frozen.loc[(cond, seed, j)])
                rec["V_N_exact"] = float(ob.f1_final_oracle_exact.loc[(cond, seed, j)])
                rec["V_j_exact"] = float(ob.f1_contemp_oracle_exact.loc[(cond, seed, j)])
            rows.append(rec)
        np.savez_compressed(SC / f"{cond}_s{seed}.npz", **store)
        del modelN
        pd.DataFrame(rows).to_parquet(OUT / "deployed_levels.parquet", index=False)
        print(
            f"  {cond} s{seed} done ({time.time()-t0:.0f}s)  theta_dep != theta_final on "
            f"{rec_th['n_labels_differ']} labels  "
            f"max|U_dep - e1| "
            f"{max(abs(r['U_dep_ck']-r['e1_final']) for r in rows if r['condition']==cond and r['seed']==seed):.2e}",
            flush=True,
        )

t = pd.DataFrame(rows)
t.to_parquet(OUT / "deployed_levels.parquet", index=False)
dev = (t.U_dep_ck - t.e1_final).abs()
ver = dict(
    n_runs=int(t.groupby(["condition", "seed"]).ngroups), n_cells=int(len(t)),
    U_dep_vs_e1_final=dict(maxabs=float(dev.max()), meanabs=float(dev.mean()),
                           median=float(dev.median()), p90=float(dev.quantile(.9)),
                           by_condition_maxabs=dev.groupby(t.condition).max().to_dict(),
                           worst_cell=t.loc[dev.idxmax(), ["condition", "seed", "j"]].to_dict(),
                           n_cells_above_5e4=int((dev > 5e-4).sum()),
                           runs_above_5e4=sorted({(r.condition, int(r.seed)) for _, r in t[dev > 5e-4].iterrows()})),
    U_final_vs_calib_f1_frozen_maxabs=float((t.U_final_ck - t.calib_f1_frozen).abs().max()),
    U_final_vs_oracle_both_maxabs=float((t.U_final_ck - t.ob_final_frozen).abs().max()),
    perlabel_maxabs_dev_vs_e1_4dp=float(t.maxabs_perlabel_dev_vs_e1.max()),
    theta_dep_eq_bridge_all=bool(all(x.get("theta_dep_eq_bridge", True) for x in thrs)),
    theta_final_eq_bridge_all=bool(all(x.get("theta_final_eq_bridge", True) for x in thrs)),
    thresholds=thrs, seconds=round(time.time() - t0, 1), device=str(device),
)
json.dump(ver, open(OUT / "deployed_scoring_verification.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in ver.items() if k != "thresholds"}, indent=1, default=str))
print(f"wrote {OUT/'deployed_levels.parquet'}, {SC}/*.npz ({time.time()-t0:.0f}s)")
