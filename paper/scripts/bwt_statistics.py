#!/usr/bin/env python3
"""Three statistics: backward transfer on ASRS by condition over both ranges (macro-F1 and macro-AUPRC,
seed-block bootstrap; DistilBERT too), labels per record for every stream, and the cross-fitting
Monte Carlo error on one stored score matrix. Reads results/e1_results.parquet, streams/ and
results/training_regime/; writes results/asrs_inherited_rule/stats_addenda.json. Backs Table D.4 and
the streams table of Appendix B. Needs the stream files for labels per record (not shipped)."""
import json
import numpy as np
import pandas as pd
from exact_fitters import exact_fit_matrix

OUT = "results/asrs_inherited_rule/stats_addenda.json"
BOOT, BSEED = 20000, 20260920
res = dict(generator="bwt_statistics.py")

# backward transfer, restricted (j = 1..6) and conventional (j = 0..6) ranges
df = pd.read_parquet("results/e1_results.parquet")
d = df[(df.stream == "primary") & (df.protocol == "v2") & (df.strategy == "naive")
       & (df.condition.isin(["C1", "C2", "C3", "C4"]))]
N = int(d.i.max()) + 1


def bwt_run(g, metric, jmin):
    fin = g[g.i == N - 1].set_index("j")[metric]
    diag = g[g.i == g.j].set_index("j")[metric]
    return float(np.mean([fin[j] - diag[j] for j in range(jmin, N - 1)]))


rows = []
for (c, s), g in d.groupby(["condition", "seed"]):
    for metric in ("macro_f1", "macro_auprc"):
        rows.append(dict(condition=c, seed=s, metric=metric,
                         restricted=bwt_run(g, metric, 1), conventional=bwt_run(g, metric, 0)))
r = pd.DataFrame(rows)
D = dict(
    source="results/e1_results.parquet (stream primary, protocol v2, strategy naive, C1-C4, 10 seeds each, 2560 rows)",
    definition="BWT per run = mean_j (M[N-1,j] - M[j,j]); restricted j=1..6 (periods 2-7); conventional j=0..6",
    bootstrap=dict(B=BOOT, seed=BSEED, blocks="seed (10); block value = mean over C1-C4"),
    per_condition={},
    block={},
    per_period_pooled={},
)
for metric in ("macro_f1", "macro_auprc"):
    D["per_condition"][metric] = {
        c: dict(
            restricted_mean=float(g.restricted.mean()),
            restricted_sd_ddof1=float(g.restricted.std(ddof=1)),
            restricted_sd_ddof0=float(g.restricted.std(ddof=0)),
            conventional_mean=float(g.conventional.mean()),
            conventional_sd_ddof1=float(g.conventional.std(ddof=1)),
            conventional_sd_ddof0=float(g.conventional.std(ddof=0)),
            n=int(len(g)),
        )
        for c, g in r[r.metric == metric].groupby("condition")
    }
    D["block"][metric] = {}
    for rng_name in ("restricted", "conventional"):
        blk = r[r.metric == metric].groupby("seed")[rng_name].mean().to_numpy()
        assert len(blk) == 10
        rng = np.random.default_rng(BSEED)
        boots = np.array([blk[rng.integers(0, 10, 10)].mean() for _ in range(BOOT)])
        D["block"][metric][rng_name] = dict(mean=float(blk.mean()), seed_se=float(blk.std(ddof=1) / np.sqrt(10)),
                                            ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                            blocks=[float(x) for x in blk])
        print(f"{metric:12s} {rng_name:12s} block mean {blk.mean():+.6f}  seed-SE {blk.std(ddof=1)/np.sqrt(10):.6f}  "
              f"CI [{np.percentile(boots,2.5):+.6f}, {np.percentile(boots,97.5):+.6f}]")
    per = []
    for j in range(N - 1):
        fin = d[(d.i == N - 1) & (d.j == j)].groupby(["condition", "seed"])[metric].mean()
        dia = d[(d.i == j) & (d.j == j)].groupby(["condition", "seed"])[metric].mean()
        per.append(float((fin - dia).mean()))
    D["per_period_pooled"][metric] = per
    print(f"  per-period pooled {metric}: {np.round(per, 6).tolist()}")
res["d_ranges"] = D

dd = df[df.condition == "distilbert"]
N2 = int(dd.i.max()) + 1
DB = dict(source="results/e1_results.parquet condition == 'distilbert'", seeds=sorted(int(s) for s in dd.seed.unique()),
          protocol=sorted(dd.protocol.unique().tolist()), stream=sorted(dd.stream.unique().tolist()), N=N2)
for metric in ("macro_f1", "macro_auprc"):
    for jmin, name in ((1, "restricted"), (0, "conventional")):
        vals = []
        for s, g in dd.groupby("seed"):
            fin = g[g.i == N2 - 1].set_index("j")[metric]; diag = g[g.i == g.j].set_index("j")[metric]
            vals.append(float(np.mean([fin[j] - diag[j] for j in range(jmin, N2 - 1)])))
        DB[f"{metric}_{name}"] = dict(
            mean=float(np.mean(vals)), sd_ddof0=float(np.std(vals)), sd_ddof1=float(np.std(vals, ddof=1)), per_seed=vals
        )
        print(
            f"DistilBERT {metric} {name}: mean {np.mean(vals):+.6f} sd0 {np.std(vals):.6f} sd1 "
            f"{np.std(vals, ddof=1):.6f} n={len(vals)}"
        )
res["d_distilbert"] = DB

# labels per record
E = dict(source="streams/stream_{name}.parquet, field labels split on ';'", streams={})
for s in ("primary", "secondary", "huffpost", "arxiv", "arxivfine", "amazon"):
    st = pd.read_parquet(f"streams/stream_{s}.parquet")
    n = st.labels.astype(str).apply(lambda x: len([l for l in x.split(";") if l]))
    nl = len({l for x in st.labels.astype(str) for l in x.split(";") if l})
    E["streams"][s] = dict(records=int(len(st)), labelset=int(nl), mean_labels_per_record=float(n.mean()),
                           share_multilabel_pct=float(100 * (n > 1).mean()), max_labels=int(n.max()),
                           test_split_mean=float(n[st.split == "test"].mean()) if "split" in st.columns else None)
    print(
        f"{s:10s} records {len(st):7d} labelset {nl:2d} mean labels/record {n.mean():.6f} share>1 "
        f"{100*(n>1).mean():.3f}% max {n.max()}"
    )
res["e_labels_per_record"] = E

# Monte Carlo standard error of cross-fitting, 200 new partitions
def mf1(P, Y, th):
    pred = P >= th; tp = (pred & (Y == 1)).sum(0)
    return float((2 * tp / (pred.sum(0) + Y.sum(0) + 1e-12)).mean())


F_FILE = "results/training_regime/primary_C4_naive_lam0.1_ep10_s0_scores.npz"
z = np.load(F_FILE); Nl = 8
rng = np.random.default_rng(12345); B = 200
pA, pV = [], []
for b in range(B):
    dA, vf = [], []
    for j in range(1, Nl - 1):
        Y = z[f"Y_{j}"]; Pf = z[f"P_{Nl-1}_{j}"].astype(np.float64); Pc = z[f"P_{j}_{j}"].astype(np.float64)
        perm = rng.permutation(len(Y)); h = len(Y) // 2; S, T = perm[:h], perm[h:]
        Vf = .5 * (mf1(Pf[T], Y[T], exact_fit_matrix(Pf[S], Y[S])) + mf1(Pf[S], Y[S], exact_fit_matrix(Pf[T], Y[T])))
        Vc = .5 * (mf1(Pc[T], Y[T], exact_fit_matrix(Pc[S], Y[S])) + mf1(Pc[S], Y[S], exact_fit_matrix(Pc[T], Y[T])))
        dA.append(Vf - Vc); vf.append(Vf)
    pA.append(np.mean(dA)); pV.append(np.mean(vf))
a, v = np.array(pA), np.array(pV)
res["f_mc_se"] = dict(
    file=F_FILE,
    fitter="exact (m1_fitters.exact_fit_matrix)",
    B=B,
    rng_seed=12345,
    range="j=1..6",
    attainable_diff=dict(
        mean=float(a.mean()),
        sd_over_splits=float(a.std(ddof=1)),
        mc_se_B200=float(a.std(ddof=1) / np.sqrt(200)),
        mc_se_B20=float(a.std(ddof=1) / np.sqrt(20)),
    ),
    V_final_level=dict(
        mean=float(v.mean()),
        sd_over_splits=float(v.std(ddof=1)),
        mc_se_B200=float(v.std(ddof=1) / np.sqrt(200)),
        mc_se_B20=float(v.std(ddof=1) / np.sqrt(20)),
    ),
)
print(
    f"MC: attainable diff mean {a.mean():+.6f} sd {a.std(ddof=1):.6f} SE(B=200) {a.std(ddof=1)/np.sqrt(200):.6f} "
    f"SE(B=20) {a.std(ddof=1)/np.sqrt(20):.6f}"
)
print(
    f"    V(final) mean {v.mean():.6f} sd {v.std(ddof=1):.6f} SE(B=200) {v.std(ddof=1)/np.sqrt(200):.6f} SE(B=20) "
    f"{v.std(ddof=1)/np.sqrt(20):.6f}"
)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
