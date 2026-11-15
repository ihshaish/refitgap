#!/usr/bin/env python3
"""Decomposition B = A + dR on MultiEURLEX (BiGRU, sequential training, frozen GloVe, ten seeds), as
decomposition_benchmarks.py, with dR - A, the difference from ASRS, threshold updates on n = 500 and 200
test records, and U-bar with restricted backward transfer. python scripts/decomposition_multieurlex.py [SEEDS]
Reads results/multieurlex_bigru/eurlex_C2_naive_lam1_s*; writes eurlex_bridge{.parquet,_summary.json} there.
Backs Table D.1 (MultiEURLEX rows). Needs nothing beyond the repository."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from constructed_drifts import fit_exact

ROOT = Path(".")
OUT = ROOT / "results/multieurlex_bigru"
SPLITS, BOOT, BSEED = 200, 20000, 20260920
GRID = np.linspace(0.05, 0.95, 19)
DRAWS, NS = 20, (200, 500)
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else None


def mf1(P, Y, th):
    return f1_score(Y, (P >= th).astype(int), average="macro", zero_division=0)


def fit_grid(P, Y):
    return np.asarray(
        [
            GRID[int(np.argmax([f1_score(Y[:, k], (P[:, k] >= g).astype(int), zero_division=0) for g in GRID]))]
            for k in range(Y.shape[1])
        ]
    )


def split_half_pair(PN, Pj, Y, rng):
    vN, vj = [], []
    for _ in range(SPLITS):
        perm = rng.permutation(len(Y)); h = len(Y) // 2; a, b = perm[:h], perm[h:]
        vN.append(0.5 * (mf1(PN[b], Y[b], fit_exact(PN[a], Y[a])) + mf1(PN[a], Y[a], fit_exact(PN[b], Y[b]))))
        vj.append(0.5 * (mf1(Pj[b], Y[b], fit_exact(Pj[a], Y[a])) + mf1(Pj[a], Y[a], fit_exact(Pj[b], Y[b]))))
    return float(np.mean(vN)), float(np.mean(vj))


def boot_mean(blk, fn):
    rng = np.random.default_rng(BSEED); vals = blk.to_numpy(); n = len(blk)
    draws = np.array([fn(pd.DataFrame(vals[rng.integers(0, n, n)], columns=blk.columns)) for _ in range(BOOT)])
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def signflip_p(x):
    x = np.asarray(x, float); obs = abs(x.mean())
    hits = sum(abs((x * np.asarray(s)).mean()) >= obs - 1e-15 for s in itertools.product((-1.0, 1.0), repeat=len(x)))
    return hits / 2 ** len(x)


rows, table_rows = [], []
for npz in sorted(OUT.glob("eurlex_C2_naive_lam1_s*_scores.npz")):
    seed = int(npz.stem.split("_s")[1].split("_")[0])
    if SEEDS is not None and seed not in SEEDS:
        continue
    z = np.load(npz); th = z["thresholds"].astype(np.float64)
    N = max(int(k.split("_")[1]) for k in z.files if k.startswith("P_")) + 1
    pq = OUT / f"eurlex_C2_naive_lam1_s{seed}.parquet"
    res = pd.read_parquet(pq) if pq.exists() else None
    for j in range(N - 1):
        Y = z[f"Y_{j}"].astype(int); PN = z[f"P_{N-1}_{j}"].astype(np.float64); Pj = z[f"P_{j}_{j}"].astype(np.float64)
        U_N, U_j = mf1(PN, Y, th), mf1(Pj, Y, th)
        V_N, V_j = split_half_pair(PN, Pj, Y, np.random.default_rng(7000 + 100 * seed + j))
        rec = dict(
            seed=seed, j=j, U_N=U_N, U_j=U_j, V_N=V_N, V_j=V_j, B=U_j - U_N, A=V_j - V_N, R_N=V_N - U_N, R_ref=V_j - U_j
        )
        rec["dR"] = rec["R_N"] - rec["R_ref"]; rec["dR_minus_A"] = rec["dR"] - rec["A"]
        assert abs(rec["B"] - (rec["A"] + rec["dR"])) < 1e-9
        for n in NS:
            gains = []
            for d in range(DRAWS):
                srng = np.random.default_rng(100000 + seed * 1000 + j * 40 + d)
                sel = srng.choice(len(Y), min(n, len(Y) // 2), replace=False)
                rest = np.setdiff1d(np.arange(len(Y)), sel)
                gains.append(mf1(PN[rest], Y[rest], fit_grid(PN[sel], Y[sel])) - mf1(PN[rest], Y[rest], th))
            rec[f"gain_n{n}"] = float(np.mean(gains))
        if res is not None:
            r = res[(res.i == N - 1) & (res.j == j)]
            rec["pq_f1_final"] = float(r.macro_f1.iloc[0])
            rec["pq_auprc_bwt_j"] = float(r.macro_auprc.iloc[0] - res[(res.i == j) & (res.j == j)].macro_auprc.iloc[0])
        rows.append(rec)
    Ubar = float(np.mean([mf1(z[f"P_{N-1}_{j}"].astype(np.float64), z[f"Y_{j}"].astype(int), th) for j in range(N)]))
    trow = dict(
        seed=seed,
        Ubar=Ubar,
        bwt_f1_restricted=-float(np.mean([r["B"] for r in rows if r["seed"] == seed and r["j"] >= 1])),
    )
    if res is not None:
        trow["bwt_auprc_restricted"] = float(
            np.mean([r["pq_auprc_bwt_j"] for r in rows if r["seed"] == seed and r["j"] >= 1])
        )
    table_rows.append(trow)
    print(f"  seed {seed} done", flush=True)

t = pd.DataFrame(rows); t.to_parquet(OUT / "eurlex_bridge.parquet", index=False)
if "pq_f1_final" in t:
    gap = float((t.U_N - t.pq_f1_final).abs().max()); print(f"  max |U_N - parquet macro-F1| = {gap:.5f}")
    assert gap < 3e-3
QUANT = (
    ["B", "A", "dR", "dR_minus_A", "R_N", "R_ref"]
    + [f"gain_n{n}" for n in NS]
    + (["pq_auprc_bwt_j"] if "pq_auprc_bwt_j" in t else [])
)
asrs = pd.read_parquet(ROOT / "results/asrs_inherited_rule/bwt_bridge.parquet")
asrs = asrs[(asrs.j >= 1) & (asrs.j <= 6)]
A_blk = asrs.groupby("seed")[["dR_e1", "A"]].mean().to_numpy()

summary = dict(
    splits=SPLITS,
    bootstrap=dict(B=BOOT, seed=BSEED, blocks="seed"),
    seeds=sorted(int(s) for s in t.seed.unique()),
    ranges={},
)
for rname, rsub in (("restricted_j1to6", t[t.j >= 1]), ("conventional_j0to6", t)):
    blk = rsub.groupby("seed")[QUANT].mean(); res = {}
    for q in QUANT:
        res[q] = dict(
            mean=float(blk[q].mean()),
            ci=boot_mean(blk, lambda d, q=q: d[q].mean()),
            sd_seeds=float(blk[q].std(ddof=0)),
            n_seeds_positive=int((blk[q] > 0).sum()),
            n_seeds=int(len(blk)),
        )
    res["dR_minus_A"]["signflip_p_two_sided"] = signflip_p(blk["dR_minus_A"].to_numpy())
    for num, den, name in (
        ("A", "B", "A_over_B"),
        ("dR", "B", "dR_over_B"),
        ("gain_n500", "R_N", "capture_n500"),
        ("gain_n200", "R_N", "capture_n200"),
    ):
        res[name] = dict(
            ratio_of_means=float(blk[num].mean() / blk[den].mean()),
            ci=boot_mean(blk, lambda d, a=num, b=den: d[a].mean() / d[b].mean()),
        )
    res["per_period_mean"] = rsub.groupby("j")[["B", "A", "dR", "R_N", "R_ref"]].mean().round(6).to_dict(orient="index")
    if rname == "restricted_j1to6":
        X = blk[["dR", "A"]].to_numpy(); rng = np.random.default_rng(BSEED); n, m = len(X), len(A_blk)
        diff = np.array([X[rng.integers(0, n, n)].mean(0) - A_blk[rng.integers(0, m, m)].mean(0) for _ in range(BOOT)])
        res["vs_asrs"] = dict(
            dR_minus_asrs=float(X[:, 0].mean() - A_blk[:, 0].mean()),
            ci=[float(np.percentile(diff[:, 0], 2.5)), float(np.percentile(diff[:, 0], 97.5))],
            A_minus_asrs=float(X[:, 1].mean() - A_blk[:, 1].mean()),
            ci_A=[float(np.percentile(diff[:, 1], 2.5)), float(np.percentile(diff[:, 1], 97.5))],
            asrs_dR=float(A_blk[:, 0].mean()),
            asrs_A=float(A_blk[:, 1].mean()),
            asrs_blocks=int(m),
        )
    summary["ranges"][rname] = res
tb = pd.DataFrame(table_rows)
summary["table_row"] = {
    c: dict(mean=float(tb[c].mean()), sd_ddof1=float(tb[c].std(ddof=1)) if len(tb) > 1 else None)
    for c in tb.columns
    if c != "seed"
}
summary["table_row"]["n"] = int(len(tb))
(OUT / "eurlex_bridge_summary.json").write_text(json.dumps(summary, indent=1))
r = summary["ranges"]["restricted_j1to6"]
print(f"seeds {summary['seeds']}")
print(
    f"restricted  B {r['B']['mean']:+.4f} {r['B']['ci']}  A {r['A']['mean']:+.4f} {r['A']['ci']}  dR "
    f"{r['dR']['mean']:+.4f} {r['dR']['ci']}"
)
print(
    f"            dR-A {r['dR_minus_A']['mean']:+.4f} {r['dR_minus_A']['ci']}  positive "
    f"{r['dR_minus_A']['n_seeds_positive']}/{r['dR_minus_A']['n_seeds']}  sign-flip p "
    f"{r['dR_minus_A']['signflip_p_two_sided']:.3f}"
)
print(
    f"            R_N {r['R_N']['mean']:+.4f}  R_ref {r['R_ref']['mean']:+.4f}  capture n500 "
    f"{r['capture_n500']['ratio_of_means']:.2f}  n200 {r['capture_n200']['ratio_of_means']:.2f}"
)
v = r["vs_asrs"]
print(f"            dR - ASRS {v['dR_minus_asrs']:+.4f} {v['ci']}   A - ASRS {v['A_minus_asrs']:+.4f} {v['ci_A']}")
c = summary["ranges"]["conventional_j0to6"]
print(
    f"conventional B {c['B']['mean']:+.4f}  A {c['A']['mean']:+.4f}  dR {c['dR']['mean']:+.4f}  dR-A "
    f"{c['dR_minus_A']['mean']:+.4f} {c['dR_minus_A']['ci']}"
)
print("table row:", json.dumps(summary["table_row"]))
