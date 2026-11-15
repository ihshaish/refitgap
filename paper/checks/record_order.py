#!/usr/bin/env python3
"""Record order behind the crossed reweighting scores (supplement E.3).

Reads: the method-comparison score files of the reweighted pair (intervention stream asrsprior100,
control stream asrsprior00; arms naive, replay, ewc, lwf, replaycur; seeds 0-9), the summary file
of that comparison (run paths and the retained count per period), the two stream parquets, and the
mask builder of the method comparison (build_clean_masks, with the seed and fraction of the
period-1 validation draw).

The score files store no record IDs. Training scored period j's test records as
df[(df.period == j) & (df.split == "test")] in parquet row order, and the batches keep that order,
so the IDs of an array are the acn column of that selection. The script tests that reading:
 (a) stored Y_j (own cell) and Yx_j (crossed cell) equal the parquet labels row by row, under the
     scorer's label index (the sorted label set of its training stream);
 (b) stored own Y_j equals stored crossed Yx_j, every seed and arm;
 (c) fingerprint: 300 to 350 test records per period occur in both streams' test sets. A scorer
     scores such a record in its own array and again in its crossed array; same model and text,
     so the two float16 scores must agree at the parquet-derived positions and disagree under a
     random re-pairing;
 (d) retained IDs intersect neither training pool (which includes the period-1 validation draw),
     and each stream's test and training records are disjoint;
 (e) retained counts equal the stored counts;
 (f) stored Yval equals the parquet labels of the validation draw.

Found: 2400 label-array comparisons, no failure; every count matches; no retained record lies in a
training pool; shared records agree to 0.00049 at the parquet positions (at least 98.7 % exactly),
against a median difference of 0.71 to 0.93 under re-pairing (record_order_summary.json).

Writes record_order_counts.csv, record_order_fingerprint.csv, record_order_summary.json.
    python checks/record_order.py SUMMARY_JSON MASKS_MODULE        (from paper/, streams/ built)
Cannot rerun from this repository alone (the score files and the mask builder are not shipped);
its stored outputs are kept here.
"""
import hashlib, importlib, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
_masks = importlib.import_module(sys.argv[2])
build_clean_masks, VAL_SEED, VAL_FRAC_P1 = _masks.build_clean_masks, _masks.VAL_SEED, _masks.VAL_FRAC_P1

t0 = time.time()
OUT = "checks"
C, X = "asrsprior00", "asrsprior100"            # control, intervention
mc = json.load(open(sys.argv[1]))
TAGS = ["naive", "replay", "ewc", "lwf", "replaycur"]   # run paths read from the summary file
df = {s: pd.read_parquet(f"streams/stream_{s}.parquet") for s in (C, X)}
labels = {s: sorted({l for v in d.labels for l in v.split(";")}) for s, d in df.items()}
N = int(df[C].period.max()) + 1

def Ymat(sel, labs):
    idx = {l: i for i, l in enumerate(labs)}; Y = np.zeros((len(sel), len(labs)), np.int8)
    for r, v in enumerate(sel.labels):
        for l in v.split(";"): Y[r, idx[l]] = 1
    return Y
test = {s: {j: df[s][(df[s].period == j) & (df[s].split == "test")] for j in range(N)} for s in (C, X)}
ids = {s: {j: test[s][j].acn.astype(str).to_numpy() for j in range(N)} for s in (C, X)}
pool = {s: set(df[s].loc[df[s].split == "train_pool", "acn"].astype(str)) for s in (C, X)}
p1 = {s: df[s][(df[s].period == 0) & (df[s].split == "train_pool")] for s in (C, X)}
valsel = {
    s: np.random.default_rng(VAL_SEED).choice(p1[s].index.to_numpy(), int(len(p1[s]) * VAL_FRAC_P1), False)
    for s in (C, X)
}
masks = build_clean_masks("streams")
stored = {e: [c["clean"] for c in mc["clean_mask_counts"][e]["counts"]] for e in (C, X)}

def load(stream, arm, seed): return np.load(mc["runs"][f"{stream}/{arm}/s{seed}"]["path"])
fails, fp_rows, ylab = [], [], {}
rng_null = np.random.default_rng(43)
for arm in TAGS:
    for seed in range(10):
        z = {C: load(C, arm, seed), X: load(X, arm, seed)}
        for tr in (C, X):                       # (f) validation draw
            ok = np.array_equal(z[tr]["Yval"], Ymat(df[tr].loc[valsel[tr]], labels[tr]))
            if not ok: fails.append(f"Yval {arm} s{seed} {tr}")
        for env, oth in ((C, X), (X, C)):
            for j in range(N):
                Yp_own = Ymat(test[env][j], labels[env]); Yp_x = Ymat(test[env][j], labels[oth])
                a = np.array_equal(z[env][f"Y_{j}"], Yp_own)        # (a) own cell
                b = np.array_equal(z[oth][f"Yx_{j}"], Yp_x)         # (a) crossed cell
                c = np.array_equal(z[env][f"Y_{j}"], z[oth][f"Yx_{j}"])   # (b)
                ylab[(arm, seed, env, j)] = (a, b, c)
                if not (a and b and c): fails.append(f"Y {arm} s{seed} env={env} j={j} own={a} crossed={b} equal={c}")
        # (c) fingerprint: scorer trained on `tr`, crossed array on env `oth`, shared records
        for tr, oth in ((C, X), (X, C)):
            for j in range(N):
                pos_o = {a_: i for i, a_ in enumerate(ids[oth][j])}
                pos_t = {a_: i for i, a_ in enumerate(ids[tr][j])}
                sh = [a_ for a_ in ids[oth][j] if a_ in pos_t]
                io = np.array([pos_o[a_] for a_ in sh]); it = np.array([pos_t[a_] for a_ in sh])
                Px = z[tr][f"Px_{N-1}_{j}"].astype(np.float64); Pown = z[tr][f"P_{N-1}_{j}"].astype(np.float64)
                d = np.abs(Px[io] - Pown[it]).max(1)
                dn = np.abs(Px[io] - Pown[rng_null.permutation(it)]).max(1)
                fp_rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        scorer_trained_on=tr,
                        crossed_env=oth,
                        j=j,
                        n_shared=len(sh),
                        max_abs_diff=float(d.max()),
                        frac_exact=float((d == 0).mean()),
                        null_median_diff=float(np.median(dn)),
                        null_frac_within_1e3=float((dn <= 1e-3).mean()),
                    )
                )
fp = pd.DataFrame(fp_rows); fp.to_csv(f"{OUT}/record_order_fingerprint.csv", index=False)

rows = []
for env, oth in ((C, X), (X, C)):
    role = "control" if env == C else "intervention"
    for j in range(N):
        k = np.asarray(masks[env]["mask"][j], bool); ret = ids[env][j][k]
        h = hashlib.sha256("\n".join(ret).encode()).hexdigest()
        rows.append(
            dict(
                environment=role,
                env_stream=env,
                other_stream=oth,
                period_j=j,
                period=j + 1,
                n_test=len(ids[env][j]),
                retained=int(k.sum()),
                stored_clean_mask_count=stored[env][j],
                count_match=int(k.sum()) == stored[env][j],
                retained_cap_control_pool=len(set(ret) & pool[C]),
                retained_cap_intervention_pool=len(set(ret) & pool[X]),
                all_test_cap_own_pool=len(set(ids[env][j]) & pool[env]),
                retained_unique=len(set(ret)) == len(ret),
                sha256_retained_ids=h,
                own_scorer=f"{env}-trained P_7_{j}",
                crossed_scorer=f"{oth}-trained Px_7_{j}",
                Y_checks_pass_all_arms_seeds=all(all(ylab[(a_, s_, env, j)]) for a_ in TAGS for s_ in range(10)),
            )
        )
cnt = pd.DataFrame(rows); cnt.to_csv(f"{OUT}/record_order_counts.csv", index=False)
naive = fp[fp.arm == "naive"]
summary = dict(
    arms={a: [os.path.basename(mc["runs"][f"{s_}/{a}/s0"]["path"]) for s_ in (C, X)] for a in TAGS},
    seeds=list(range(10)),
    n_label_array_comparisons=len(ylab) * 3,
    failures=fails,
    counts_all_match=bool(cnt.count_match.all()),
    any_retained_in_a_pool=int(cnt.retained_cap_control_pool.sum() + cnt.retained_cap_intervention_pool.sum()),
    own_test_in_own_pool=int(cnt.all_test_cap_own_pool.sum()),
    fingerprint=dict(
        cells=len(fp),
        shared_per_cell_range=[int(fp.n_shared.min()), int(fp.n_shared.max())],
        max_abs_diff_all=float(fp.max_abs_diff.max()),
        min_frac_exact=float(fp.frac_exact.min()),
        naive_max_abs_diff=float(naive.max_abs_diff.max()),
        naive_min_frac_exact=float(naive.frac_exact.min()),
        null_median_diff_range=[float(fp.null_median_diff.min()), float(fp.null_median_diff.max())],
        null_frac_within_1e3_max=float(fp.null_frac_within_1e3.max()),
    ),
    runtime_s=time.time() - t0,
)
json.dump(summary, open(f"{OUT}/record_order_summary.json", "w"), indent=1)
pd.options.display.width = 250
print(
    cnt[
        [
            "environment",
            "period",
            "n_test",
            "retained",
            "stored_clean_mask_count",
            "retained_cap_control_pool",
            "retained_cap_intervention_pool",
            "all_test_cap_own_pool",
            "Y_checks_pass_all_arms_seeds",
            "sha256_retained_ids",
        ]
    ].to_string(index=False)
)
print(json.dumps(summary, indent=1))
