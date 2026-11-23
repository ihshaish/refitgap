"""Carried rules, item 3 (plans/carried-rules.txt): replay's lead over LwF on MultiEURLEX with the BiGRU,
as for Table C.1 (periods 2 to 7, ten blocks sharing the first-period model, theta1, labels and partitions):
at the inherited thresholds, after both refit, and the change, with percentile bootstrap intervals (20000
resamples, default_rng(20260920)) and counts by sign. Backs the BiGRU LwF row of supplement Table C.1.

    python scripts/carried_rules_lwf.py SCORES_DIR [--out DIR] [--workers N]

Sequential and replay runs: results/multieurlex_bigru/ (in the repository); LwF runs:
eurlex_C2_lwf_a0.1_T2_lam1_s0-9_scores.npz in SCORES_DIR (not in the repository). Writes item3.json and
raw/item3.pkl; compares every arm's levels per block with the stored results/carried_rules/item3.json.
"""
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carried_rules_core as C                       # noqa: E402

SEEDS = list(range(10))
STORED = C.PAPER / "results" / "carried_rules" / "item3.json"
FILES = {}


def sharing_checks():
    rows = {}
    for s in SEEDS:
        z = {k: np.load(f.format(s=s)) for k, f in FILES.items()}
        ref = z["sequential training"]
        N = 1 + max(int(k.split("_")[1]) for k in ref.files if k.startswith("P_"))
        rows[s] = dict(
            first_period_model_identical=all(np.array_equal(ref[f"P_0_{k}"], z[a][f"P_0_{k}"]) for a in z for k in range(N)),
            theta1_identical=all(np.array_equal(ref["thresholds"], z[a]["thresholds"]) for a in z),
            labels_identical=all(np.array_equal(ref[f"Y_{j}"], z[a][f"Y_{j}"]) for a in z for j in range(N)),
            final_models_differ=all(not np.array_equal(z["replay"][f"P_{N - 1}_{j}"], z["LwF"][f"P_{N - 1}_{j}"])
                                    for j in range(N)),
            n_periods=N)
    return rows


def main():
    a = C.arguments(__doc__)
    FILES.update({"sequential training": str(C.BIGRU_EURLEX / "eurlex_C2_naive_lam1_s{s}_scores.npz"),
                  "replay": str(C.BIGRU_EURLEX / "eurlex_C2_replay_lam1_s{s}_scores.npz"),
                  "LwF": str(C.SCORES / "eurlex_C2_lwf_a0.1_T2_lam1_s{s}_scores.npz")})
    C.need([f.format(s=s) for f in FILES.values() for s in SEEDS])
    t0 = time.time()
    chk = sharing_checks()
    tasks = [(f.format(s=s), s, C.SPLITS) for f in FILES.values() for s in SEEDS]
    keys = [(arm, s) for arm in FILES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = dict(zip(keys, ex.map(C.refit_arm, tasks)))
    U, V, Uh = {}, {}, {}
    for (arm, s), r in res.items():
        U[(arm, s)], _ = C.arm_U(r)
        V[(arm, s)], _ = C.arm_V(r, 1)
        Uh[(arm, s)], _ = C.arm_V(r, 0)
    worst = None
    if STORED.is_file():
        st = json.loads(STORED.read_text())["levels"]
        worst = max(max(abs(U[(arm, s)] - st[arm]["U_per_block"][i]), abs(V[(arm, s)] - st[arm]["V_per_block"][i]))
                    for arm in FILES for i, s in enumerate(SEEDS))
    part_same = all(len({tuple(res[(arm, s)]["partition_hash"]) for arm in FILES}) == 1 for s in SEEDS)
    lab_same = all(len({tuple(res[(arm, s)]["label_hash"]) for arm in FILES}) == 1 for s in SEEDS)

    def contrast(x, y):
        gu = np.array([U[(x, s)] - U[(y, s)] for s in SEEDS])
        gv = np.array([V[(x, s)] - V[(y, s)] for s in SEEDS])
        return C.boot_summary([gu, gv, gv - gu], ["G_U", "G_V", "change"])

    out = dict(
        plan_sha256=C.sha256(C.PLAN), seeds=SEEDS, files={k: Path(v).name for k, v in FILES.items()},
        sharing_checks=chk, partitions_identical_across_arms=part_same, labels_identical_by_hash=lab_same,
        largest_difference_from_stored_levels=worst,
        replay_minus_LwF=contrast("replay", "LwF"),
        orientation=dict(replay_minus_sequential=contrast("replay", "sequential training"),
                         LwF_minus_sequential=contrast("LwF", "sequential training")),
        levels={arm: dict(U=float(np.mean([U[(arm, s)] for s in SEEDS])),
                          V=float(np.mean([V[(arm, s)] for s in SEEDS])),
                          U_same_folds=float(np.mean([Uh[(arm, s)] for s in SEEDS])),
                          U_per_block=[U[(arm, s)] for s in SEEDS], V_per_block=[V[(arm, s)] for s in SEEDS])
                for arm in FILES},
        bootstrap=dict(resamples=C.BOOT, seed=C.BOOT_SEED, note="one index matrix shared by G_U, G_V and change"),
        seconds=time.time() - t0)
    (a.out / "raw").mkdir(exist_ok=True)
    with open(a.out / "raw/item3.pkl", "wb") as fh:
        pickle.dump(dict(results={f"{arm}|{s}": r for (arm, s), r in res.items()}), fh, protocol=4)
    (a.out / "item3.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"item 3 done in {time.time() - t0:.0f} s; largest difference from the stored levels {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
