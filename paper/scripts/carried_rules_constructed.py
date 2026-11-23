"""Carried rules, item 4 (plans/carried-rules.txt): the constructed streams under the paired refit. Per
manipulation, the change in replay's lead over sequential training on refitting (G_V - G_U, as for Table C.1,
periods 2 to 7, BiGRU) on the manipulated stream minus the same change on its control, paired by
initialisation; mean, percentile bootstrap (20000, default_rng(20260920)), counts by sign. The plan's condition
(every file present with periods 2 to 7, labels and theta1 shared by the arms) is checked first; if it fails
the item is dropped and the reason written to ITEM4_STATUS.txt. Backs supplement Table C.9.

    python scripts/carried_rules_constructed.py SCORES_DIR [--out DIR] [--workers N]

Reads the method comparison's runs (not in the repository) from SCORES_DIR:
{asrsprior100, asrsprior00, asrsperm100, primary}_C2_{naive,replay}_lam1_s0-9_scores.npz (the primary files
of the method comparison, not those in results/asrs_bigru_paired/). Writes item4.json and raw/item4.pkl.
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
STREAMS = {"reweighted stream": "asrsprior100", "control of the reweighted stream": "asrsprior00",
           "vocabulary stream": "asrsperm100", "control of the vocabulary stream": "primary"}
MANIP = [("reweighted stream", "control of the reweighted stream"),
         ("vocabulary stream", "control of the vocabulary stream")]
ARMS = {"sequential training": "naive", "replay": "replay"}


def path(stream, arm, s):
    return C.SCORES / f"{STREAMS[stream]}_C2_{ARMS[arm]}_lam1_s{s}_scores.npz"


def condition():
    missing = []
    for st in STREAMS:
        for s in SEEDS:
            z = {}
            for a in ARMS:
                p = path(st, a, s)
                if not p.is_file():
                    missing.append(f"{p.name} missing")
                    continue
                z[a] = np.load(p)
                need = [f"P_7_{j}" for j in C.PIDS] + [f"Y_{j}" for j in C.PIDS] + ["thresholds"]
                lack = [k for k in need if k not in z[a].files]
                if lack:
                    missing.append(f"{p.name} lacks {lack}")
            if len(z) == 2:
                if not all(np.array_equal(z["sequential training"][f"Y_{j}"], z["replay"][f"Y_{j}"]) for j in C.PIDS):
                    missing.append(f"{st} s{s}: labels differ between the arms")
                if not np.array_equal(z["sequential training"]["thresholds"], z["replay"]["thresholds"]):
                    missing.append(f"{st} s{s}: theta1 differs between the arms")
    return missing


def main():
    args = C.arguments(__doc__)
    files = [path(st, a, s) for st in STREAMS for a in ARMS for s in SEEDS]
    if not any(f.is_file() for f in files):
        C.need(files)                                    # none present: say where they should be
    t0 = time.time()
    miss = condition()
    status = args.out / "ITEM4_STATUS.txt"
    if miss:
        status.write_text("Item 4 dropped: the plan's condition is not met.\n" + "\n".join(miss) + "\n")
        print("item 4 dropped")
        return 0
    status.write_text(
        "Item 4 runs: the plan's condition is met.\n"
        "Files: {asrsprior100 (reweighted stream), asrsprior00 (its control), asrsperm100 "
        "(vocabulary stream), primary (its control)}_C2_{naive,replay}_lam1_s0..9_scores.npz, the method comparison "
        "of supplement Appendix F (5 arms x 4 streams x 10 initialisations = 200 runs). Every file holds P_7_j and Y_j for periods "
        "2 to 7 and theta1; within each stream and initialisation the two arms share labels and theta1. The same ten "
        "initialisations (0-9) exist for all four streams.\n"
        f"Checked {time.strftime('%Y-%m-%d %H:%M')} before any outcome of the item.\n")
    tasks = [(str(path(st, a, s)), s, C.SPLITS) for st in STREAMS for a in ARMS for s in SEEDS]
    keys = [(st, a, s) for st in STREAMS for a in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = dict(zip(keys, ex.map(C.refit_arm, tasks)))
    per = {}
    for st in STREAMS:
        U = {a: np.array([C.arm_U(res[(st, a, s)])[0] for s in SEEDS]) for a in ARMS}
        V = {a: np.array([C.arm_V(res[(st, a, s)], 1)[0] for s in SEEDS]) for a in ARMS}
        gu, gv = U["replay"] - U["sequential training"], V["replay"] - V["sequential training"]
        per[st] = dict(G_U=gu, G_V=gv, change=gv - gu, U=U, V=V)
    out = dict(plan_sha256=C.sha256(C.PLAN), seeds=SEEDS, streams=STREAMS,
               per_stream={st: dict(C.boot_summary([p["G_U"], p["G_V"], p["change"]], ["G_U", "G_V", "change"]),
                                    levels={a: dict(U=float(p["U"][a].mean()), V=float(p["V"][a].mean()))
                                            for a in ARMS})
                           for st, p in per.items()},
               contrasts={}, bootstrap=dict(resamples=C.BOOT, seed=C.BOOT_SEED))
    for m, c in MANIP:
        d = per[m]["change"] - per[c]["change"]
        out["contrasts"][f"{m} minus {c}"] = C.boot_summary(
            [d, per[m]["G_U"] - per[c]["G_U"], per[m]["G_V"] - per[c]["G_V"]],
            ["change_difference", "G_U_difference", "G_V_difference"])
    out["seconds"] = time.time() - t0
    (args.out / "raw").mkdir(exist_ok=True)
    with open(args.out / "raw/item4.pkl", "wb") as fh:
        pickle.dump(dict(results={f"{st}|{a}|{s}": r for (st, a, s), r in res.items()}), fh, protocol=4)
    (args.out / "item4.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"item 4 done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
