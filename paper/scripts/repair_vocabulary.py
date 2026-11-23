"""Vocabulary-stream repair (asrsperm100, 10 seeds): the four repairs of the sequential final model on replay's final
buffer (threshold refit, logit adapter, head, full model) minus replay on historical periods 1 to 7 at the shared rule,
as one non-inferiority family. python scripts/repair_vocabulary.py PATHS_JSON OUT_JSON [workers]
PATHS_JSON maps each key of results/expected/repair_vocabulary_inputs.json ("asrsperm100|s0|naive", ...) to a score
file, and "stored_results" to the stored repair results file. Every file is checked against the sha16 listed there.
Backs Table G.2 (vocabulary stream); exit 0 if every number matches the stored results to 1e-12.
Cannot rerun from this repository alone: the 50 score files and the stored results file are not shipped."""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import refitgap as rg
from refitgap import inference

ARMS = {"r1": "naive", "r_aff": "r_aff|final", "r_head": "r_head|final", "r_full": "r_full|final"}
KEYS = ("mean", "sd", "p_t", "p_tost", "p_holm", "p_tost_holm", "p_signflip", "p_ni", "p_ni_holm",
        "p_ni_signflip", "p_inf", "p_inf_holm", "p_inf_signflip")


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def hist_level(task):
    path, seed = task
    ev = rg.evaluate_npz(path, seed, splits=200)
    pp = ev.per_period()
    return float(np.mean([pp[j]["V_shared"] for j in range(7)]))


def main(paths_json, out, workers=4):
    here = Path(__file__).resolve().parents[1]
    exp = json.loads((here / "results/expected/repair_vocabulary_inputs.json").read_text())
    paths = json.loads(Path(paths_json).read_text())
    stored_json = Path(paths["stored_results"])
    want = exp["stored_results_sha16"]
    inputs = []
    for s in range(10):
        for arm in ("naive", "replay", "r_aff|final", "r_head|final", "r_full|final"):
            key = f"asrsperm100|s{s}|{arm}"
            inputs.append(dict(key=key, file=paths[key], seed=s, sha16_expected=exp["score_files_sha16"][key]))
    for r in inputs:
        r["sha16"] = sha16(r["file"])
    bad = [r["file"] for r in inputs if r["sha16"] != r["sha16_expected"]]
    got_json = sha16(stored_json)
    if bad or got_json != want:
        print("input hash mismatch:", bad[:3], got_json, want)
        return 2
    with ProcessPoolExecutor(max_workers=workers) as ex:
        levels = list(ex.map(hist_level, [(str(r["file"]), r["seed"]) for r in inputs]))
    hist = {r["key"]: v for r, v in zip(inputs, levels)}
    members = {rung: np.array([hist[f"asrsperm100|s{s}|{arm}"] - hist[f"asrsperm100|s{s}|replay"]
                               for s in range(10)]) for rung, arm in ARMS.items()}
    fam = inference.family_ni(members)
    stored = json.loads(stored_json.read_text())["streams"]["asrsperm100"]["blocks"]["final"]["families"]["hist"]
    pm = json.loads(stored_json.read_text())["per_model"]
    worst_level = max(abs(hist[k] - pm[k]["hist_sp"]) for k in hist)
    rows, worst = [], 0.0
    for rung in ARMS:
        a, b = fam[rung], stored[rung]
        d = max([abs(a[k] - b[k]) for k in KEYS] + [abs(x - y) for x, y in zip(a["ci95"], b["ci95"])])
        same_class = (a["class"], a["ni"], a["inf"]) == (b["class"], b["ni"], b["inf"])
        worst = max(worst, d)
        rows.append(dict(rung=rung, mean=a["mean"], ci95=a["ci95"], cls=a["class"], ni=a["ni"], inf=a["inf"],
                         stored_mean=b["mean"], max_abs_diff=d, same_class=same_class))
    ok = worst < 1e-12 and worst_level < 1e-12 and all(r["same_class"] for r in rows)
    rec = dict(result="asrsperm100, historical shared-policy level, repair minus replay (family_ni)",
               package_version=rg.__version__, package_file=rg.__file__, stored_json_sha16=got_json, inputs=inputs,
               per_model_max_abs_diff=worst_level, family=rows,
               max_abs_diff=worst, passed=bool(ok))
    Path(out).write_text(json.dumps(rec, indent=1))
    print(f"refitgap {rg.__version__} from {rg.__file__}")
    print(f"inputs: {len(inputs)} score files and the stored results file, all with the expected checksums")
    print(f"per-model historical shared-policy level vs stored hist_sp: max |diff| {worst_level:.1e}")
    for r in rows:
        print(f"  {r['rung']:7s} {r['mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {r['cls']}"
              f"{' INFERIOR' if r['inf'] else ''}  stored {r['stored_mean']:+.4f}  max |diff| {r['max_abs_diff']:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], a[1], int(a[2]) if len(a) > 2 else 4))
