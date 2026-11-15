"""MultiEURLEX repair (BiGRU, frozen GloVe, seeds 500-505): full-model repair on replay's final buffer minus replay
(historical periods 1 to 7 and current period 8) and minus sequential training (period 8), at the shared rule, each
a family of one against the 0.005 margin. python scripts/repair_multieurlex.py SCORES_DIR OUT_JSON [workers]
Backs Table G.2 and Main Table 3 (MultiEURLEX); exit 0 if every level matches the stored analysis to 1e-12.
SCORES_DIR holds runs/, repair/ and analysis/meur_analysis.json. Cannot rerun from this repository alone: those
files are not shipped; the expected output is results/expected/repair_multieurlex.json."""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import refitgap as rg
from refitgap import inference

SEEDS = range(500, 506)
MARGIN = 0.005
ANALYSIS_SHA16 = "31a0e7c6eae2a46a"


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def files(pb, s):
    return {"naive": pb / f"runs/eurlex_C2_naive_lam1_s{s}_scores.npz",
            "replay": pb / f"runs/eurlex_C2_replay_lam1_s{s}_scores.npz",
            "r_full": pb / f"repair/eurlex_C2_naive_lam1_s{s}__r_full__final_scores.npz"}


def levels(task):
    path, seed = task
    pp = rg.evaluate_npz(str(path), seed, splits=200).per_period()
    return float(np.mean([pp[j]["V_shared"] for j in range(7)])), float(pp[7]["V_shared"])


def reading(st):
    if st.get("ni"):
        return "non-inferior within the margin"
    if st.get("inf"):
        return "inferior beyond the margin"
    return "unresolved against the margin"


def main(pb, out, workers=4):
    pb = Path(pb)
    aj = pb / "analysis/meur_analysis.json"
    if sha16(aj) != ANALYSIS_SHA16:
        sys.exit(f"{aj} is not the expected analysis file ({sha16(aj)}, expected {ANALYSIS_SHA16})")
    stored = json.loads(aj.read_text())
    tasks = [(f, s, arm) for s in SEEDS for arm, f in files(pb, s).items()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        got = dict(zip([(arm, s) for _, s, arm in tasks], ex.map(levels, [(f, s) for f, s, _ in tasks])))
    worst = 0.0
    for (arm, s), (h, c) in got.items():
        m = stored["metrics"][f"{arm}|s{s}"]
        worst = max(worst, abs(h - m["hist_sp"]), abs(c - m["cur_sp"]))
    d = lambda a, b, k: np.array([got[(a, s)][k] - got[(b, s)][k] for s in SEEDS])
    members = {"hist": d("r_full", "replay", 0), "curR": d("r_full", "replay", 1), "curN": d("r_full", "naive", 1)}
    res = {}
    for name, x in members.items():
        st = inference.family_ni({name: x}, margin=MARGIN)[name]   # each a family of one
        reg = stored["principal"]["families"][name]
        res[name] = dict(values=x.tolist(), mean=st["mean"], ci95=st["ci95"], p_t=st["p_t"], ni=st["ni"],
                         inf=st["inf"], reading=reading(st), planned_reading=reading(reg))
    ok = worst <= 1e-12 and all(r["reading"] == r["planned_reading"] for r in res.values())
    print(f"levels: {len(got)} recomputed; largest difference from the stored values {worst:.1e}")
    for name, r in res.items():
        nz = "excludes zero" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "includes zero"
        print(f"{name}: mean {r['mean']:+.4f}, 95 % t-interval [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
              f"({nz}; two-sided p {r['p_t']:.2g}); against -{MARGIN}: {r['reading']}")
    print("Reading hist: the repair is below replay (the interval excludes zero), and whether the gap exceeds the "
          "0.005 margin is not decided (the interval crosses -0.005). A nonzero difference is not a practical "
          "inferiority, and an unresolved comparison is not non-inferiority.")
    Path(out).write_text(json.dumps(dict(analysis_sha16=ANALYSIS_SHA16, max_level_diff=worst, results=res,
                                         fe_version=rg.__version__, ok=ok), indent=1))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
