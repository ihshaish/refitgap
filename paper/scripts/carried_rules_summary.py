"""Carried rules (plans/carried-rules.txt): SUMMARY.json from the item outputs, with the seeds of every draw
and the sha256 of the plan and of every input file found. Run after the item scripts:

    python scripts/carried_rules_summary.py [SCORES_DIR] [--out DIR]

Reads item1.json, item2a_<stream>.json, item2b.json, item3.json, item4.json, ITEM4_STATUS.txt and, when present,
reproduction.json (the check of the published DistilBERT and PubMed rows run before the items) from the output
directory. The stored SUMMARY.json (results/carried_rules/) carries the hash of the dated original of the plan.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carried_rules_core as C                       # noqa: E402


def input_files(scores):
    files = [C.PLAN, C.PAPER / "results" / "pubmed_paired_refit" / "pubmed_paired_refit.json",
             Path(__file__).resolve().parent / "paired_refit_pubmed.py"]
    if scores is not None:
        C.SCORES = Path(scores)
        files += C.comparison_files()
        files += [C.arm_file("PubMed", "LwF", s) for s in range(8)]
        files += [C.SCORES / f"eurlex_C2_lwf_a0.1_T2_lam1_s{s}_scores.npz" for s in range(10)]
        files += [C.SCORES / f"{st}_C2_{a}_lam1_s{s}_scores.npz"
                  for st in ("asrsprior100", "asrsprior00", "asrsperm100", "primary") for a in ("naive", "replay")
                  for s in range(10)]
    files += [C.BIGRU_EURLEX / f"eurlex_C2_{a}_lam1_s{s}_scores.npz" for a in ("naive", "replay") for s in range(10)]
    out = {}
    for f in files:
        f = Path(f)
        try:
            key = str(f.relative_to(C.PAPER))
        except ValueError:
            key = f.name
        out[key] = C.sha256(f)
    return out


def main():
    a = C.arguments(__doc__, scores=False) if len(sys.argv) < 2 or sys.argv[1].startswith("-") else None
    scores = None
    if a is None:
        scores = sys.argv.pop(1)
        a = C.arguments(__doc__, scores=False)
    j = lambda name: json.loads((a.out / name).read_text()) if (a.out / name).is_file() else None   # noqa: E731
    rep, i1, i2b, i3, i4 = j("reproduction.json"), j("item1.json"), j("item2b.json"), j("item3.json"), j("item4.json")
    if i1 is None:
        raise SystemExit(f"no item1.json in {a.out}: run carried_rules_items1_2b.py first")
    i2a = {st: j(f"item2a_{st}.json") for st in ("ASRS", "MultiEURLEX")}
    status = a.out / "ITEM4_STATUS.txt"
    summary = dict(
        plan_file="plans/carried-rules.txt", plan_sha256=C.sha256(C.PLAN), written=time.strftime("%Y-%m-%d %H:%M"),
        step0=(None if rep is None else dict(
            result_pass=rep["result_pass"], largest_difference=rep["largest_difference"],
            largest_full_precision_difference=rep["largest_full_precision_difference"],
            numbers_compared=len(rep["rows"]), rowcheck=rep["rowcheck"])),
        seeds=dict(partitions="default_rng(7000 + 100*seed + period id)",
                   rule_d="default_rng(11000 + 100*seed + period id)",
                   refit500="default_rng(9000 + 100*seed + period id)", item2a="default_rng(20261118)",
                   bootstrap_bigru="default_rng(20260920), 20000 resamples"),
        item1=i1, item2a=i2a, item2b=i2b, item3=i3, item4=i4,
        item4_status=status.read_text() if status.is_file() else None,
        input_sha256=input_files(scores))
    (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"SUMMARY.json written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
