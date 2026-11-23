"""Common inherited thresholds: how much of DistilBERT's lead at the inherited thresholds comes from
the two arms carrying different theta1 vectors. Descriptive, stored scores only. Backs the sentence
on scoring both arms at one arm's vector in supplement Appendix C.2.

    python scripts/carried_rules_common_thresholds.py SCORES_DIR [--out DIR]

For ASRS and MultiEURLEX (six pairs each), the lead over periods 2 to 7 on whole test sets with
(a) each arm at its own theta1, (b) both at the first-named arm's theta1, (c) both at the
comparator's, and (b) - (a), (c) - (a), with paired t intervals and counts by sign; plus the labels
whose theta1 differs within a pair. Definition and stored outputs:
results/carried_rules/common_thresholds/ (DEFINITION.txt, RESULTS.txt, results.json).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carried_rules_core as C                       # noqa: E402

COMPS = [c for c in C.COMPARISONS if c[0] in ("ASRS", "MultiEURLEX")]
PUBLISHED = {"ASRS: replay minus sequential training": (0.0109, 0.0089, 0.0129),
             "MultiEURLEX: replay minus sequential training": (0.0241, 0.0219, 0.0263),
             "MultiEURLEX: replay minus extra updates on current data": (0.0217, 0.0179, 0.0255)}
ASRS_TOP = ("Ground Event / Encounter", "Inflight Event / Encounter")


def fmt(x):
    return f"{x:+.4f}"


def fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def macro_whole(X, th):
    """Macro-F1 on each whole test set of periods 2 to 7 at thresholds th, mean over periods."""
    return float(np.mean([C.f1_from(*C.counts(X["P"][j] >= th, X["Y"][j])).mean() for j in C.PIDS]))


def main():
    a_ = C.arguments(__doc__)
    C.need(C.comparison_files(COMPS))
    out = {}
    for stream, a, b in COMPS:
        name = C.comp_name(stream, a, b)
        labels = None
        rows = dict(a=[], b=[], c=[], listing=[])
        for seed in C.STREAMS[stream]["seeds"]:
            fa, fb = C.arm_file(stream, a, seed), C.arm_file(stream, b, seed)
            XA, XB = C.load_arm(fa), C.load_arm(fb)
            if labels is None:
                z = np.load(fa)
                labels = [str(x) for x in z["labels"]] if "labels" in z.files else [str(k) for k in range(len(XA["theta"]))]
            for j in C.PIDS:
                assert np.array_equal(XA["Y"][j], XB["Y"][j])
            tA, tB = XA["theta"], XB["theta"]
            rows["a"].append(macro_whole(XA, tA) - macro_whole(XB, tB))
            rows["b"].append(macro_whole(XA, tA) - macro_whole(XB, tA))
            rows["c"].append(macro_whole(XA, tB) - macro_whole(XB, tB))
            diff = [dict(label=labels[k], theta1_first=float(tA[k]), theta1_comparator=float(tB[k]),
                         **({"asrs_top_two": labels[k] in ASRS_TOP} if stream == "ASRS" else {}))
                    for k in np.flatnonzero(tA != tB)]
            rows["listing"].append(dict(seed=seed, files=[Path(fa).name, Path(fb).name], n_differ=len(diff),
                                        labels=diff))
        A, B, Cc = (np.array(rows[k]) for k in ("a", "b", "c"))
        res = dict(arms=[a, b], seeds=C.STREAMS[stream]["seeds"],
                   a_own=C.t_summary(A), b_first_theta1=C.t_summary(B), c_comparator_theta1=C.t_summary(Cc),
                   b_minus_a=C.t_summary(B - A), c_minus_a=C.t_summary(Cc - A), differing_labels=rows["listing"])
        pub = PUBLISHED[name]
        mine = (res["a_own"]["mean"], *res["a_own"]["t95"])
        ok = all(round(x, 4) == p for x, p in zip(mine, pub))
        res["reproduction"] = dict(published=list(pub), computed=list(mine), equal_to_4_decimals=ok)
        out[name] = res
        if not ok:
            raise SystemExit(f"STOP: (a) does not reproduce the published G_U for {name}: {mine} vs {pub}")
    (a_.out / "results.json").write_text(json.dumps(out, indent=1))

    def row(lbl, s):
        pp = " ".join(fmt(v) for v in s["per_pair"])
        return f"  {lbl:<34}{fmt(s['mean'])} {fmt_ci(s['t95'])} {s['positive']}+/{s['negative']}-   {pp}"

    L = ["Common inherited thresholds: results",
         "====================================",
         "22 November 2026. Definition: DEFINITION.txt in this folder (designed after all results were",
         "known; descriptive). DistilBERT, final models, macro-F1 (0 to 1) over periods 2 to 7, whole test sets. Leads",
         "are the first-named arm minus the comparator. Brackets: paired t 95% intervals over 6 pairs; n+/n- counts pairs",
         "by sign; per-pair values in seed order 0 to 5. The refitted lead does not depend on theta1 and is not recomputed.",
         "Pairs whose two theta1 vectors are identical give (b) - (a) = (c) - (a) = 0 exactly; they count in neither n+ nor n-.",
         "MultiEURLEX labels are named by their EuroVoc codes as stored in the score files.",
         ""]
    for name, r in out.items():
        rp = r["reproduction"]
        L += [name, "-" * len(name),
              f"  reproduction of published G_U: computed {fmt(rp['computed'][0])} {fmt_ci(rp['computed'][1:])}, "
              f"published {fmt(rp['published'][0])} {fmt_ci(rp['published'][1:])}: "
              f"{'equal to 4 decimals' if rp['equal_to_4_decimals'] else 'NOT EQUAL'}",
              f"  {'quantity':<34}mean    t95                n+/n-  per pair"]
        L += [row("(a) each arm at its own theta1", r["a_own"]),
              row("(b) both at first-named theta1", r["b_first_theta1"]),
              row("(c) both at comparator theta1", r["c_comparator_theta1"]),
              row("(b) - (a)", r["b_minus_a"]),
              row("(c) - (a)", r["c_minus_a"]),
              "  labels whose theta1 differs between the arms (first-named / comparator):"]
        for p in r["differing_labels"]:
            if not p["labels"]:
                L.append(f"    seed {p['seed']}: none")
            for d in p["labels"]:
                tag = (" [one of the two ASRS labels]" if d.get("asrs_top_two") else
                       " [not one of the two ASRS labels]" if "asrs_top_two" in d else "")
                L.append(f"    seed {p['seed']}: {d['label']}  {d['theta1_first']:.4g} / {d['theta1_comparator']:.4g}{tag}")
        L.append("")
    (a_.out / "RESULTS.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
