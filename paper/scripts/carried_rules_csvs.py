"""Carried rules (plans/carried-rules.txt): the CSV outputs, from the raw counts saved by
carried_rules_items1_2b.py, carried_rules_lwf.py and carried_rules_constructed.py and from the item 2a JSON
files: one row per stream, pair, period, arm, rule, source of fitting records, number of fitting records,
partition, half and label, with scoring records, positives, tp, fp, fn and F1 (items 1, 3, 4; gzipped, too
large to ship), and the summaries of items 2a and 2b (item2a_<stream>.csv, item2b_labels.csv,
item2b_errors.csv, stored in results/carried_rules/).

    python scripts/carried_rules_csvs.py SCORES_DIR [--out DIR] [--items 1 2b 2a 3 4]
"""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carried_rules_core as C                       # noqa: E402

OUT = RAW = None
COLS = ["comparison", "stream", "pair", "initialisation", "period", "arm", "rule", "fitting_source", "n_fitting",
        "partition", "half", "label", "scoring_records", "positives", "tp", "fp", "fn", "F1"]
SOURCE = {"theta1": "period 1 validation split, first-period model (stored theta1)",
          "cut0.5": "none", "own_val": "period 1 validation split, final model",
          "prev_period": "500 test records of the previous period",
          "matched_rate": "none (matched share of positive predictions at theta1)",
          "refit": "opposite half of the period", "refit500": "500 records of the opposite half"}


def f1s(tp, fp, fn):
    return C.f1_from(tp, fp, fn)


def write_block(fh, comp, stream, pair, seed, arm, rule, source, nfit, TP, FP, FN, NS, POS, labels):
    """TP etc. (period, partition, half, K) -> rows."""
    NP, S, H, K = TP.shape
    F = f1s(TP, FP, FN)
    lines = []
    for c in range(NP):
        per = C.PIDS[c] + 1
        for b in range(S):
            for d in range(H):
                nf = nfit[c, b, d] if isinstance(nfit, np.ndarray) else nfit
                pre = f"{comp},{stream},{pair},{seed},{per},{arm},{rule},{source},{nf},{b},{d},"
                for k in range(K):
                    lines.append(f"{pre}{labels[k]},{NS[c, b, d]},{POS[c, b, d, k]},{TP[c, b, d, k]},{FP[c, b, d, k]},"
                                 f"{FN[c, b, d, k]},{F[c, b, d, k]:.10g}\n")
    fh.write("".join(lines))


def write_whole(fh, comp, stream, pair, seed, arm, rule, source, nfit, Wm, WN, WPOS, labels):
    """Wm (period, 3, K) whole-period counts."""
    F = f1s(Wm[:, 0], Wm[:, 1], Wm[:, 2])
    for c in range(Wm.shape[0]):
        for k in range(Wm.shape[2]):
            fh.write(f"{comp},{stream},{pair},{seed},{C.PIDS[c] + 1},{arm},{rule},{source},{nfit},whole,whole,"
                     f"{labels[k]},{WN[c]},{WPOS[c, k]},{Wm[c, 0, k]},{Wm[c, 1, k]},{Wm[c, 2, k]},{F[c, k]:.10g}\n")


def label_names(path):
    z = np.load(path)
    return [str(x).replace(",", ";") for x in z["labels"]] if "labels" in z.files else \
        [f"label{k}" for k in range(z["thresholds"].shape[0])]


def item1():
    with gzip.open(OUT / "item1_carried_rules.csv.gz", "wt") as fh:
        fh.write(",".join(COLS) + "\n")
        for n in range(len(C.COMPARISONS)):
            d = pickle.loads((RAW / f"item1_{n}.pkl").read_bytes())
            seeds = [x["seed"] for x in d["results"]]
            labels = label_names(C.arm_file(d["stream"], d["arms"][0], seeds[0]))
            for pi, x in enumerate(d["results"]):
                for m, arm in enumerate(x["arms"]):
                    for r, rule in enumerate(C.RULES):
                        nf = {"theta1": x["n_val"], "own_val": x["n_val"], "cut0.5": 0, "matched_rate": 0,
                              "prev_period": C.PREV_N, "refit": x["NFIT"],
                              "refit500": np.minimum(x["NFIT"], C.LIMITED_N)}[rule]
                        write_block(fh, d["name"], d["stream"], pi + 1, x["seed"], arm,
                                    f"{C.RULE_LETTER[rule]} {rule}", SOURCE[rule], nf, x["TP"][m, r], x["FP"][m, r],
                                    x["FN"][m, r], x["NS"], x["POS"], labels)
                    for rule in C.WHOLE_RULES:
                        nf = 0 if rule == "cut0.5" else x["n_val"]
                        write_whole(fh, d["name"], d["stream"], pi + 1, x["seed"], arm,
                                    f"{C.RULE_LETTER[rule]} {rule} (whole period)", SOURCE[rule], nf,
                                    x["W"][rule][m], x["WN"], x["WPOS"], labels)
    print("item1 csv written")


def item2b():
    j = json.loads((OUT / "item2b.json").read_text())
    with open(OUT / "item2b_labels.csv", "w") as fh:
        fh.write("comparison,label,quantity,mean,t95_low,t95_high,positive_pairs,negative_pairs,zero_pairs,"
                 "positives_periods_2_to_7\n")
        for comp, v in j["comparisons"].items():
            for row in v["labels"]:
                for q in ("part_G_U", "part_G_V", "part_change", "left_out_G_U", "left_out_G_V", "left_out_change"):
                    t = row[q]
                    fh.write(f"{comp},{row['label'].replace(',', ';')},{q},{t['mean']:.10g},{t['t95'][0]:.10g},"
                             f"{t['t95'][1]:.10g},{t['positive']},{t['negative']},{t['zero']},"
                             f"{' '.join(map(str, row['positives_per_period']))}\n")
    with open(OUT / "item2b_errors.csv", "w") as fh:
        fh.write("comparison,label,arm,thresholds,fp_per_1000,fn_per_1000,tp_per_1000\n")
        for comp, v in j["comparisons"].items():
            for lab, e in v["errors_top_two"].items():
                for key, x in e.items():
                    if "|" not in key:
                        continue
                    arm, rule = key.split("|")
                    fh.write(f"{comp},{lab.replace(',', ';')},{arm},{rule},{x['fp_per_1000']:.10g},"
                             f"{x['fn_per_1000']:.10g},{x['tp_per_1000']:.10g}\n")
    print("item2b csv written")


def bigru(pkl, name, keyfun):
    d = pickle.loads((RAW / pkl).read_bytes())["results"]
    with gzip.open(OUT / name, "wt") as fh:
        fh.write(",".join(COLS) + "\n")
        for key, x in d.items():
            comp, stream, arm, seed = keyfun(key)
            labels = label_names(x["path"])
            for r, rule in enumerate(("theta1", "refit")):
                nf = 0 if rule == "theta1" else x["NFIT"]
                src = "stored theta1" if rule == "theta1" else SOURCE["refit"]
                write_block(fh, comp, stream, seed + 1, seed, arm, f"{rule}", src, nf, x["TP"][r], x["FP"][r], x["FN"][r],
                            x["NS"], x["POS"], labels)
            write_whole(fh, comp, stream, seed + 1, seed, arm, "theta1 (whole period)", "stored theta1", 0, x["W"],
                        x["WN"], x["WPOS"], labels)
    print(name, "written")


def item2a():
    for f in sorted(OUT.glob("item2a_*.json")):
        j = json.loads(f.read_text())
        if "summary" not in j:
            continue
        with open(OUT / (f.stem + ".csv"), "w") as fh:
            fh.write("stream,comparison,replicate,pairs_drawn,G_U_same_folds,G_V,change,U_replay,U_comparator,V_replay,"
                     "V_comparator\n")
            for r in j["rows"]:
                fh.write(f"{j['stream']},{j['comparison']},{r['replicate']},{' '.join(map(str, r['pairs']))},"
                         f"{r['G_U']:.10g},{r['G_V']:.10g},{r['change']:.10g},{r['U_a']:.10g},{r['U_b']:.10g},"
                         f"{r['V_a']:.10g},{r['V_b']:.10g}\n")
        print(f.stem, "csv written")


def main():
    global OUT, RAW
    ap = argparse.ArgumentParser(description="CSV outputs of the carried-rule items")
    ap.add_argument("scores", help="directory holding the score files (label names are read from them)")
    ap.add_argument("--out", default=str(C.DEFAULT_OUT))
    ap.add_argument("--items", nargs="*", default=["1", "2b", "2a", "3", "4"])
    a = ap.parse_args()
    C.SCORES = Path(a.scores)
    OUT = Path(a.out)
    RAW = OUT / "raw"
    which = a.items
    if "1" in which:
        item1()
    if "2b" in which:
        item2b()
    if "2a" in which:
        item2a()
    if "3" in which and (RAW / "item3.pkl").is_file():
        bigru("item3.pkl", "item3_lwf_replay.csv.gz",
              lambda k: ("MultiEURLEX BiGRU: replay, LwF, sequential training", "MultiEURLEX", k.split("|")[0],
                         int(k.split("|")[1])))
    if "4" in which and (RAW / "item4.pkl").is_file():
        bigru("item4.pkl", "item4_constructed.csv.gz",
              lambda k: ("replay minus sequential training (BiGRU)", k.split("|")[0], k.split("|")[1],
                         int(k.split("|")[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
