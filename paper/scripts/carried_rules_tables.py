#!/usr/bin/env python3
"""Carried rules (plans/carried-rules.txt): the printed tables of every item from the JSON outputs, 4 decimals,
intervals in brackets, pairs by sign. Figure 3 is checked against these lines.

    python scripts/carried_rules_tables.py [DIR] > report_tables.txt

DIR defaults to the stored outputs, results/carried_rules/ (whose report_tables.txt this prints).
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "carried_rules"
NAMES = {"theta1": "a inherited thresholds", "cut0.5": "b cut of 0.5", "own_val": "c own validation",
         "prev_period": "d previous period's rule", "matched_rate": "f matched rate"}
PUB = {"ASRS": dict(G_U=(+0.0109, +0.0089, +0.0129), G_V=(-0.0004, -0.0030, +0.0022), change=(-0.0113, -0.0138, -0.0088)),
       "MultiEURLEX": dict(G_U=(+0.0217, +0.0179, +0.0255), G_V=(+0.0198, +0.0157, +0.0240),
                           change=(-0.0019, -0.0027, -0.0011))}


def ld(n):
    p = OUT / n
    return json.loads(p.read_text()) if p.is_file() else None


def t(x, key="t95"):
    return f"{x['mean']:+.4f} [{x[key][0]:+.4f}, {x[key][1]:+.4f}] {x['positive']}+/{x['negative']}-" + \
        (f"/{x['zero']}0" if x.get("zero") else "")


def main():
    i1 = ld("item1.json")
    print("ITEM 1")
    for c, v in i1["comparisons"].items():
        a, b = v["arms"]
        print(f"\n{c} ({v['n_pairs']} pairs). G_V {t(v['G_V'])}; G_V with 500 records {t(v['G_V500'])}")
        print(f"  refitted levels: {a} {v['levels_refit'][a]:.4f}, {b} {v['levels_refit'][b]:.4f}; 500-record refit: "
              f"{a} {v['levels_refit500'][a]:.4f}, {b} {v['levels_refit500'][b]:.4f}")
        print(f"  {'rule':26s} {'lead G_r (same folds)':34s} {'G_r - G_V':34s} level {a} / {b}")
        for r, e in v["rules"].items():
            lv = e["levels_same_folds"]
            print(f"  {NAMES[r]:26s} {t(e['G_same_folds']):34s} {t(e['G_minus_G_V']):34s} {lv[a]:.4f} / {lv[b]:.4f}")
            if "G_minus_G_V500" in e:
                print(f"  {'':26s} {'G_d - G_V500':34s} {t(e['G_minus_G_V500'])}")
            if "G_whole_period" in e:
                lw = e["levels_whole_period"]
                print(f"  {'   whole period':26s} {t(e['G_whole_period']):34s} {'':34s} {lw[a]:.4f} / {lw[b]:.4f}")
    i2a = {s: ld(f"item2a_{s}.json") for s in ("ASRS", "MultiEURLEX")}
    print("\nITEM 2a")
    for s, v in i2a.items():
        if not v or "summary" not in v:
            print(f"  {s}: not run")
            continue
        print(f"\n{s}, {v['comparison']}: {v['replicates']} replicates, {v['seconds'] / 60:.1f} min on {v['workers']} workers")
        for q in ("G_U", "G_V", "change"):
            x = v["summary"][q]
            p = PUB[s][q]
            lab = "G_U (same folds)" if q == "G_U" else q
            print(f"  {lab:18s} mean of replicates {x['mean_of_replicates']:+.4f}  percentile [{x['percentile95'][0]:+.4f}, "
                  f"{x['percentile95'][1]:+.4f}]   published {p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}] (t, over initialisations)")
        for q in ("U_a", "U_b", "V_a", "V_b"):
            x = v["summary"][q]
            print(f"  {q:18s} mean {x['mean_of_replicates']:.4f}  percentile [{x['percentile95'][0]:.4f}, {x['percentile95'][1]:.4f}]")
    i2b = ld("item2b.json")
    print("\nITEM 2b")
    for c, v in i2b["comparisons"].items():
        print(f"\n{c}: macro G_U {t(v['macro']['G_U'])}, G_V {t(v['macro']['G_V'])}, change {t(v['macro']['change'])}; "
              f"parts sum to the macro values to {v['sum_of_parts_largest_difference']:.1e}")
        print(f"  {'label':38s} {'positives, periods 2-7':30s} {'part of G_U':32s} {'part of G_V':32s} {'part of change':32s}")
        for r in v["labels"]:
            print(f"  {r['label'][:38]:38s} {' '.join(map(str, r['positives_per_period'])):30s} {t(r['part_G_U']):32s} "
                  f"{t(r['part_G_V']):32s} {t(r['part_change']):32s}")
        print(f"  label left out: {'G_U':32s} {'G_V':32s} {'change':32s}")
        for r in v["labels"]:
            print(f"  {r['label'][:38]:38s} {t(r['left_out_G_U']):32s} {t(r['left_out_G_V']):32s} {t(r['left_out_change']):32s}")
        print("  two labels with the largest part of the change: per 1000 test records of the scoring halves")
        for lab, e in v["errors_top_two"].items():
            print(f"   {lab} (positives per 1000 scoring records {e['positives_per_1000_scoring']:.1f})")
            for k, x in e.items():
                if "|" in k:
                    arm, rule = k.split("|")
                    print(f"     {arm:30s} {('inherited thresholds' if rule == 'theta1' else 'refitted'):22s} "
                          f"FP {x['fp_per_1000']:.1f}  FN {x['fn_per_1000']:.1f}")
    i3 = ld("item3.json")
    print("\nITEM 3")
    if i3:
        for nm, blk in (("replay minus LwF", i3["replay_minus_LwF"]),
                        ("replay minus sequential training", i3["orientation"]["replay_minus_sequential"]),
                        ("LwF minus sequential training", i3["orientation"]["LwF_minus_sequential"])):
            print(f"  {nm}")
            for q in ("G_U", "G_V", "change"):
                print(f"    {q:7s} {t(blk[q], 'boot95')}   per block " + " ".join(f"{x:+.4f}" for x in blk[q]["per_pair"]))
        print("  levels " + "; ".join(f"{a}: U {v['U']:.4f}, V {v['V']:.4f}" for a, v in i3["levels"].items()))
        worst = i3.get("largest_difference_from_stored_per_period", i3.get("largest_difference_from_stored_levels"))
        print(f"  largest difference from stored values {worst if worst is None else format(worst, '.1e')}; "
              f"sharing checks all true: {all(all(x for k, x in r.items() if k != 'n_periods') for r in i3['sharing_checks'].values())}; "
              f"partitions identical {i3['partitions_identical_across_arms']}")
    i4 = ld("item4.json")
    print("\nITEM 4")
    if i4:
        for st, v in i4["per_stream"].items():
            print(f"  {st:34s} G_U {t(v['G_U'], 'boot95')}  G_V {t(v['G_V'], 'boot95')}  change {t(v['change'], 'boot95')}")
            print(f"  {'':34s} levels " + "; ".join(f"{a}: U {x['U']:.4f}, V {x['V']:.4f}" for a, x in v["levels"].items()))
        for c, v in i4["contrasts"].items():
            print(f"  {c}: change difference {t(v['change_difference'], 'boot95')}; G_U difference "
                  f"{t(v['G_U_difference'], 'boot95')}; G_V difference {t(v['G_V_difference'], 'boot95')}")
            print("     per initialisation " + " ".join(f"{x:+.4f}" for x in v["change_difference"]["per_pair"]))


if __name__ == "__main__":
    main()
