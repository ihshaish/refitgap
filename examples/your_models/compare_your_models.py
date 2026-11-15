"""Compare your own models: method a against method b, at the carried rule and after both refit.

    python examples/your_models/compare_your_models.py [DATA_DIR] [--a mine] [--b baseline] [--splits 200]

DATA_DIR (default data/, written by make_data.py) holds
  labels.csv                    period, record_id, one 0/1 column per label
  scores_<method>_seed<k>.csv   period, record_id, one score column per label (final model, seed k)
  thresholds.csv                method, seed, one column per label: the carried rule
Scores are probabilities in [0, 1] (sigmoid your logits first); a score equal to its threshold is
a positive. Records match by (period, record_id), labels by column name, in any row order. A
missing or extra record, a duplicate id, a different label column or a seed held by one method
only raises: nothing is dropped quietly. A label with no positive in a period has F1 0 there.

Out: per seed and over seeds, a's lead over b at the carried rule (G_U), after both refit
(G_V; partitions from default_rng(7000 + 100*seed + period)) and the change G_V - G_U, with paired
t 95% intervals. Positive means a scores higher.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import refitgap as rg

HERE = Path(__file__).resolve().parent


def read_table(path, key_cols):
    """CSV -> (label names, {key: row of floats}); refuses duplicate keys."""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    labels = header[len(key_cols):]
    if header[:len(key_cols)] != key_cols or not labels:
        raise ValueError(f"{path.name}: expected columns {key_cols} followed by one column per label")
    out = {}
    for r in body:
        key = tuple(r[:len(key_cols)])
        if key in out:
            raise ValueError(f"{path.name}: duplicate row {key}")
        out[key] = np.array(r[len(key_cols):], dtype=float)
    return labels, out


def per_period(table, periods):
    """{(period, record_id): row} -> {period: (record ids, matrix)}."""
    out = {}
    for j in periods:
        ids = sorted(rid for p, rid in table if p == j)
        out[j] = (ids, np.vstack([table[(j, rid)] for rid in ids]))
    return out


def load(data, a, b):
    labels, lab = read_table(data / "labels.csv", ["period", "record_id"])
    periods = sorted({p for p, _ in lab}, key=int)
    Y = per_period(lab, periods)
    names, th = read_table(data / "thresholds.csv", ["method", "seed"])
    if names != labels:
        raise ValueError("thresholds.csv: label columns differ from labels.csv")
    runs = {}
    for m in (a, b):
        seeds = sorted(int(s) for meth, s in th if meth == m)
        if not seeds:
            raise ValueError(f"thresholds.csv: no rows for method {m!r}")
        for s in seeds:
            names, sc = read_table(data / f"scores_{m}_seed{s}.csv", ["period", "record_id"])
            if names != labels:
                raise ValueError(f"scores_{m}_seed{s}.csv: label columns differ from labels.csv")
            extra = sorted({p for p, _ in sc} - set(periods))
            if extra:
                raise ValueError(f"scores_{m}_seed{s}.csv: periods {extra} have no labels")
            runs[(m, s)] = (per_period(sc, periods), th[(m, str(s))])
    sa, sb = {s for m, s in runs if m == a}, {s for m, s in runs if m == b}
    if sa != sb:
        raise ValueError(f"seeds differ: {a} has {sorted(sa)}, {b} has {sorted(sb)}")
    return labels, [int(p) for p in periods], Y, runs, sorted(sa)


def compare(P_a, P_b, lab, th, kw, files):
    """rg.compare, with a mismatch reported in terms of the user's file names."""
    try:
        return rg.compare(P_a, P_b, lab, th, **kw)
    except ValueError as e:
        msg = str(e)
        for k, v in files.items():
            msg = msg.replace(k, v)
        raise ValueError(msg) from None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", nargs="?", default=HERE / "data", type=Path)
    ap.add_argument("--a", default="mine")
    ap.add_argument("--b", default="baseline")
    ap.add_argument("--splits", type=int, default=200)
    args = ap.parse_args(argv)
    labels, periods, Y, runs, seeds = load(args.data, args.a, args.b)
    keys = [str(p) for p in periods]
    lab = [Y[k][1] for k in keys]
    ids = [Y[k][0] for k in keys]
    ev_a, ev_b = {}, {}
    for s in seeds:
        (sa, th_a), (sb, th_b) = runs[(args.a, s)], runs[(args.b, s)]
        kw = dict(record_ids={"labels": ids, "ref": [sa[k][0] for k in keys], "final": [sb[k][0] for k in keys]},
                  label_ids=labels, seed=s, splits=args.splits, period_ids=periods, rule_name="carried")
        P_a, P_b = [sa[k][1] for k in keys], [sb[k][1] for k in keys]
        files = {"scores_ref": f"scores_{args.a}_seed{s}.csv", "scores_final": f"scores_{args.b}_seed{s}.csv",
                 "the labels": "labels.csv"}
        # each model at its own carried rule: a's evaluation under th_a, b's under th_b
        ev_a[s] = compare(P_a, P_b, lab, th_a, kw, files)["evaluations"]["ref"]
        ev_b[s] = compare(P_a, P_b, lab, th_b, kw, files)["evaluations"]["final"]
    same = all(np.array_equal(runs[(args.a, s)][1], runs[(args.b, s)][1]) for s in seeds)
    mc = rg.method_contrast(ev_a, ev_b, "carried", names=(args.a, args.b), same_rule=same)
    t = mc["terms"]
    print(
        f"{args.a} against {args.b}: {len(labels)} labels, periods {periods}, seeds {seeds}, {args.splits} partitions"
    )
    print(f"{'seed':>5s} {'G_U carried':>12s} {'G_V refitted':>13s} {'change':>9s}")
    for s in seeds:
        gu, gv = t["lead_at_carried_rule"]["per_seed"][s], t["lead_after_refit"]["per_seed"][s]
        print(f"{s:5d} {gu:+12.4f} {gv:+13.4f} {gv - gu:+9.4f}")
    for name, c in (("G_U carried", t["lead_at_carried_rule"]), ("G_V refitted", t["lead_after_refit"]),
                    ("change", t["change"])):
        print(f"{name:13s} {c['mean']:+.4f}  95% t interval [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
