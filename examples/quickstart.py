"""Two methods, four seeds, three periods: the lead at the carried rule and after both refits.

    python examples/quickstart.py

Reads the synthetic CSVs in examples/your_models/data/ (made by make_data.py).
"""
import csv
from pathlib import Path

import numpy as np
import refitgap as rg

DATA = Path(__file__).resolve().parent / "your_models" / "data"


def read(name):
    """CSV -> {first column: (second column, numeric rest as an array)}."""
    out = {}
    for key, rid, *vals in list(csv.reader(open(DATA / name)))[1:]:
        out.setdefault(key, []).append((rid, [float(v) for v in vals]))
    return {k: ([r for r, _ in v], np.array([x for _, x in v])) for k, v in out.items()}


labels, carried = read("labels.csv"), read("thresholds.csv")
periods = sorted(labels, key=int)
evals = {}
for m in ("mine", "baseline"):
    for s, rule in zip(map(int, carried[m][0]), carried[m][1]):
        scores = read(f"scores_{m}_seed{s}.csv")
        Ps = [rg.align_to(labels[j][0], *scores[j]) for j in periods]          # rows in the labels' order
        Ys = [labels[j][1].astype(int) for j in periods]
        evals.setdefault(m, {})[s] = rg.evaluate(Ps, Ys, s, [int(j) for j in periods], rules={"carried": rule})

mc = rg.method_contrast(evals["mine"], evals["baseline"], "carried", names=("mine", "baseline"), same_rule=False)
print(f"mine minus baseline, macro-F1, {len(mc['seeds'])} seeds, 95% t intervals")
for label, t in (("at the carried rule", "lead_at_carried_rule"), ("after both refits", "lead_after_refit"),
                 ("change", "change")):
    c = mc["terms"][t]
    print(f"  {label:20s} {c['mean']:+.4f}  [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
