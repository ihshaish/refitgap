"""Table 1 of the paper: two final scorers that no rank summary can tell apart.

    python examples/two_scorer/table1.py

Input: nothing; the fourteen records, one label and three scorers are built in
refitgap.identification_example(), in exact fractions.
Output: the table, and exit status 0 when the paper's values hold: the same AUROC (6/11) and
average precision (5/18) for all three scorers, a deficit of 1/26 for both finals, all of it
change in mismatch for final A and all of it attainable gap for final B.
"""
import sys
from fractions import Fraction as F

import refitgap as rg


def main():
    ex = rg.identification_example()
    print(f"{ex['n_records']} records, one label, carried rule {ex['threshold']}")
    print(f"{'scorer':10s} {'AUROC':>6s} {'AP':>6s} {'F1 carried':>11s} {'F1 refitted':>12s}")
    for name, s in ex["scorers"].items():
        print(f"{name:10s} {str(s['auroc']):>6s} {str(s['average_precision']):>6s} {str(s['carried_f1']):>11s} "
              f"{str(s['best_f1']):>12s}")
    print(f"\n{'against reference':18s} {'deficit':>8s} {'attainable gap':>15s} {'change in mismatch':>19s}")
    for name, d in ex["decomposition"].items():
        print(f"{name:18s} {str(d['deficit']):>8s} {str(d['attainable_gap']):>15s} {str(d['mismatch_change']):>19s}")

    a, b = ex["decomposition"]["final A"], ex["decomposition"]["final B"]
    ok = (all(s["auroc"] == F(6, 11) and s["average_precision"] == F(5, 18) for s in ex["scorers"].values())
          and a == dict(deficit=F(1, 26), attainable_gap=F(0), mismatch_change=F(1, 26))
          and b == dict(deficit=F(1, 26), attainable_gap=F(1, 26), mismatch_change=F(0)))
    print("\nsame AUROC and AP, same deficit, opposite components: " + ("as in the paper" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
