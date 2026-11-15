"""Write the tiny synthetic example of the public input format into data/.

    python examples/your_models/make_data.py

Two methods ("mine", "baseline"), four seeds, three periods, four labels, 150 records a period.
Thresholds are fitted once on a first-period validation draw (19-point grid), then carried.
Each score file shuffles its rows, as an outside pipeline might. Deterministic.
"""
from pathlib import Path

import numpy as np

from refitgap import fit_validation_rule

OUT = Path(__file__).resolve().parent / "data"
LABELS = ["engine", "fuel", "weather", "crew"]
PERIODS, SEEDS, N = [1, 2, 3], [0, 1, 2, 3], 150


def write(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    ids = {j: [f"p{j}-{i:03d}" for i in range(N)] for j in PERIODS}
    Y = {j: (rng.random((N, len(LABELS))) < [0.15, 0.25, 0.35, 0.2]).astype(int) for j in PERIODS}
    write(OUT / "labels.csv", ["period", "record_id", *LABELS],
          [(j, ids[j][i], *Y[j][i]) for j in PERIODS for i in range(N)])
    thresholds = []
    for method, sep, drift in (("mine", 2.4, 0.15), ("baseline", 2.0, 0.45)):
        for s in SEEDS:
            r = np.random.default_rng(100 * s + (method == "mine"))
            score = lambda Yj, lag: np.round(1 / (1 + np.exp(-(sep * Yj + r.normal(0, 1, Yj.shape) - 1.5
                                                                + drift * lag))), 4)
            rows = []
            for j in PERIODS:
                p = score(Y[j], 3 - j)
                rows += [(j, ids[j][i], *p[i]) for i in r.permutation(N)]
            write(OUT / f"scores_{method}_seed{s}.csv", ["period", "record_id", *LABELS], rows)
            Yval = (r.random((300, len(LABELS))) < [0.15, 0.25, 0.35, 0.2]).astype(int)
            thresholds.append((method, s, *fit_validation_rule(score(Yval, 3), Yval).round(2)))
    write(OUT / "thresholds.csv", ["method", "seed", *LABELS], thresholds)


if __name__ == "__main__":
    main()
