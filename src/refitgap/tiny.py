"""The tiny example: eight records, two labels, two periods, two models, one partition.
Every number it prints is worked by hand in API.txt.

    refitgap example"""
import numpy as np

from .analysis import analyse, make_run

SEED, SPLITS = 0, 1
THETA = np.array([0.5, 0.5])
LABEL_NAMES = ["label a", "label b"]
# labels, columns (label a, label b), rows r0..r3 of each period
Y0 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
Y1 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
# final model
F0 = np.array([[0.4, 0.6], [0.4, 0.2], [0.1, 0.8], [0.1, 0.3]])
F1 = np.array([[0.9, 0.3], [0.2, 0.8], [0.9, 0.3], [0.2, 0.8]])
# reference model of period 0 (the reference of period 1 is the final model, F1)
S0 = np.array([[0.9, 0.3], [0.9, 0.3], [0.2, 0.8], [0.2, 0.8]])

# the values worked by hand in API.txt (fractions)
HAND = {
    "final": {0: dict(V_sep=3 / 4, V_shared=3 / 4, U_half=1 / 4, U_full=1 / 4, C=0.0, M=1 / 2, R_half=1 / 2),
              1: dict(V_sep=1.0, V_shared=11 / 12, U_half=1.0, U_full=1.0, C=1 / 12, M=-1 / 12, R_half=0.0),
              "mean": dict(V_sep=7 / 8, V_shared=5 / 6, U_half=5 / 8, U_full=5 / 8, C=1 / 24, M=5 / 24, R_half=1 / 4)},
    "reference": {0: dict(V_sep=1.0, U_half=1.0, U_full=1.0), 1: dict(V_sep=1.0, U_half=1.0, U_full=1.0),
                  "mean": dict(V_sep=1.0, U_half=1.0, U_full=1.0)},
    "decomposition": {0: dict(B=3 / 4, A=1 / 4, dR=1 / 2), 1: dict(B=0.0, A=0.0, dR=0.0),
                      "mean": dict(B=3 / 8, A=1 / 8, dR=1 / 4)},
}


def tiny_run():
    return make_run([F0, F1], [Y0, Y1], SEED, {"inherited": THETA}, reference=[S0, F1], period_ids=[0, 1],
                    source="tiny example", label_names=LABEL_NAMES)


def tiny_result(u_convention="full"):
    return analyse(
        [tiny_run()], rule="inherited", splits=SPLITS, u_convention=u_convention, names=("tiny example", "b")
    )


def lines(result):
    """The printed lines, one per quantity, 4 decimals (API.txt gives the exact fractions)."""
    out = [f"tiny example: 2 periods x 4 records, 2 labels, seed {SEED}, {SPLITS} partition, inherited thresholds 0.5",
           "levels (macro-F1):"]
    for r in result["levels"]:
        p = "mean" if str(r["period"]).startswith("mean") else r["period"]
        vals = [("V_sep", r["V_sep"]), ("V_shared", r["V_shared"]), ("U_half", r["U_half"]), ("U_full", r["U_full"]),
                ("C", r["C"]), ("M", r["M"]), ("R_half", r["R_half"])]
        out.append(f"  {r['model']} period {p}: " + "  ".join(f"{k} {v:.4f}" for k, v in vals if v is not None))
    out.append("decomposition B = A + dR against the per-period reference, U on the whole period:")
    for r in result["decomposition"]:
        p = "mean" if str(r["period"]).startswith("mean") else r["period"]
        out.append(f"  period {p}: B {r['B']:.4f}  A {r['A']:.4f}  dR {r['dR']:.4f}")
    out.extend(result["interpretation"])
    return out


def main(argv=None):
    for line in lines(tiny_result()):
        print(line)


if __name__ == "__main__":
    main()
