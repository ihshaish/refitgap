#!/usr/bin/env python3
"""The stronger prior-reweighting stream and its control with 200 split-half draws, as every row of Table E.1.
Writes results/calib_splits200_priors.parquet. Backs Table E.1.
Needs the stream files and the trained checkpoints (checkpoints/, not shipped)."""
import pandas as pd

import constructed_drift_levels as C

OUT = "results/calib_splits200_priors.parquet"
STREAMS = ["asrsprior00", "asrsprior100"]


def main():
    C.SPLITS = 200
    print("SPLITS=%d, DRAWS=%d, seeds %s" % (C.SPLITS, C.DRAWS, list(C.SEEDS)), flush=True)
    rows = []
    for s in STREAMS:
        print("%s:" % s, flush=True)
        C.run_stream(s, rows)
    t = pd.DataFrame(rows)
    if t.empty:
        print("no rows produced")
        return
    t.to_parquet(OUT, index=False)
    print("\nwrote %s" % OUT)
    print(t.groupby("stream")[["f1_frozen", "f1_oracle_insample",
                               "f1_oracle_splithalf"]].mean().round(4).to_string())
    print("\nseeds per stream:")
    print(t.groupby("stream").seed.nunique().to_string())


if __name__ == "__main__":
    main()
