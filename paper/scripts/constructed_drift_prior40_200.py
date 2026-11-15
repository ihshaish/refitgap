#!/usr/bin/env python3
"""The weaker prior-reweighting stream (asrsprior40) with 200 split-half draws, as every row of Table E.1.
Writes results/calib_splits200_prior40.parquet. Backs Table E.1.
Needs the stream files and the trained checkpoints (checkpoints/, not shipped)."""
import pandas as pd

import constructed_drift_levels as C

OUT = "results/calib_splits200_prior40.parquet"


def main():
    C.SPLITS = 200
    print("SPLITS=%d, DRAWS=%d, seeds %s" % (C.SPLITS, C.DRAWS, list(C.SEEDS)), flush=True)
    rows = []
    C.run_stream("asrsprior40", rows)
    t = pd.DataFrame(rows)
    t.to_parquet(OUT, index=False)
    print("wrote %s" % OUT)
    print(t.groupby("stream")[["f1_frozen", "f1_oracle_splithalf"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
