#!/usr/bin/env python3
"""The unmodified stream (control for the vocabulary streams) under the split-half estimator of
constructed_drift_levels.py, written to its own file rather than through that module's main().
Writes results/calib_control_primary.parquet. Backs Table E.1.
Needs the stream files and the trained checkpoints (checkpoints/, not shipped)."""
import pandas as pd

from constructed_drift_levels import run_stream, SEEDS

OUT = "results/calib_control_primary.parquet"


def main():
    rows = []
    print("primary (control for the vocabulary streams), seeds %s" % list(SEEDS), flush=True)
    run_stream("primary", rows)
    t = pd.DataFrame(rows)
    if t.empty:
        print("no rows: check checkpoints/primary_C2_s*_v2 exist")
        return
    print("\nseeds found:", sorted(t.seed.unique()))
    t.to_parquet(OUT, index=False)
    print("wrote %s" % OUT)
    print(t[["f1_frozen", "f1_oracle_insample", "f1_oracle_splithalf"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
