#!/usr/bin/env python3
"""Build the HuffPost stream (2012-2017) under the ASRS construction rules.

Reads --raw (default data/external/huffpost.json); writes streams/stream_huffpost.parquet and
streams/periods_huffpost.json. Builds the HuffPost stream of streams and settings (Tables B.1, B.2).
Needs the News Category Dataset v3 from Kaggle, which the repository does not ship."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from build_streams import DATA_SEED, N_PERIODS, TEST_FRAC, MIN_POSITIVES, minhash_dedup

# Categories present under one name for the whole window; renamed ones (HEALTHY
# LIVING to WELLNESS, TASTE to FOOD & DRINK) would show a schema change as forgetting.
STABLE = ["BLACK VOICES", "BUSINESS", "COMEDY", "ENTERTAINMENT", "QUEER VOICES"]
WINDOW = (201201, 201712)

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")


def normalise(text):
    return " ".join(TOKEN_RE.findall(text.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/external/huffpost.json",
                    help="News Category Dataset v3, one JSON object per line")
    ap.add_argument("--out-dir", default="streams")
    args = ap.parse_args()

    rows = []
    with open(args.raw) as f:
        for line in f:
            r = json.loads(line)
            date = r.get("date", "")
            if len(date) < 7:
                continue
            ym = int(date[:4]) * 100 + int(date[5:7])
            if not (WINDOW[0] <= ym <= WINDOW[1]):
                continue
            cat = r.get("category", "")
            if cat not in STABLE:
                continue
            # headline, then the standfirst
            text = normalise((r.get("headline") or "") + " " +
                             (r.get("short_description") or ""))
            if not text:
                continue
            rows.append((ym, text, cat))

    df = pd.DataFrame(rows, columns=["yyyymm", "text", "labels"])
    print(f"in window, stable categories: {len(df)}")
    df = df.drop_duplicates("text").reset_index(drop=True)
    print(f"after exact-duplicate removal: {len(df)}")

    rng = np.random.default_rng(DATA_SEED)
    keep_idx, n_drop = minhash_dedup(df.text.tolist(), rng)
    df = df.iloc[keep_idx].reset_index(drop=True)
    print(f"near-duplicate collapse: dropped {n_drop} "
          f"({n_drop / (len(df) + n_drop):.2%}) -> {len(df)}")

    df["acn"] = np.arange(len(df)).astype(str)
    df = df.sort_values(["yyyymm", "acn"]).reset_index(drop=True)
    bounds = np.linspace(0, len(df), N_PERIODS + 1).astype(int)
    df["period"] = 0
    for k in range(N_PERIODS):
        df.loc[bounds[k]:bounds[k + 1] - 1, "period"] = k

    df["split"] = "train_pool"
    for k in range(N_PERIODS):
        idx = df.index[df.period == k]
        df.loc[rng.choice(idx, int(len(idx) * TEST_FRAC), replace=False), "split"] = "test"

    # The label set is fixed by schema stability, not by the positives rule: three
    # of the five categories fall below 75 in their thinnest period, and applying
    # both filters would leave two labels. Thin labels make the task harder, not
    # easier; the counts are printed and stored.
    label_ok = {}
    for lab in STABLE:
        per_period = [int(((df.period == k) & (df.split == "test") &
                           (df.labels == lab)).sum()) for k in range(N_PERIODS)]
        label_ok[lab] = {"min_test_positives": min(per_period),
                         "below_aviation_floor": bool(min(per_period) < MIN_POSITIVES)}
    print(f"min test positives per period (aviation floor is {MIN_POSITIVES}):")
    for lab, v in sorted(label_ok.items(), key=lambda kv: kv[1]["min_test_positives"]):
        flag = "  <- below the aviation floor" if v["below_aviation_floor"] else ""
        print(f"   {v['min_test_positives']:4d}  {lab}{flag}")

    out = f"{args.out_dir}/stream_huffpost.parquet"
    df[["acn", "yyyymm", "text", "labels", "period", "split"]].to_parquet(out, index=False)
    json.dump({"stream": "huffpost", "window": list(WINDOW), "n_periods": N_PERIODS,
               "data_seed": DATA_SEED, "test_frac": TEST_FRAC,
               "near_dups_dropped": n_drop, "labels": label_ok},
              open(f"{args.out_dir}/periods_huffpost.json", "w"), indent=2)
    print(f"wrote {out} ({len(df)} records, {df.split.eq('test').sum()} test)")
    print(df.groupby("period").yyyymm.agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()
