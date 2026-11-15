#!/usr/bin/env python3
"""Build the MultiEURLEX stream (English, EUROVOC level 1, 1988-2014) under the ASRS construction rules.

Reads data_eurlex/en_all_level1.parquet; writes streams/stream_eurlex.parquet and
data_eurlex/stream_eurlex_build.json, used in streams and settings (Tables B.1, B.2). Records are
ordered by CELEX year, then sequence number. Needs the MultiEURLEX corpus, not shipped."""
import json
import re

import numpy as np
import pandas as pd

from build_streams import DATA_SEED, N_PERIODS, TEST_FRAC, minhash_dedup

LO, HI, MIN_POS = 1988, 2014, 75
LEVEL1 = ["100149", "100160", "100148", "100147", "100152", "100143", "100156", "100158", "100154", "100153",
          "100142", "100145", "100150", "100162", "100159", "100144", "100151", "100157", "100161", "100146", "100155"]

d = pd.read_parquet("data_eurlex/en_all_level1.parquet")
d = d[~d.text.duplicated()]
d = d[(d.year >= LO) & (d.year <= HI)].copy()
d["seq"] = d.celex_id.str.extract(r"^\d\d{4}[A-Z](\d+)")[0].astype(float).fillna(0)
d = d.sort_values(["year", "seq"]).reset_index(drop=True)
d["text"] = d.text.map(lambda t: re.sub(r"\s+", " ", t).strip())
rng = np.random.default_rng(DATA_SEED)
keep_idx, n_drop = minhash_dedup(d.text.tolist(), rng)
d = d.iloc[sorted(keep_idx)].reset_index(drop=True)
print(f"{len(d)} records after removing {n_drop} near-duplicates")

df = pd.DataFrame(dict(acn=d.celex_id, yyyymm=(d.year.astype(int) * 100 + 1),
                       text=d.text, labels=d.labels.map(lambda a: ";".join(LEVEL1[int(x)] for x in a))))
n = len(df)
bounds = np.linspace(0, n, N_PERIODS + 1).astype(int)
df["period"] = 0
for k in range(N_PERIODS):
    df.loc[bounds[k]:bounds[k + 1] - 1, "period"] = k
df["split"] = "train_pool"
for k in range(N_PERIODS):
    idx = df.index[df.period == k]
    df.loc[rng.choice(idx, int(len(idx) * TEST_FRAC), replace=False), "split"] = "test"
spans = df.groupby("period").yyyymm.agg(["min", "max", "count"])
print(spans.to_string())

label_ok = {}
for lab in LEVEL1:
    per_period = [
        sum(lab in ls.split(";") for ls in df[(df.period == k) & (df.split == "test")].labels) for k in range(N_PERIODS)
    ]
    label_ok[lab] = dict(min_test_positives=int(min(per_period)), kept=bool(min(per_period) >= MIN_POS))
keep = [l for l, v in label_ok.items() if v["kept"]]
df["labels"] = df.labels.map(lambda s: ";".join(l for l in s.split(";") if l in keep))
df = df[df.labels != ""].reset_index(drop=True)
print(f"{len(keep)} labels kept: {keep}; {len(df)} records with at least one kept label")
df.to_parquet("streams/stream_eurlex.parquet", index=False)
json.dump(dict(range=[LO, HI], min_pos=MIN_POS, n_periods=N_PERIODS, test_frac=TEST_FRAC, data_seed=DATA_SEED,
               near_duplicates_removed=int(n_drop), records=int(len(df)), labels_kept=keep, label_support=label_ok,
               period_spans=[[int(a), int(b), int(c)] for a, b, c in spans.to_numpy()],
               labels_per_record=float(df.labels.map(lambda s: s.count(";") + 1).mean())),
          open("data_eurlex/stream_eurlex_build.json", "w"), indent=1)
