#!/usr/bin/env python3
"""Build the ASRS streams: primary 2010-2025, secondary 1999-2008, crossbreak 2004-2015.

Reads ASRS CSV exports from --data-dir (e.g. ../data_asrs); writes streams/stream_<name>.parquet,
periods_<name>.json and labelset_<name>.json. Backs streams and settings (Tables B.1, B.2).
Needs the ASRS export, which the repository does not ship."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from asrs_profile import YEAR_RE, map_top14, OTHER_UNSPECIFIED

DATA_SEED = 20260802
N_PERIODS = 8
TEST_FRAC = 0.20          # per-period held-out test fraction
MIN_POSITIVES = 75        # test positives per label in every period
SHINGLE = 5
MINHASH_PERMS = 64
LSH_BANDS = 16            # 16 bands x 4 rows: catches Jaccard >= ~0.7-0.8
JACCARD_THRESH = 0.8

STREAMS = {
    "primary": (2010, 2025),   # 2026 H1 left out: partial year
    "secondary": (1999, 2008),
    # Spans the 2009 break on purpose; abbreviations are expanded
    # (streams/asrs_abbrev_decode.json) so the orthography change is not the drift.
    "crossbreak": (2004, 2015),
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9/-]+")


def normalise(text):
    """Lower-case and keep word tokens."""
    return " ".join(TOKEN_RE.findall(text.lower()))


def read_records(path):
    """(acn, yyyymm, anomaly, narrative) for each dated record of one export."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            sections = next(reader)
            fields = next(reader)
        except StopIteration:
            return []
        header = [(s.strip(), t.strip()) for s, t in zip(sections, fields)]
        idx = {}
        for key, sec, fld in [("date", "Time", "Date"), ("anomaly", "Events", "Anomaly"),
                              ("narrative", "Report 1", "Narrative")]:
            hits = [i for i, (s, t) in enumerate(header) if s == sec and t == fld]
            idx[key] = hits[0] if hits else {"date": 1, "anomaly": 112, "narrative": 120}[key]
        out = []
        for row in reader:
            if len(row) <= idx["narrative"] or not row[0].strip().isdigit():
                continue
            date = row[idx["date"]].strip()
            if not re.fullmatch(r"(19|20)\d{4}", date):
                continue
            out.append((row[0].strip(), int(date), row[idx["anomaly"]],
                        row[idx["narrative"]].strip()))
        return out


def minhash_dedup(texts, rng):
    """Return indices to KEEP after near-dup collapse (earliest kept)."""
    perms = rng.integers(1, 2**61 - 1, size=(MINHASH_PERMS, 2), dtype=np.int64)
    prime = (1 << 61) - 1
    sigs = np.full((len(texts), MINHASH_PERMS), np.iinfo(np.int64).max, dtype=np.int64)
    shingle_sets = []
    for i, t in enumerate(texts):
        toks = t.split()
        sh = {hash(" ".join(toks[j:j + SHINGLE])) & 0x7FFFFFFFFFFFFFFF
              for j in range(max(1, len(toks) - SHINGLE + 1))}
        shingle_sets.append(sh)
        if not sh:
            continue
        vals = np.fromiter(sh, dtype=np.int64)
        for p in range(MINHASH_PERMS):
            sigs[i, p] = ((perms[p, 0] * vals + perms[p, 1]) % prime).min()
    rows_per_band = MINHASH_PERMS // LSH_BANDS
    drop = set()
    for b in range(LSH_BANDS):
        buckets = defaultdict(list)
        band = sigs[:, b * rows_per_band:(b + 1) * rows_per_band]
        for i in range(len(texts)):
            if i in drop:
                continue
            buckets[band[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            keep = members[0]
            for m in members[1:]:
                if m in drop:
                    continue
                a, c = shingle_sets[keep], shingle_sets[m]
                if a and c and len(a & c) / len(a | c) >= JACCARD_THRESH:
                    drop.add(m)
    return [i for i in range(len(texts)) if i not in drop], len(drop)


def main():
    global N_PERIODS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="streams")
    ap.add_argument("--stream", choices=list(STREAMS), required=True)
    ap.add_argument("--n-periods", type=int, default=N_PERIODS)
    ap.add_argument("--min-positives", type=int, default=MIN_POSITIVES,
                    help="label-filter sensitivity")
    ap.add_argument("--tag", default="", help="suffix for the output stream name")
    args = ap.parse_args()
    N_PERIODS = args.n_periods
    lo, hi = STREAMS[args.stream]
    decode = None
    if args.stream == "crossbreak":
        decode = json.load(open("streams/asrs_abbrev_decode.json"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DATA_SEED)

    rows = []
    for p in sorted(Path(args.data_dir).glob("*.csv")):
        m = YEAR_RE.search(p.name)
        if not (m and lo <= int(m.group(1)) <= hi):
            continue
        rows.extend(read_records(p))
    seen = set()
    records = []
    for acn, yyyymm, anomaly, narrative in rows:
        if acn in seen or not narrative or not (lo * 100 <= yyyymm <= hi * 100 + 12):
            continue
        seen.add(acn)
        labels = sorted({t for t in (map_top14(v) for v in anomaly.split(";") if v.strip())
                         if t and t not in ("UNMAPPED", OTHER_UNSPECIFIED)})
        if not labels:
            continue
        text = normalise(narrative)
        if decode is not None:
            text = " ".join(w for t in text.split() for w in decode.get(t, [t]))
        records.append((acn, yyyymm, text, labels))
    records.sort(key=lambda r: (r[1], r[0]))
    print(f"{args.stream}: {len(records)} labelled records {lo}-{hi}")

    keep_idx, n_drop = minhash_dedup([r[2] for r in records], rng)
    records = [records[i] for i in keep_idx]
    print(f"near-duplicate collapse: dropped {n_drop} ({n_drop / (len(records) + n_drop):.2%})")

    # equal-count contiguous periods over the deduped, time-sorted stream
    bounds = np.linspace(0, len(records), N_PERIODS + 1).astype(int)
    df = pd.DataFrame(records, columns=["acn", "yyyymm", "text", "labels"])
    df["period"] = 0
    for k in range(N_PERIODS):
        df.loc[bounds[k]:bounds[k + 1] - 1, "period"] = k
    spans = df.groupby("period")["yyyymm"].agg(["min", "max", "count"])

    # per-period held-out test split, one draw with the global seed
    df["split"] = "train_pool"
    for k in range(N_PERIODS):
        idx = df.index[df.period == k]
        test = rng.choice(idx, size=int(len(idx) * TEST_FRAC), replace=False)
        df.loc[test, "split"] = "test"

    MIN_POS = args.min_positives
    # label set: >= MIN_POS in EVERY period test set, uniformly
    all_labels = sorted({l for ls in df.labels for l in ls})
    label_ok = {}
    for lab in all_labels:
        per_period = [
            sum(lab in ls for ls in df[(df.period == k) & (df.split == "test")].labels)
            for k in range(N_PERIODS)
        ]
        label_ok[lab] = {"min_test_positives": int(min(per_period)),
                         "kept": bool(min(per_period) >= MIN_POS)}
    final = [l for l, v in label_ok.items() if v["kept"]]
    print(f"label set: {len(final)}/{len(all_labels)} pass the >={MIN_POS} rule: {final}")
    df["labels"] = df["labels"].map(lambda ls: ";".join(l for l in ls if l in final))
    df = df[df.labels.str.len() > 0].reset_index(drop=True)

    tag = args.stream if N_PERIODS == 8 else f"{args.stream}{N_PERIODS}"
    if args.tag:
        tag = f"{tag}{args.tag}"
    df.to_parquet(out_dir / f"stream_{tag}.parquet", index=False)
    (out_dir / f"periods_{tag}.json").write_text(json.dumps({
        "stream": args.stream, "window": [lo, hi], "n_periods": N_PERIODS,
        "data_seed": DATA_SEED, "test_frac": TEST_FRAC,
        "near_dups_dropped": n_drop,
        "spans": {int(k): {"from": int(v["min"]), "to": int(v["max"]),
                           "n": int(v["count"])} for k, v in spans.iterrows()},
    }, indent=2))
    (out_dir / f"labelset_{tag}.json").write_text(json.dumps(
        {"min_positives": MIN_POS, "labels": label_ok}, indent=2))
    print(f"wrote {out_dir}/stream_{tag}.parquet "
          f"({len(df)} records, {df.split.eq('test').sum()} test)")
    print(spans)


if __name__ == "__main__":
    main()
