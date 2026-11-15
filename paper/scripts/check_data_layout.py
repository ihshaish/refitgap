"""Report which corpora and embeddings are present before a rebuild, and what DATA.txt says for the rest.

Reads $ASRS_DIR/*.csv (default data/asrs, parsed as build_streams.py parses them), $EXTERNAL_DIR (default
data/external) and $EMB_DIR (default embeddings); writes nothing. The ASRS line counts records and years,
since a missed export slice shows up only later as an odd period boundary."""
import glob
import os
from pathlib import Path

from build_streams import read_records

EXT = Path(os.environ.get("EXTERNAL_DIR", "data/external"))
EMB = os.environ.get("EMB_DIR", "embeddings")
ok = True


def report(name, good, detail, hint):
    global ok
    mark = "ok " if good else "MISSING"
    print(f"{mark:8s} {name:10s} {detail}")
    if not good:
        print(f"         -> {hint}")
        ok = False


csvs = sorted(glob.glob(f'{os.environ.get("ASRS_DIR", "data/asrs")}/*.csv'))
if csvs:
    rows = 0
    years = set()
    for p in csvs:
        for acn, yyyymm, anomaly, narrative in read_records(p):
            rows += 1
            years.add(yyyymm // 100)
    span = f"{min(years)}-{max(years)}" if years else "no dates parsed"
    # the paper's export is ~226k records over 1988-2026
    report("ASRS", rows > 200_000, f"{len(csvs)} csvs, {rows:,} records, {span}",
           "DATA.txt; expect ~226k records across 1988-2026")
else:
    report("ASRS", False, "no csvs found", "DATA.txt, manual export")

report("HuffPost", (EXT / "huffpost.json").exists(),
       str(EXT / "huffpost.json"),
       "Kaggle link in DATA.txt; rename the download to huffpost.json")

arx = sorted((EXT / "arxiv").glob("*.parquet")) if (EXT / "arxiv").exists() else []
report("arXiv", len(arx) >= 12, f"{len(arx)} parquet files in {EXT/'arxiv'}",
       "python scripts/fetch_data.py arxiv")

report("Amazon", (EXT / "amazon" / "CDs_and_Vinyl.jsonl").exists(),
       str(EXT / "amazon" / "CDs_and_Vinyl.jsonl"),
       "python scripts/fetch_data.py amazon")

report("GloVe", Path(f"{EMB}/glove.6B.200d.txt").exists(),
       f"{EMB}/glove.6B.200d.txt",
       "python scripts/fetch_data.py glove")

print()
print("all set, see DATA.txt for the build order" if ok
      else "gaps above; nothing here is fatal until you rebuild that corpus")
