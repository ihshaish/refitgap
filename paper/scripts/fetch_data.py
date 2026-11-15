"""Download arXiv, Amazon CDs and Vinyl and GloVe, skipping files already present and resuming partial downloads.

Writes $EXTERNAL_DIR/{arxiv,amazon}/ (default data/external) and $EMB_DIR/glove.6B.* (default embeddings).
ASRS (NASA web form) and HuffPost (Kaggle licence) are manual; see DATA.txt, then run check_data_layout.py.
Usage: python scripts/fetch_data.py [arxiv] [amazon] [glove]; needs network access and curl."""
import os
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

EXT = Path(os.environ.get("EXTERNAL_DIR", "data/external"))
EMB = Path(os.environ.get("EMB_DIR", "embeddings"))

ARXIV_API = ("https://huggingface.co/api/datasets/"
             "bluuebunny/arxiv_metadata_by_year")
ARXIV_FILE = ("https://huggingface.co/datasets/"
              "bluuebunny/arxiv_metadata_by_year/resolve/main/{}")
AMAZON = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
          "resolve/main/raw/review_categories/CDs_and_Vinyl.jsonl")
GLOVE = "https://nlp.stanford.edu/data/glove.6B.zip"


def curl(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest}, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url} -> {dest}")
    # -C - resumes a partial download, which matters for the 800 MB GloVe zip
    r = subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest), url])
    if r.returncode != 0:
        sys.exit(f"curl failed on {url}, rerun to resume")


def arxiv():
    print("arXiv metadata (parquet per year, 2012-2023 is what the paper uses)")
    with urllib.request.urlopen(ARXIV_API) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]
                 if s["rfilename"].endswith(".parquet")]
    for f in files:
        year = "".join(c for c in f if c.isdigit())[:4]
        if year and 2012 <= int(year) <= 2023:
            curl(ARXIV_FILE.format(f), EXT / "arxiv" / Path(f).name)


def amazon():
    print("Amazon CDs and Vinyl reviews (one jsonl, about 1.3 GB)")
    curl(AMAZON, EXT / "amazon" / "CDs_and_Vinyl.jsonl")


def glove():
    print("GloVe 6B (zip is about 800 MB; the paper uses the 200d file)")
    curl(GLOVE, EMB / "glove.6B.zip")
    target = EMB / "glove.6B.200d.txt"
    if not target.exists():
        print(f"  unzipping into {EMB}")
        subprocess.run(["unzip", "-o", str(EMB / "glove.6B.zip"), "-d", str(EMB)])


if __name__ == "__main__":
    which = sys.argv[1:] or ["arxiv", "amazon", "glove"]
    for name in which:
        {"arxiv": arxiv, "amazon": amazon, "glove": glove}[name]()
    print("\nDone. ASRS and HuffPost are manual, see DATA.txt. Then run:")
    print("  python scripts/check_data_layout.py")
