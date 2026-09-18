"""
Download the Transfermarkt tables the VALUE model needs.

prepare_tm_data.py expects a directory of plain CSVs - players, appearances,
player_valuations - which used to be produced by hand-downloading a Kaggle
archive. That is why the value model was stuck on three leagues: extending it
meant someone re-downloading and re-extracting a zip.

Same CC0 source as src/fetch_lineups.py, fetched directly and decompressed so
the step can run unattended in CI.

appearances.csv is ~189MB expanded. prepare_tm_data reads it with usecols, so
memory stays modest, but the disk space is real.

Run:  python src/fetch_tm_archive.py
Out:  data/external/archive/{players,appearances,player_valuations}.csv
"""

import gzip
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import requests

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE = os.path.join(BASE, "data", "external", "archive")
R2 = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

TABLES = ["players", "appearances", "player_valuations"]


def fetch(name: str) -> str:
    os.makedirs(ARCHIVE, exist_ok=True)
    csv_path = os.path.join(ARCHIVE, f"{name}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10_000:
        print(f"  {name:<20} cached ({os.path.getsize(csv_path) / 1e6:.1f} MB)")
        return csv_path

    gz_path = csv_path + ".gz"
    print(f"  {name:<20} downloading...", end=" ", flush=True)
    with requests.get(f"{R2}/{name}.csv.gz", stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)

    with gzip.open(gz_path, "rb") as src, open(csv_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    os.remove(gz_path)

    print(f"{os.path.getsize(csv_path) / 1e6:.1f} MB")
    return csv_path


def main():
    print(f"Fetching Transfermarkt archive -> {ARCHIVE}")
    for t in TABLES:
        fetch(t)
    print("\nNext: python src/prepare_tm_data.py "
          f"--archive-dir \"{ARCHIVE}\"")


if __name__ == "__main__":
    main()
