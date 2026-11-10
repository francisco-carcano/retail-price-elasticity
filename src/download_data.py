"""Downloads the Dominick's canned tuna files from the Kilts Center into data/raw.

The data are for academic research only and may not be redistributed, so they are
fetched from the source instead of being committed. Usage: python src/download_data.py
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
KILTS = "https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset"
FILES = {
    "wtna.zip": f"{KILTS}/movement_csv-files/wtna.zip",  # weekly sales per store and product
    "upctna.csv": f"{KILTS}/upc_csv-files/upctna.csv",  # product catalogue
    "demo_stata.zip": "https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count/demo_stata.zip",
}


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(target, "wb") as out:
        out.write(response.read())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = RAW_DIR / name
        if not target.exists():
            print(f"Downloading {name}")
            download(url, target)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as archive:
                archive.extractall(RAW_DIR / target.stem)
    print(f"Data ready in {RAW_DIR}")


if __name__ == "__main__":
    main()
