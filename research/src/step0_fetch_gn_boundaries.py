"""Step 0 - Fetch the authoritative GN division boundary source.

Downloads the OCHA/HDX Common Operational Dataset for Sri Lankan administrative
boundaries (dataset id `cod-ab-lka`).  ADM4 in that dataset is the Grama
Niladhari (GN) division layer, ADM3 the Divisional Secretariat (DS) division
layer and ADM2 the district layer.

Nothing is filtered or renamed here; this script only pulls the raw archive so
that later steps have a fixed, reproducible input.

Run:  python src/step0_fetch_gn_boundaries.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HDX_PACKAGE = "cod-ab-lka"
HDX_API = "https://data.humdata.org/api/3/action/package_show"

# Formats we want out of the package, in order of preference.
WANTED = ("GeoJSON", "SHP")


def list_resources() -> list[dict]:
    resp = requests.get(HDX_API, params={"id": HDX_PACKAGE}, timeout=120)
    resp.raise_for_status()
    pkg = resp.json()["result"]
    print(f"HDX package : {pkg['title']}")
    print(f"last modified: {pkg.get('last_modified')}")
    out = []
    for res in pkg["resources"]:
        print(f"  {res['format']:>10s} | {str(res.get('size')):>10s} | {res['name']}")
        out.append(res)
    return out


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print(f"  saved: {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    resources = list_resources()

    picked = None
    for fmt in WANTED:
        for res in resources:
            if res["format"] == fmt and res["name"].endswith(".zip"):
                picked = res
                break
        if picked:
            break

    if picked is None:
        print("ERROR: no GeoJSON/SHP zip resource found in the HDX package.")
        return 1

    archive = download(picked["url"], RAW / picked["name"])

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        print(f"\narchive contains {len(names)} entries:")
        for n in names:
            print("   ", n)
        target = RAW / archive.name.replace(".zip", "")
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
        print(f"\nextracted to {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
