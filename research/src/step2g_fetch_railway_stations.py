"""Step 2g - Fetch the Sri Lanka Railways station list.

Source: Department of Sri Lanka Railways, "Station Details" -
railway.gov.lk/web/index.php?option=com_content&view=article&id=165&Itemid=191
This is the page the original study cites.

The page lists stations by LINE, each with its distance in km from Colombo
Fort.  That distance is kept: it is an independent check on every geocoded
coordinate, since a station listed 15.5 km from Fort should not geocode 60 km
away.  Colombo district reaches roughly 40 km from Fort, so the distance also
bounds which stations can possibly fall inside the district or its 10 km
buffer.

Output: data/reference/railway_stations_source.json

Run:  python src/step2g_fetch_railway_stations.py
"""

from __future__ import annotations

import json
import re
import sys
from html import unescape
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"

URL = ("https://www.railway.gov.lk/web/index.php?option=com_content&view=article"
       "&id=165&Itemid=191&lang=en")
UA = "colombo-land-price-replication/1.0 (MSc dissertation replication)"

# Distance from Fort beyond which a station cannot be in Colombo district or
# its 10 km buffer.  Generous: the district's farthest point is ~40 km.
MAX_KM_FROM_FORT = 60.0


def strip_html(s: str) -> str:
    return " ".join(unescape(re.sub(r"(?s)<[^>]+>", " ", s)).split())


def main() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    r = session.get(URL, timeout=90)
    r.raise_for_status()
    html = r.text

    # Lines are introduced by a heading like "Station Details in Main Line".
    parts = re.split(r"(Station\s+Details\s+in\s+[^<]{0,60})", html, flags=re.I)
    lines: dict[str, list] = {}
    current = None
    for chunk in parts:
        m = re.match(r"Station\s+Details\s+in\s+(.+)", chunk.strip(), re.I)
        if m:
            current = " ".join(m.group(1).split()).strip(" -:")
            lines.setdefault(current, [])
            continue
        if current is None:
            continue
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", chunk, re.I | re.S):
            cells = [strip_html(c) for c in
                     re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.I | re.S)]
            cells = [c for c in cells if c]
            if len(cells) < 2:
                continue
            name, dist = cells[0], cells[1]
            if name.lower().startswith("station") or not re.match(r"^[\d.]+$", dist):
                continue
            lines[current].append({"name": name, "km_from_fort": float(dist)})

    lines = {k: v for k, v in lines.items() if v}
    total = sum(len(v) for v in lines.values())

    near = {k: [s for s in v if s["km_from_fort"] <= MAX_KM_FROM_FORT]
            for k, v in lines.items()}
    near = {k: v for k, v in near.items() if v}
    n_near = sum(len(v) for v in near.values())

    out = {
        "source": URL,
        "note": ("Sri Lanka Railways 'Station Details'. Distance from Colombo "
                 "Fort is retained from the source as an independent check on "
                 "each geocoded coordinate."),
        "max_km_from_fort_kept": MAX_KM_FROM_FORT,
        "counts": {"all_lines": {k: len(v) for k, v in lines.items()},
                   "total": total,
                   "within_max_km": n_near},
        "stations_all": lines,
        "stations_within_max_km": near,
    }
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "railway_stations_source.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"lines found: {len(lines)}")
    for k, v in lines.items():
        print(f"  {k:<34} {len(v):>4} stations  ({len(near.get(k, []))} within "
              f"{MAX_KM_FROM_FORT:.0f}km of Fort)")
    print(f"\ntotal stations: {total}   within {MAX_KM_FROM_FORT:.0f}km: {n_near}")
    print(f"wrote {REF / 'railway_stations_source.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
