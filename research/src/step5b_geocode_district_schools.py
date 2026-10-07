"""Step 5b - geocode the schools of the other three zones (Step 5a's output).

Same machinery, same guards as step2n: Google address-first, class/quality gate,
acceptance radius against the address anchor, duplicate-pin rejection, Sri Lanka
bounding box, point-in-polygon test for `in_colombo_district`.

Only the 252 NEW units are geocoded. Nothing already in data/amenities/ is
touched: this writes data/amenities/govtschools_extra.csv and the existing
cache is reused, so previously resolved units cost nothing.

Run:  python src/step5b_geocode_district_schools.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import step2_geocode_amenities as G  # noqa: E402
import step2n_run_google_first as GF  # noqa: E402

REF = ROOT / "data" / "reference"
RESULTS = ROOT / "results"
CAT = "govtschools_extra"

G.GOOGLE_RESULT_TYPES_OK |= {"school", "primary_school", "secondary_school"}
G.ACCEPT_KM[CAT] = 2.0


def main() -> int:
    if not os.environ.get("GOOGLE_MAPS_API_KEY", "").strip():
        print("GOOGLE_MAPS_API_KEY is not set.")
        return 1

    doc = json.loads((REF / "schools_district_source.json").read_text(encoding="utf-8"))

    # "Wp/Ho/" is the school census code prefix (Western Province / Homagama
    # zone), not part of the school's name, and Google resolves nothing with it
    # attached. Strip it. A school the source itself marks as closed is dropped.
    units, closed = [], []
    for r in doc["rows"]:
        nm = re.sub(r"^\s*[Ww][Pp]\s*/\s*[A-Za-z]{1,4}\s*/\s*", "", r["name"]).strip()
        if re.search(r"\bclosed\b", nm, re.I):
            closed.append(nm)
            continue
        nm = re.sub(r"\s*-\s*Has been.*$", "", nm, flags=re.I).strip()
        units.append((nm, f"{r['address']}, {r['ds_division']}"))
    if closed:
        print(f"dropped {len(closed)} school(s) the source marks closed: {closed}")

    G.SOURCES[CAT] = (
        doc["source"],
        "Schools of the Piliyandala, Sri Jayawardenapura and Homagama education "
        "zones - the three Colombo district zones absent from "
        "schools_source.json, which covers the Colombo zone only. "
        + doc["classification_limitation"] + ". Excluded: " + doc["excluded"] + ".")

    polys = G.build_polygon_cache()
    cache = G.load_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": G.UA})
    print(f"polygons {len(polys)} | cache {len(cache)} | units {len(units)}")

    G.resolve = GF.make_resolver({})
    s = G.write_category(CAT, units, polys, cache, session)

    dups, seen = [], {}
    for r in s["rows"]:
        for (a, b), owner in seen.items():
            if G.haversine(a, b, r["latitude"], r["longitude"]) * 1000 <= G.DUPLICATE_M:
                dups.append(f"{r['name']} shares a pin with {owner}")
        seen[(r["latitude"], r["longitude"])] = r["name"]

    n, u = len(s["rows"]), len(units)
    txt = (f"{CAT}\n"
           f"  rows written / units / coverage : {n} / {u} / {n / u * 100:.1f}%\n"
           f"  duplicate pins : {len(dups)}\n"
           f"  in Colombo district : {s['in']}\n"
           f"  coord sources : {s['split']}\n")
    (RESULTS / "step5b_report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
