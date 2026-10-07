"""Step 2b - Extract the near-Colombo geocoding failures for the comparison test.

A failure only matters if the unit is inside Colombo district or within the
10 km buffer; a Divisional Hospital in Ratnapura that will not geocode costs
the study nothing.  This reads the provenance files written by
step2_geocode_amenities.py and pulls out exactly those units.

Output: data/amenities/_near_colombo_failures.json

Run:  python src/step2b_extract_near_colombo_failures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMEN = ROOT / "data" / "amenities"

CATEGORIES = ["govt_hospitals", "intlschools", "universities", "expressway_entrances"]
NEAR_MARKERS = ("anchor IN Colombo district", "INSIDE 10km buffer")


def main() -> int:
    out: dict[str, list] = {}
    total = 0
    for cat in CATEGORIES:
        path = AMEN / f"{cat}_PROVENANCE.txt"
        if not path.exists():
            print(f"  {cat}: no provenance file")
            continue
        txt = path.read_text(encoding="utf-8")
        if "UNITS THAT COULD NOT BE GEOCODED:" not in txt:
            out[cat] = []
            continue
        body = txt.split("UNITS THAT COULD NOT BE GEOCODED:")[1].split("EVERY UNIT")[0]

        units, cur = [], None
        for line in body.splitlines():
            m = re.match(r"^  - (.+?)\s+\((.*)\)\s*$", line)
            if m:
                cur = {"name": m.group(1).strip(), "locality": m.group(2).strip(),
                       "reason": ""}
                units.append(cur)
                continue
            m = re.match(r"^      reason: (.*)$", line)
            if m and cur is not None:
                cur["reason"] = m.group(1).strip()

        near = [u for u in units if any(k in u["reason"] for k in NEAR_MARKERS)]
        out[cat] = near
        total += len(near)
        print(f"  {cat:<22} {len(near):>3} near-Colombo failures "
              f"(of {len(units)} total failures)")

    (AMEN / "_near_colombo_failures.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\ntotal near-Colombo failures: {total}")
    print(f"wrote {AMEN / '_near_colombo_failures.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
