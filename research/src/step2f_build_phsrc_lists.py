"""Step 2f - Build pvt_hospitals and pvt_med_centers source lists from PHSRC.

ADDRESS DE-DUPLICATION.  The register is a list of REGISTRATIONS, not of
buildings.  Durdans appears as PHSRC/PH/08 and PHSRC/PH/308, both at
3 Alfred Place, Colombo 3.  The variables this feeds are "distance to the
nearest private hospital" and "count of private hospitals within 2 km": two
registrations in one building are ONE hospital for both.  Two rows at one
coordinate would inflate the count and change nothing in the distance.

Same principle as the R11 advert de-duplication in Step 1 - the model sees a
location, not a paperwork entry.

Rows are therefore keyed on the NORMALISED ADDRESS.  Where several
registrations share an address, one row is written and every merged
registration code is recorded.

Output: data/reference/phsrc_source_lists.json

Run:  python src/step2f_build_phsrc_lists.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"

# Noise that varies between registrations of the same building.
ADDR_NOISE = re.compile(
    r"\b(no|nos|number|pvt|private|ltd|limited|plc|the|floor|level|ground|"
    r"1st|2nd|3rd|4th|5th|st|street)\b\.?", re.IGNORECASE)


def norm_address(a: str) -> str:
    s = (a or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = ADDR_NOISE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build(records: list[dict]) -> tuple[list[dict], int]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = norm_address(r["address"])
        if not key:                      # no address - cannot merge safely
            key = f"__nokey__{r['code']}"
        groups.setdefault(key, []).append(r)

    rows, collapsed = [], 0
    for key, members in groups.items():
        members = sorted(members, key=lambda m: m["code"])
        primary = members[0]
        if len(members) > 1:
            collapsed += len(members) - 1
        rows.append({
            "name": primary["name"],
            "address": primary["address"],
            "codes": [m["code"] for m in members],
            "merged_names": [m["name"] for m in members] if len(members) > 1 else [],
        })
    rows.sort(key=lambda r: r["codes"][0])
    return rows, collapsed


def main() -> int:
    src = REF / "phsrc_register.json"
    if not src.exists():
        raise SystemExit("Run step2e_fetch_phsrc.py first.")
    reg = json.loads(src.read_text(encoding="utf-8"))

    out = {"source": reg["retrieved_from"], "note": reg["note"], "categories": {}}
    for prefix, cat in (("PH", "pvt_hospitals"), ("MC", "pvt_med_centers")):
        recs = reg["records"][prefix]
        rows, collapsed = build(recs)
        out["categories"][cat] = {
            "prefix": prefix,
            "label": reg["labels"][prefix],
            "registrations": len(recs),
            "rows_after_address_dedup": len(rows),
            "registrations_collapsed": collapsed,
            "rows": rows,
        }
        print(f"{cat:<18} registrations {len(recs):>4} -> rows {len(rows):>4}  "
              f"(collapsed {collapsed})")
        multi = [r for r in rows if len(r["codes"]) > 1]
        for r in multi[:8]:
            print(f"    merged {r['codes']} :: {r['name'][:46]} @ {r['address'][:44]}")
        if len(multi) > 8:
            print(f"    ... and {len(multi) - 8} more merged groups")

    (REF / "phsrc_source_lists.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REF / 'phsrc_source_lists.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
