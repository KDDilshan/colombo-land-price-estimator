"""Step 5c - merge the three new zones into the cleaned school pool.

Backs up the Colombo-zone-only file first, then applies the SAME Step 3 rules
to the union (invalid coordinates, Sri Lanka bbox, duplicate names, duplicate
coordinates at 5 dp). Nothing else in data/amenities/clean/ is touched.

  data/amenities/clean/govtschools_colombo_zone_only.csv   <- backup (91 rows)
  data/amenities/clean/govtschools.csv                     <- merged pool

Run:  python src/step5c_merge_schools.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step3_clean_amenities import FIELDS, clean_rows, new_stats  # noqa: E402

CLEAN = ROOT / "data" / "amenities" / "clean"
EXTRA = ROOT / "data" / "amenities" / "govtschools_extra.csv"
BACKUP = CLEAN / "govtschools_colombo_zone_only.csv"
TARGET = CLEAN / "govtschools.csv"


def read(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    existing = read(TARGET)
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"backed up {len(existing)} Colombo-zone rows -> {BACKUP.name}")
    else:
        existing = read(BACKUP)
        print(f"backup already exists; rebuilding from it ({len(existing)} rows)")

    extra = read(EXTRA)
    st = new_stats()
    merged = clean_rows(existing + extra, st)

    with open(TARGET, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)

    n_in = sum(1 for r in merged
               if str(r["in_colombo_district"]).strip() in ("1", "True", "true"))
    print(f"\nColombo zone {len(existing)} + three other zones {len(extra)} "
          f"= {len(existing) + len(extra)} in")
    print(f"  dropped: {st['dup_name']} duplicate names, {st['dup_coord']} duplicate "
          f"coordinates, {st['bad_coord']} invalid, {st['out_of_box']} outside bbox")
    print(f"  kept   : {len(merged)} rows, {n_in} in Colombo district")
    for r in st["dup_name_rows"] + st["dup_coord_rows"]:
        print(f"    - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
