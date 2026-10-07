"""Step 2e - Fetch the PHSRC register (address-bearing edition).

SOURCE NOTE - this is NOT a source substitution.  Both pages are PHSRC:

  phsrc.lk/pages_e.php?id=12                    partial listing: registration
                                                code + name only, no address.
                                                Cannot be geocoded under the
                                                project's guards, because with
                                                no locality there is no anchor
                                                and the acceptance-radius check
                                                cannot be applied.
  phsrc.lk/mis/web/web/regInstitute26.php       the complete edition, carrying
                                                full street addresses.

Moving from the abridged view to the full one is an upgrade within the same
authority, not a change of authority.

CATEGORY ASSIGNMENT is by the registration-code prefix, which is the register's
own named classification:
    PHSRC/PH/...  Private Hospitals, Nursing Homes & Maternity Homes
    PHSRC/MC/...  Medical Centres
This is NOT a residual bucket.  PH and MC membership is asserted by the
register; neither is derived by excluding the other or anything else.

Output: data/reference/phsrc_register.json

Run:  python src/step2e_fetch_phsrc.py
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

BASE = "https://www.phsrc.lk/mis/web/web/regInstitute26.php"
PARTIAL = "https://www.phsrc.lk/pages_e.php?id=12"
UA = "colombo-land-price-replication/1.0 (MSc dissertation replication)"

PREFIX_LABEL = {
    "PH": "Private Hospitals, Nursing Homes & Maternity Homes",
    "MC": "Medical Centres",
    "L": "Laboratories",
    "OMI": "Other Medical Institutions",
    "PGP": "Part Time General Practitioners",
    "FGP": "Full Time General Practitioners",
    "PDS": "Part Time Dental Surgeries",
    "FDS": "Full Time Dental Surgeries",
}

ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.S)


def strip_html(s: str) -> str:
    return " ".join(unescape(re.sub(r"(?s)<[^>]+>", " ", s)).split())


def main() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    records: dict[str, dict] = {}
    # The MIS form exposes institution-type and province selectors; sweeping
    # both and de-duplicating on the registration code yields the whole register
    # without relying on any one selector being complete.
    for ins in range(1, 9):
        for pro in (0, 4):
            url = f"{BASE}?ins={ins}&pro={pro}"
            try:
                r = session.get(url, timeout=60)
            except Exception as exc:
                print(f"  ! {url}: {exc}")
                continue
            if r.status_code != 200:
                continue
            for row in ROW_RE.findall(r.text):
                cells = [strip_html(c) for c in CELL_RE.findall(row)]
                if len(cells) >= 3 and cells[0].startswith("PHSRC/"):
                    code = cells[0]
                    if code not in records:
                        records[code] = {"name": cells[1], "address": cells[2]}
            print(f"  ins={ins} pro={pro}: cumulative {len(records)}")

    by_prefix: dict[str, list] = {}
    for code, v in records.items():
        parts = code.split("/")
        if len(parts) < 3:
            continue
        by_prefix.setdefault(parts[1], []).append(
            {"code": code, "name": v["name"], "address": v["address"]})

    out = {
        "retrieved_from": BASE,
        "partial_listing_also_exists": PARTIAL,
        "note": ("Both URLs are PHSRC. The pages_e.php?id=12 listing carries "
                 "code+name only and cannot be geocoded under the project's "
                 "guards (no locality anchor). The MIS edition carries "
                 "addresses. Category is the registration-code prefix, the "
                 "register's own named classification - not a residual bucket."),
        "counts": {p: len(v) for p, v in sorted(by_prefix.items())},
        "labels": PREFIX_LABEL,
        "records": {p: sorted(v, key=lambda x: x["code"]) for p, v in by_prefix.items()},
    }
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "phsrc_register.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\ntotal distinct registrations: {len(records)}")
    for p, n in sorted(out["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {p:<5} {n:>4}  {PREFIX_LABEL.get(p, '?')}")
    print(f"\nwrote {REF / 'phsrc_register.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
