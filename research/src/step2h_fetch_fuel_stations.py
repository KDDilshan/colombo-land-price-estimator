"""Step 2h - Fetch fuel station dealer lists (Ceypetco, Lanka IOC, Laugfs).

Sources, per the original study:
  Ceypetco   ceypetco.gov.lk district filling-station pages
  Lanka IOC  lankaioc.com retail outlet list
  Laugfs     laugfseco.com / laugfs.lk filling station list

Scope: Colombo district plus the districts that touch it (Gampaha, Kalutara),
because fuel stations carry the 10 km buffer.  Point-in-polygon decides actual
membership later; the district pages only bound the candidate set.

If one distributor's list is unreachable the other two are still built and the
missing one is reported - never substituted.

Output: data/reference/fuel_stations_source.json

Run:  python src/step2h_fetch_fuel_stations.py
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
UA = "colombo-land-price-replication/1.0 (MSc dissertation replication)"

CEYPETCO_PAGES = {
    "Colombo": ["https://ceypetco.gov.lk/fs-colombo-2/",
                "https://ceypetco.gov.lk/fs-colombo/"],
    "Gampaha": ["https://ceypetco.gov.lk/fs-gampaha/"],
    "Kalutara": ["https://ceypetco.gov.lk/fs-kalutara/",
                 "https://ceypetco.gov.lk/fs-kaluthara/"],
}


def strip_html(s: str) -> str:
    return " ".join(unescape(re.sub(r"(?s)<[^>]+>", " ", s)).split())


def fetch_table_rows(session, url: str) -> list[list[str]]:
    try:
        r = session.get(url, timeout=60)
    except Exception as exc:
        print(f"    ! {url}: {exc}")
        return []
    if r.status_code != 200:
        print(f"    ! {url}: HTTP {r.status_code}")
        return []
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.I | re.S):
        cells = [strip_html(c) for c in
                 re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.I | re.S)]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def main() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    out = {"sources": {}, "unreachable": [], "stations": []}

    # ---- Ceypetco -------------------------------------------------------
    seen = set()
    for district, urls in CEYPETCO_PAGES.items():
        got = 0
        for url in urls:
            rows = fetch_table_rows(session, url)
            if not rows:
                continue
            out["sources"].setdefault("Ceypetco", []).append(url)
            for cells in rows:
                loc = next((c for c in cells if " - " in c or c.isupper()), None)
                if not loc or len(loc) < 4:
                    continue
                if re.match(r"^(location|dealer|account|no\.?|#)$", loc, re.I):
                    continue
                dealer = next((c for c in cells if c != loc), "")
                key = (loc.lower(), dealer.lower())
                if key in seen:
                    continue
                seen.add(key)
                locality = loc.split(" - ")[-1].strip() if " - " in loc else loc
                out["stations"].append({
                    "brand": "Ceypetco",
                    "name": f"Ceypetco Filling Station {locality.title()}",
                    "locality": locality.title(),
                    "raw_location": loc,
                    "dealer": dealer,
                    "source_district": district,
                })
                got += 1
            if got:
                break
        print(f"  Ceypetco {district:<10} {got:>4} stations")
        if got == 0:
            out["unreachable"].append(f"Ceypetco {district}")

    REF.mkdir(parents=True, exist_ok=True)
    out["counts"] = {}
    for b in ("Ceypetco", "Lanka IOC", "Laugfs"):
        out["counts"][b] = sum(1 for s in out["stations"] if s["brand"] == b)
    (REF / "fuel_stations_source.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\ntotal captured: {len(out['stations'])}")
    print(f"by brand: {out['counts']}")
    if out["unreachable"]:
        print(f"UNREACHABLE (reported, not substituted): {out['unreachable']}")
    print(f"wrote {REF / 'fuel_stations_source.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
