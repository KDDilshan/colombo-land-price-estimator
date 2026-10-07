"""Step 5a - close the school coverage hole (quality-report limitation #3).

`schools_source.json` covers the Colombo Education Zone only (cmbez.wp.gov.lk).
Colombo district has four zones. This step retrieves the other three.

WHAT IS AVAILABLE, AND WHAT IS NOT
  Piliyandala zone       pilizone.lk - the zone's OWN site, one page per
                         division (Dehiwala, Moratuwa, Kesbewa). Carries the
                         school type (1AB / 1C / 2 / 3) and the address.
  Sri Jayawardenapura    NO SITE. wpedu.sch.lk (the Western Province department
                         site that would carry it) does not resolve in DNS at
                         all; no *.wp.gov.lk host exists for the zone.
  Homagama               NO SITE. The address published for it,
                         www.mahozone.sch.lk, does not connect.

SUBSTITUTION, DECLARED
  For the two zones with no site, schools are taken from the Divisional
  Secretariat pages `<division>.ds.gov.lk/index.php/en/schools.html`, one per DS
  division of the zone. These are official government pages listing the schools
  in the division by name and address. They are NOT the zonal education office
  and they do NOT classify a school as government / semi-government / private,
  so every school taken from them is recorded as `zone_class = "government"`
  with `source_kind = "ds_secretariat"`; the semi-government pool keeps only the
  6 units the Colombo zone itself declares.

EXCLUDED everywhere: pirivenas (monastic, not general schools) and anything
naming itself International / private - the private and international sector is
covered by intlschools.csv and must not be double-counted here.

Output: data/reference/schools_district_source.json
Run:    python src/step5a_fetch_district_schools.py
"""

from __future__ import annotations

import json
import re
import sys
from html import unescape
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
UA = "colombo-land-price-replication/1.0 (MSc dissertation replication)"

# zone -> the DS divisions it administers, and where its schools were found
PILIZONE = {
    "Dehiwala": "https://pilizone.lk/dehiwaladivision/",
    "Moratuwa": "https://pilizone.lk/moratuwadivision/",
    "Kesbewa": "https://pilizone.lk/kesbewadivision/",
}
DS_PAGES = {
    # Piliyandala zone - no division page on the zone site for Ratmalana
    "Ratmalana": "http://ratmalana.ds.gov.lk/index.php/en/schools.html",
    # Sri Jayawardenapura zone
    "Kaduwela": "http://kaduwela.ds.gov.lk/index.php/en/schools.html",
    "Kolonnawa": "http://kolonnawa.ds.gov.lk/index.php/en/schools.html",
    "Maharagama": "http://maharagama.ds.gov.lk/index.php/en/schools.html",
    "Sri Jayawardanapura Kotte": "http://kotte.ds.gov.lk/index.php/en/schools.html",
    # Homagama zone
    "Homagama": "http://homagama.ds.gov.lk/index.php/en/schools.html",
    "Seethawaka": "http://seethawaka.ds.gov.lk/index.php/en/schools.html",
    "Padukka": "http://padukka.ds.gov.lk/index.php/en/schools.html",
}
ZONE_OF = {
    "Dehiwala": "Piliyandala", "Moratuwa": "Piliyandala", "Kesbewa": "Piliyandala",
    "Ratmalana": "Piliyandala",
    "Kaduwela": "Sri Jayawardenapura", "Kolonnawa": "Sri Jayawardenapura",
    "Maharagama": "Sri Jayawardenapura",
    "Sri Jayawardanapura Kotte": "Sri Jayawardenapura",
    "Homagama": "Homagama", "Seethawaka": "Homagama", "Padukka": "Homagama",
}

SCHOOL = re.compile(
    r"vidyalaya|vidayala|viddalaya|vidyala|college|school|\bm\.?v\.?\b|\bk\.?v\.?\b"
    r"|\bm\.?m\.?v\.?\b|balika|maha vid|kanishta|kanita|central|convent|madya",
    re.I)
EXCLUDE = re.compile(r"piriven|international|montessori|pre.?school|tuition", re.I)
PHONE = re.compile(r"^[\d\s()+/,-]{6,}$")
CLASSRANGE = re.compile(r"^\s*\d{1,2}\s*[-–]\s*\d{1,2}\s*$")
TYPECODE = re.compile(r"^\s*(1\s?ab|1\s?c|2|3|type\s?\d)\s*$", re.I)
NOISE = re.compile(r"^(m/f|mix|male|female|f|m|no\.?|address|tel|telephone|classes"
                   r"|calcification|classification|name|school.*no|-|\W*)$", re.I)
URL = re.compile(r"https?://|goo\.gl|maps\.", re.I)
# a header row describes the table instead of holding a school
HEADER = re.compile(r"school\s*(reg|census)|name\s*of\s*(the\s*)?school|class\s*range"
                    r"|school\s*type|g\.?n\.?\s*division|telephone|^address$", re.I)


def cells(html: str):
    """Yield the text cells of every table row on the page."""
    html = re.sub(r"(?s)<(script|style).*?</\1>", " ", html)
    for tr in re.findall(r"(?s)<tr[^>]*>(.*?)</tr>", html, re.I):
        out = []
        for td in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", tr, re.I):
            txt = unescape(re.sub(r"(?s)<[^>]+>", " ", td))
            out.append(" ".join(txt.split()))
        if out:
            yield out


def usable(c: str) -> bool:
    return bool(c) and not (PHONE.match(c) or CLASSRANGE.match(c) or URL.search(c)
                            or TYPECODE.match(c) or NOISE.match(c))


def parse_rows(html: str, division: str, kind: str):
    """A row contributes one school if any cell reads like a school name."""
    got = []
    for row in cells(html):
        if any(HEADER.search(c) for c in row):
            continue
        # the zonal tables put "<name>, <locality>." in the last cell
        if kind == "zonal":
            idx = len(row) - 1 if row and usable(row[-1]) else None
            if idx is not None and (not SCHOOL.search(row[idx])
                                    or EXCLUDE.search(row[idx])):
                idx = None
        else:
            idx = next((i for i, c in enumerate(row)
                        if usable(c) and SCHOOL.search(c) and not EXCLUDE.search(c)), None)
        if idx is None:
            continue
        name = row[idx]
        addr = ""
        # pilizone packs "Name, Locality." into one cell
        if kind == "zonal" and "," in name:
            head, tail = name.rsplit(",", 1)
            if len(tail.strip(" .")) > 2 and not SCHOOL.search(tail):
                name, addr = head.strip(), tail.strip(" .")
        if not addr:
            for c in row[idx + 1:]:
                if usable(c) and not SCHOOL.search(c):
                    addr = c
                    break
        if len(name) < 5:
            continue
        got.append({"name": " ".join(name.split()),
                    "address": " ".join(addr.split()) or division,
                    "ds_division": division,
                    "zone": ZONE_OF[division],
                    "zone_class": "government",
                    "source_kind": kind})
    return got


def main() -> int:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    rows, per_page, unreachable = [], {}, {}

    for division, url in list(PILIZONE.items()) + list(DS_PAGES.items()):
        kind = "zonal" if "pilizone" in url else "ds_secretariat"
        try:
            r = s.get(url, timeout=90, verify=False)
            r.raise_for_status()
        except Exception as exc:
            unreachable[url] = f"{type(exc).__name__}: {exc}"
            per_page[division] = 0
            print(f"  ! {division:28s} {url}  {type(exc).__name__}")
            continue
        got = parse_rows(r.text, division, kind)
        per_page[division] = len(got)
        rows.extend(got)
        print(f"  {division:28s} {len(got):4d} schools  <- {url}")

    seen, uniq = set(), []
    for r in rows:
        k = (r["name"].lower(), r["address"].lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    doc = {
        "source": "Piliyandala Zonal Education Office (pilizone.lk) for the "
                  "Piliyandala zone; Divisional Secretariat school pages "
                  "(<division>.ds.gov.lk) for the Sri Jayawardenapura and "
                  "Homagama zones, which have no website of their own",
        "substitution_declared": {
            "Sri Jayawardenapura zone": "no zonal website exists - wpedu.sch.lk "
                                        "does not resolve in DNS and no "
                                        "*.wp.gov.lk host answers for the zone; "
                                        "substituted with the DS secretariat "
                                        "school pages of Kaduwela, Kolonnawa, "
                                        "Maharagama and Kotte",
            "Homagama zone": "no zonal website - www.mahozone.sch.lk does not "
                             "connect; substituted with the DS secretariat "
                             "school pages of Homagama, Seethawaka and Padukka",
        },
        "classification_limitation": "the DS secretariat pages do not say whether "
                                     "a school is government or semi-government, "
                                     "so every school from them is recorded as "
                                     "government; the semi-government pool keeps "
                                     "only the 6 units the Colombo zone declares",
        "excluded": "pirivenas, international and private schools "
                    "(intlschools.csv covers that sector)",
        "source_urls": {**PILIZONE, **DS_PAGES},
        "unreachable": unreachable,
        "counts_per_division": per_page,
        "rows_before_dedup": len(rows),
        "rows": uniq,
    }
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "schools_district_source.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} rows -> {len(uniq)} after (name, address) de-duplication")
    return 0


if __name__ == "__main__":
    sys.exit(main())
