"""Step 2k - Fetch the Colombo Zonal Education Office school list.

ONE list, not three.  The original study split government schools into Class A
(top-20 A/L performers per stream), Class B and semi-government using
Department of Examinations ranking data.  That ranking was not retrievable, so
the split is not reproduced - see NOTES.md.  Every school the zone administers
goes into one file.

Pages taken (the zone's own classification):
  /national-school/    national schools
  /provincial-school/  provincial council schools
  /semi-gov-school/    semi-government schools

Pages NOT taken: /private-school/ (not government; the private/international
sector is already covered by intlschools.csv) and /piriven-school/ (pirivenas
are monastic institutions, not general schools).

Each block on these pages is:
    <school name>
    School Level - ...
    Principal Name - ...
    Year Span - ...
    Medium - ...
    Gender - ...
    Contact - ...
    Address - <postal locality>

so the name is the line before "School Level" and the address is the first
"Address -" line after it.

Output: data/reference/schools_source.json

Run:  python src/step2k_fetch_schools.py
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

BASE = "https://cmbez.wp.gov.lk"
PAGES = {
    "national": f"{BASE}/national-school/",
    "provincial": f"{BASE}/provincial-school/",
    "semi_government": f"{BASE}/semi-gov-school/",
}

# Headings and menu items that sit in the same text stream as the school names.
# A field line, e.g. "Medium - Sinhala".  Matched on the LABEL, not on the
# presence of " - " - "Hindu College - Colombo 15" is a school name.
FIELD = re.compile(r"^(school level|school type|principal name|year span|medium"
                   r"|gender|contact|address)\b", re.I)

NOISE = re.compile(
    r"^(north|central|south|borella)\s+colombo$|^colombo\s+(north|central|south)$"
    r"|^borella$|^home$|^about$|^contact$|^gallery$|^events$|^blog$"
    r"|^national schools?$|^provincial schools?$|^semi|^private|^piriven"
    r"|^schools?$|^more$|^read more$|^\W*$", re.I)


def page_lines(session, url: str) -> list[str]:
    r = session.get(url, timeout=90, verify=False)
    r.raise_for_status()
    t = re.sub(r"(?s)<(script|style).*?</\1>", " ", r.text)
    t = re.sub(r"(?s)<[^>]+>", "\n", t)
    return [unescape(x).strip() for x in t.split("\n") if x.strip()]


def parse(lines: list[str], level: str) -> list[dict]:
    out = []
    for i, ln in enumerate(lines):
        if not ln.lower().startswith("school level"):
            continue
        name = ""
        for j in range(i - 1, max(i - 4, -1), -1):
            cand = lines[j]
            if NOISE.match(cand) or FIELD.match(cand):
                continue
            name = cand
            break
        addr = ""
        for k in range(i + 1, min(i + 9, len(lines))):
            if lines[k].lower().startswith("address"):
                addr = lines[k].split("-", 1)[1].strip() if "-" in lines[k] else ""
                break
        if name:
            out.append({"name": " ".join(name.split()),
                        "address": " ".join(addr.split()),
                        "zone_class": level})
    return out


def main() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    rows, counts, unreachable = [], {}, []
    for level, url in PAGES.items():
        try:
            got = parse(page_lines(session, url), level)
        except Exception as exc:
            print(f"  ! {url}: {exc}")
            unreachable.append(url)
            counts[level] = 0
            continue
        counts[level] = len(got)
        rows.extend(got)
        print(f"  {level:<16} {len(got):>4} schools  <- {url}")

    # De-duplicate on (name, address): a school listed on two pages is one
    # school for both the count and the distance variable.
    seen, uniq = set(), []
    for r in rows:
        k = (r["name"].lower(), r["address"].lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    doc = {
        "source": "Colombo Zonal Education Office, cmbez.wp.gov.lk "
                  "(national, provincial and semi-government school pages)",
        "source_urls": PAGES,
        "excluded_pages": {
            f"{BASE}/private-school/": "not government; the private/international "
                                       "sector is covered by intlschools.csv",
            f"{BASE}/piriven-school/": "pirivenas are monastic institutions, not "
                                       "general schools",
        },
        "unreachable": unreachable,
        "counts": counts,
        "rows_before_dedup": len(rows),
        "rows": uniq,
    }
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "schools_source.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} rows -> {len(uniq)} after (name,address) de-duplication")
    no_addr = [r["name"] for r in uniq if not r["address"]]
    print(f"rows with no address: {len(no_addr)}")
    for n in no_addr[:10]:
        print(f"  - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
