"""Step 2l - Fetch branch/outlet lists for banks, finance companies, supermarkets.

BRAND SELECTION follows the original study: the top brands by brand value
(Brand Finance Sri Lanka's Top 100 Most Valuable Brands, banking and
non-banking financial services sectors), not every licensed institution.  The
Central Bank register fixes WHICH institutions are licensed; the brand ranking
fixes which of them the study counts; the institution's own branch directory
supplies the branches, because CBSL publishes head-office details only.

    banks              9 brands
    finance_companies  6 brands
    supermarkets       5 chains (Keells, Cargills, Arpico, Laugfs, SPAR)

RETRIEVABILITY.  Each brand's directory is taken from that brand's own site.
Several are not retrievable from this environment and are recorded as coverage
limitations rather than substituted - the same treatment Lanka IOC received in
step2h:

    HNB, DFCC          WAF rejects every request (F5 "requested URL was
                       rejected"), in-browser navigation included.
    Nations Trust      no branch directory published on the site.
    Central Finance,   Cloudflare interstitial / no branch directory.
    Senkadagala
    all five           no machine-readable outlet list: Keells and Cargills
    supermarkets       serve their locators from authenticated back ends,
                       Arpico's site carries only the head office, SPAR's
                       Sri Lanka site is an online-delivery storefront with no
                       store list.  The supermarket category is therefore NOT
                       produced.  It is not substituted from OSM or Google
                       Places, because membership would then come from the
                       geocoder rather than from the chain - the failure mode
                       recorded in NOTES.md 3A.1.

Output:
  data/reference/banks_source.json
  data/reference/finance_companies_source.json
  data/reference/supermarkets_source.json   (limitation record only)

Run:  python src/step2l_fetch_commercial_lists.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from html import unescape
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

S = requests.Session()
S.headers.update({"User-Agent": UA})


def get(url, **kw):
    return S.get(url, timeout=90, verify=False, **kw)


def post(url, data, **kw):
    return S.post(url, data=data, timeout=90, verify=False, **kw)


def text_lines(html: str) -> list[str]:
    t = re.sub(r"(?s)<(script|style).*?</\1>", " ", html)
    t = re.sub(r"(?s)<[^>]+>", "\n", t)
    return [" ".join(unescape(x).split()) for x in t.split("\n") if x.strip()]


def clean(s: str) -> str:
    return " ".join(unescape(re.sub(r"(?s)<[^>]+>", " ", s or "")).split())


# --------------------------------------------------------------------- banks
def boc() -> list[dict]:
    """Bank of Ceylon - server-rendered touch-point list, 5 per page."""
    rows, page, seen = [], 1, set()
    while page <= 200:
        try:
            L = text_lines(get(f"https://www.boc.lk/branches?page={page}").text)
        except Exception as exc:
            print(f"    ! BOC page {page}: {exc}")
            break
        got = 0
        # Each entry is  NAME / ADDRESS / phone(s) / bocNNN@boc.lk / Get
        # directions / More Details.  Anchor on the e-mail, which is unique and
        # carries the branch code, then walk back over the phone lines.
        for e, ln in enumerate(L):
            if not re.fullmatch(r"boc\d+@boc\.lk", ln):
                continue
            j = e - 1
            while j > 0 and re.fullmatch(r"[\d\s\-+()/]{6,}", L[j]):
                j -= 1
            if j < 1:
                continue
            name, addr = L[j - 1], L[j]
            if not name.rstrip().endswith("Branch") and "Branch" not in name:
                continue          # ATMs, agent points and branches on wheels
            key = (name, addr)
            if key in seen:
                continue
            seen.add(key)
            got += 1
            rows.append({"brand": "Bank of Ceylon", "name": name,
                         "address": addr, "code": ln.split("@")[0]})
        if not got:
            break
        page += 1
        time.sleep(0.15)
    print(f"    BOC {len(rows)} touch points over {page - 1} pages")
    return rows


def combank() -> list[dict]:
    """Commercial Bank - one server-rendered page, '<NAME> BRANCH -' blocks."""
    L = text_lines(get("https://www.combank.lk/branches").text)
    rows = []
    for i, ln in enumerate(L):
        if not re.search(r"\bBRANCH\b\s*-\s*$", ln):
            continue
        name = re.sub(r"\s*-\s*$", "", ln).strip()
        addr = ""
        for j in range(i + 1, min(i + 12, len(L))):
            x = L[j]
            if re.fullmatch(r"\(\d+\)", x) or x in {"ATM", "CDM", "CRM", "ACDM"}:
                continue
            if re.search(r"@combank\.net", x) or re.match(r"^[\d\s\-]{7,}$", x):
                break
            addr = x
            break
        if addr:
            rows.append({"brand": "Commercial Bank of Ceylon",
                         "name": name.title(), "address": addr})
    print(f"    Commercial Bank {len(rows)}")
    return rows


def sampath() -> list[dict]:
    js = get("https://www.sampath.lk/api/branches").json()
    rows = [{"brand": "Sampath Bank", "name": r["branch_name"],
             "address": r.get("address") or "",
             "lat": r.get("latitude"), "lon": r.get("longitude")}
            for r in js if (r.get("category") or "").upper() == "BRANCH"]
    print(f"    Sampath {len(rows)} of {len(js)} entries")
    return rows


PB_REGIONS = {"Colombo": 5, "Gampaha": 7, "Kalutara": 10}


def peoples_bank() -> list[dict]:
    """People's Bank - two-step admin-ajax: region -> branch ids -> detail."""
    url = "https://www.peoplesbank.lk/wp-admin/admin-ajax.php"
    rows = []
    for region, rid in PB_REGIONS.items():
        try:
            opts = post(url, {"action": "branches_filter_ajax",
                              "regionID": str(rid)}).text
        except Exception as exc:
            print(f"    ! People's Bank region {region}: {exc}")
            continue
        ids = re.findall(r'<option value="(\d+)"[^>]*>\s*(.*?)\s*</option>', opts)
        for bid, nm in ids:
            try:
                d = post(url, {"action": "branch_details_ajax",
                               "regionID2": str(rid), "branchID": bid}).text
            except Exception:
                continue
            # The detail block runs  Address : <address>  <phone> <phone>
            # <email> ... so cut at the first phone number or e-mail.
            m = re.search(r"Address\s*:\s*(.*)", clean(d))
            addr = ""
            if m:
                addr = re.split(r"\s(?=[\d][\d\-/\s]{7,}|\S+@|Business Hours"
                                r"|Land Mark|Weekend Banking)", m.group(1))[0]
                addr = clean(addr)
            rows.append({"brand": "People's Bank", "name": clean(nm),
                         "address": addr, "source_region": region})
            time.sleep(0.05)
        print(f"    People's Bank {region}: {len(ids)}")
    return rows


def ndb() -> list[dict]:
    t = get("https://www.ndbbank.com/branch-locator").text
    m = re.search(r"branches:\s*(\[.*?\])\s*,?\s*\n", t, re.S)
    if not m:
        m = re.search(r"branches:\s*(\[.*?\}\])", t, re.S)
    js = json.loads(m.group(1))
    rows = [{"brand": "National Development Bank", "name": r["Name"],
             "address": r.get("Address") or "",
             "lat": r.get("Latitude"), "lon": r.get("Longitude")}
            for r in js if r.get("IsBranch")]
    print(f"    NDB {len(rows)} of {len(js)} entries")
    return rows


def seylan() -> list[dict]:
    rows = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        try:
            js = get(f"https://www.seylan.lk/branch/get-by-letter/{ch}").json()
        except Exception:
            continue
        for f in js.get("features", []):
            p = f.get("properties", {})
            nm = clean(p.get("name"))
            if re.search(r"\bATM\b|Off\s*Site|Off-Site", nm, re.I):
                continue
            c = (f.get("geometry") or {}).get("coordinates") or [None, None]
            rows.append({"brand": "Seylan Bank", "name": nm,
                         "address": clean(p.get("address")),
                         "lat": c[1], "lon": c[0]})
        time.sleep(0.1)
    print(f"    Seylan {len(rows)}")
    return rows


# ---------------------------------------------------------- finance companies
def lolc() -> list[dict]:
    js = get("https://www.lolcfinance.com/map-location/map-location.json").json()
    rows = [{"brand": "LOLC Finance", "name": r["name"],
             "address": r.get("address") or "",
             "lat": r.get("lat"), "lon": r.get("lng")} for r in js]
    print(f"    LOLC {len(rows)}")
    return rows


def plc() -> list[dict]:
    js = get("https://www.plc.lk/wp-admin/admin-ajax.php"
             "?action=asl_load_stores").json()
    rows = [{"brand": "People's Leasing & Finance", "name": clean(r["title"]),
             "address": ", ".join(x for x in (clean(r.get("street")),
                                              clean(r.get("city"))) if x),
             "lat": r.get("lat"), "lon": r.get("lng")} for r in js]
    print(f"    People's Leasing {len(rows)}")
    return rows


def lb_finance() -> list[dict]:
    L = text_lines(get("https://www.lbfinance.com/branch-network").text)
    rows, seen = [], set()
    for i, ln in enumerate(L):
        if not re.search(r"^No[\s.]*\d|,\s*(Colombo|Sri Lanka)\b", ln):
            continue
        if not re.search(r"Road|Mawatha|Street|Rd\b|Colombo|Junction|Place", ln):
            continue
        name = ""
        for j in range(i - 1, max(i - 4, -1), -1):
            if 2 < len(L[j]) < 45 and not re.search(r"\d{6,}|@", L[j]):
                name = L[j]
                break
        key = (name, ln)
        if name and key not in seen:
            seen.add(key)
            rows.append({"brand": "LB Finance", "name": name, "address": ln})
    print(f"    LB Finance {len(rows)}")
    return rows


def commercial_credit() -> list[dict]:
    t = get("https://www.cclk.lk/help/branch-locator/en").text
    m = re.search(r'(\[\s*\{\s*"type"\s*:\s*"Feature".*?\}\s*\])', t, re.S)
    if not m:
        print("    ! Commercial Credit: GeoJSON block not found")
        return []
    js = json.loads(m.group(1))
    rows = []
    for f in js:
        p = f.get("properties", {})
        c = (f.get("geometry") or {}).get("coordinates") or [None, None]
        rows.append({"brand": "Commercial Credit & Finance",
                     "name": clean(p.get("title")),
                     "address": clean((p.get("address") or "").replace("\r\n", ", ")),
                     "lat": float(c[1]) if c[1] else None,
                     "lon": float(c[0]) if c[0] else None})
    print(f"    Commercial Credit {len(rows)}")
    return rows


# ------------------------------------------------------------------ assembly
BANK_BRANDS = [
    ("Bank of Ceylon", boc),
    ("Commercial Bank of Ceylon", combank),
    ("Sampath Bank", sampath),
    ("People's Bank", peoples_bank),
    ("National Development Bank", ndb),
    ("Seylan Bank", seylan),
]
BANK_UNREACHABLE = {
    "Hatton National Bank": "hnb.net rejects every request at the WAF "
                            "(F5 'The requested URL was rejected'), including "
                            "in-browser navigation.",
    "DFCC Bank": "dfcc.lk rejects every request at the WAF (HTTP 403, same F5 "
                 "error page).",
    "Nations Trust Bank": "no branch directory published on nationstrust.com; "
                          "the contact page carries head office only.",
}

FIN_BRANDS = [
    ("LOLC Finance", lolc),
    ("People's Leasing & Finance", plc),
    ("LB Finance", lb_finance),
    ("Commercial Credit & Finance", commercial_credit),
]
FIN_UNREACHABLE = {
    "Central Finance": "cf.lk serves a 403 on every branch path; the contact "
                       "page carries head office only.",
    "Senkadagala Finance": "senfin.com is behind a Cloudflare interstitial that "
                           "does not resolve without a browser challenge.",
}

SUPERMARKET_UNREACHABLE = {
    "Keells Super": "store locator is a React app served from an authenticated "
                    "back end (foxback.keellssuper.com returns 401); no public "
                    "outlet endpoint.",
    "Cargills Food City": "cargillsonline.com store locator is an Angular app; "
                          "every outlet endpoint returns the site error page.",
    "Arpico Supercentre": "arpico.com is a static mirror carrying the head "
                          "office address only; no outlet list.",
    "Laugfs Supermarket": "laugfs.lk map plugin exposes no readable endpoint "
                          "(admin-ajax returns 0).",
    "SPAR Sri Lanka": "spar2u.lk is an online-delivery storefront; its page "
                      "sitemap contains no store-locator page.",
}


def run(brands, unreachable, source_note, out_name):
    rows, counts, dead = [], {}, {}
    for label, fn in brands:
        try:
            got = fn()
        except Exception as exc:
            print(f"    ! {label}: {exc}")
            dead[label] = f"fetch failed: {exc}"
            counts[label] = 0
            continue
        counts[label] = len(got)
        rows.extend(got)
    dead.update(unreachable)

    # De-duplicate on (brand, address) - two registrations at one address are
    # one branch for both the count and the distance variable.
    seen, uniq = set(), []
    for r in rows:
        k = (r["brand"], (r.get("address") or "").lower(), r["name"].lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    doc = {"source": source_note,
           "brands_retrieved": counts,
           "brands_not_retrievable": dead,
           "rows_before_dedup": len(rows),
           "rows": uniq}
    REF.mkdir(parents=True, exist_ok=True)
    (REF / out_name).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    print(f"  -> {out_name}: {len(rows)} -> {len(uniq)} rows, "
          f"{len(dead)} brands not retrievable")
    return doc


def main() -> int:
    print("banks")
    run(BANK_BRANDS, BANK_UNREACHABLE,
        "Brand Finance Sri Lanka Top 100 Most Valuable Brands (banking sector) "
        "fixes the brand set; each brand's own published branch directory "
        "supplies the branches. CBSL "
        "(cbsl.gov.lk/en/authorized-financial-institutions/licensed-commercial-banks) "
        "fixes which institutions are licensed but publishes head-office "
        "details only.", "banks_source.json")

    print("\nfinance companies")
    run(FIN_BRANDS, FIN_UNREACHABLE,
        "Brand Finance Sri Lanka Top 100 Most Valuable Brands (non-banking "
        "financial services) fixes the brand set; each brand's own published "
        "branch directory supplies the branches. CBSL "
        "(cbsl.gov.lk/en/authorized-financial-institutions/licensed-finance-companies) "
        "fixes which institutions are licensed.", "finance_companies_source.json")

    print("\nsupermarkets")
    doc = {"source": "Keells, Cargills Food City, Arpico Supercentre, Laugfs "
                     "Supermarket and SPAR outlet directories",
           "brands_retrieved": {},
           "brands_not_retrievable": SUPERMARKET_UNREACHABLE,
           "rows_before_dedup": 0,
           "rows": []}
    (REF / "supermarkets_source.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  -> supermarkets_source.json: 0 rows, 5 brands not retrievable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
