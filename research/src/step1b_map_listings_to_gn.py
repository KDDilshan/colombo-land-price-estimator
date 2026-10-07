"""Step 1b - Map each land listing to a Grama Niladhari (GN) division.

Location unit is the GN division (section 3.2.1 of the dissertation), never the
town and never the street address.  Every listing must resolve to one of the
merged GN divisions produced by `step1a_build_gn_divisions.py`; listings that do
not resolve are reported, never silently dropped.

Matching cascade (first hit wins, and the stage is recorded on every row):

  1. postal      "Colombo 7" / "Col 07" -> the GN division of that postal area,
                 via the named list POSTAL_TO_GN below.
  2. alias       an entry in data/reference/gn_alias_manual.csv - informal town
                 names, roads, landmarks and Sinhala spellings that are not GN
                 division names themselves.  Every entry carries a rationale.
  3. exact       a token n-gram of the address equals a GN division name after
                 orthographic normalisation (see `sound_key`).
  4. sinhala     the Sinhala address text contains an official Sinhala GN name
                 (adm4_name1 from the boundary source), reduced to its base.
  5. fuzzy       a token n-gram is within FUZZY_CUTOFF of a GN name key.
                 Reported separately so it can be audited.

Text searched, in order: Address_Raw, then Town, then Title.  A hit in an
earlier field wins.  Within one field, an exact hit beats a fuzzy hit; among
equally-good hits the *most specific* division wins (most tokens matched, then
smallest area), because an advertisement reads "<specific place>, <town>".

Name collisions: 8 base names exist in more than one DS division and can be up
to 31 km apart.  They are resolved against an anchor - the GN division implied
by the listing's Town field - by choosing the nearest candidate.  Unresolvable
collisions are reported.

Outputs
-------
data/interim/listings_gn_mapped.csv    one row per Colombo listing + gn_division
results/step1_mapping_report.txt       counts by stage, and every failure
results/step1_unmapped_listings.csv    the listings that did not resolve

Run:  python src/step1b_map_listings_to_gn.py
"""

from __future__ import annotations

import difflib
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"

LISTINGS = Path(r"C:\Users\HP\Desktop\Reserch\THESIS_FINAL\data\dataset_v6.csv")

DISTRICT = "colombo"
FUZZY_CUTOFF = 0.90
OOD_FUZZY_CUTOFF = 0.90   # R12: spelling tolerance on out-of-district names
MAX_NGRAM = 4

# Generic address vocabulary.  An n-gram made up *entirely* of these words can
# never be a place name, so it is never offered to the matcher.  Multi-word GN
# names that contain one of them ("Havelock Town", "Habarakada Watta",
# "Egoda Uyana") are unaffected, because those n-grams also contain a word that
# is not in this set.
NON_PLACE_WORDS = {
    # thoroughfares
    "road", "roads", "rd", "roas", "raod", "lane", "ln", "street", "st",
    "avenue", "ave", "mawatha", "mawata", "mw", "maawatha", "pawatha",
    "place", "pl", "highway", "bypass", "pass", "drive", "path", "para",
    "paara", "cross", "circular", "bus", "route",
    # locality words
    "town", "city", "junction", "jn", "juncion", "junchion", "handiya",
    "village", "gama", "nagar", "nagara", "nagaraya",
    # generic landmark nouns, mostly Sinhala.  "Pitipana Pansala Junction" is
    # "Pitipana temple junction", not a place called Pansala.  NOTE: do not add
    # "vihara" - Vihara is a real GN division in Ratmalana.
    "pansala", "pansela", "palliya", "kovil", "devalaya", "devala", "pola",
    "vidyalaya", "maha", "school", "temple", "church", "mosque", "hospital",
    "bank", "market", "stadium", "ground",
    # positional / marketing filler
    "near", "facing", "front", "side", "back", "main", "new", "old", "upper",
    "lower", "beside", "opposite", "close", "next", "behind", "off", "at",
    "in", "on", "to", "for", "and", "the", "of", "with", "no", "nos",
    # listing filler
    "land", "lands", "plot", "plots", "perch", "perches", "sale", "sell",
    "selling", "house", "houses", "residencies", "residence", "residency",
    "block", "blocks", "phase", "meter", "meters", "metre", "km", "m",
    "sri", "lanka", "colombo",
}

# Words that mark the preceding token as a road name rather than a location.
THOROUGHFARE_WORDS = {
    "road", "roads", "rd", "roas", "raod", "mawatha", "mawata", "mw", "pawatha",
    "lane", "street", "st", "avenue", "ave", "highway", "bypass", "para", "paara",
}

# Columns of the source listing file that this study is allowed to read.  The
# study has no description-derived, seller or environmental features, so those
# columns are never loaded.
LISTING_COLS = [
    "Listing_ID",
    "Title",
    "District",
    "Town",
    "Address_Raw",
    "Land_Size_Perch",
    "Land_Type",
    "Price_LKR",
    "Price_per_Perch",
    "Price_Scale",
    "Posted_Date",
]

# ---------------------------------------------------------------------------
# Named source list: Colombo municipal postal areas -> GN division.
# Sri Lanka Post postal-code areas for Colombo 1-15.  These are the GN division
# names that the dissertation's own address list uses for the same areas.
# ---------------------------------------------------------------------------
POSTAL_TO_GN = {
    1: "Fort",
    2: "Slave Island",
    3: "Kollupitiya",
    4: "Bambalapitiya",
    5: "Havelock Town",
    6: "Wellawatta",
    7: "Kurunduwatta",
    8: "Borella",
    9: "Dematagoda",
    10: "Maradana",
    11: "Pettah",
    12: "Aluthkade",
    13: "Kotahena",
    14: "Grandpass",
    15: "Mattakkuliya",
}

POSTAL_RE = re.compile(
    r"(?:^|[^a-z])(?:colombo|col|cmb)[\s\-.]*0*(\d{1,2})(?![\d])", re.IGNORECASE
)

SINHALA_RE = re.compile(r"[\u0D80-\u0DFF]")

# Sinhala qualifier words, stripped so Sinhala names reduce the same way the
# English ones do in step 1a.
SI_QUALIFIERS = [
    "උතුර", "දකුණ", "නැගෙනහිර", "බටහිර", "මධ්‍යම",
    "ඉහළ", "පහළ", "මෙගොඩ", "මැද", "නගරය",
]


# ---------------------------------------------------------------------------
# Orthographic normalisation
# ---------------------------------------------------------------------------
_ASPIRATES = (
    ("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"),
    ("kh", "k"), ("ph", "p"), ("sh", "s"), ("ch", "c"),
)
_VOWELS = (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("uu", "u"))


def sound_key(token: str) -> str:
    """Collapse the common Sinhala->Latin romanisation variants.

    Kotahena/Kottahena, Thalawathugoda/Talawatugoda, Gangodavila/Gangodawila
    and Kalubovila/Kalubowila all reduce to a single key.
    """
    s = token.lower()
    for a, b in _ASPIRATES:
        s = s.replace(a, b)
    for a, b in _VOWELS:
        s = s.replace(a, b)
    s = s.replace("w", "v")
    s = re.sub(r"(.)\1+", r"\1", s)  # doubled consonants
    return s


_PUNCT_RE = re.compile(r"[^\w\u0D80-\u0DFF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    parts = _PUNCT_RE.sub(" ", text).split()
    return [p for p in parts if p and not p.isdigit()]


def latin_key(text: str) -> str:
    return " ".join(sound_key(t) for t in tokenize(text) if re.search(r"[a-zA-Z]", t))


def si_base(name: str) -> str:
    """Reduce an official Sinhala GN name to its base form."""
    s = name
    changed = True
    while changed:
        changed = False
        for q in SI_QUALIFIERS:
            for trunc in (q, q[:-1]):  # the source truncates some long names
                if len(trunc) < 2:
                    continue
                if s.endswith(" " + trunc) or s.endswith(trunc) and " " in s:
                    cand = s[: s.rfind(trunc)].strip()
                    if cand:
                        s, changed = cand, True
                        break
            if changed:
                break
        for q in ("ඉහළ", "පහළ", "මහ", "කුඩා", "මෙගොඩ", "මැද"):
            if s.startswith(q + " "):
                s, changed = s[len(q) :].strip(), True
    return s


def haversine(lat1, lon1, lat2, lon2, radius=6371.0):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
class GNIndex:
    def __init__(self):
        self.merged = pd.read_csv(REF / "gn_divisions_merged_399.csv")
        self.lookup = pd.read_csv(REF / "gn_lookup.csv")
        self.raw = pd.read_csv(REF / "gn_divisions_raw_557.csv")

        self.area = dict(zip(self.lookup.gn_division, self.lookup.area_sqkm))
        self.coord = {
            r.gn_division: (r.lat, r.lon) for r in self.lookup.itertuples()
        }
        self.names = set(self.lookup.gn_division)

        # per-DS coordinates, for collision resolution
        self.by_ds = defaultdict(list)
        for r in self.merged.itertuples():
            self.by_ds[r.gn_division].append((r.ds_division, r.lat, r.lon))

        # latin key -> {gn_division}
        self.key_index: dict[str, set[str]] = defaultdict(set)
        for name in self.names:
            self.key_index[latin_key(name)].add(name)
        # source (unmerged) names also point at their merged base
        for r in self.raw.itertuples():
            self.key_index[latin_key(r.gn_name_source)].add(r.base_name)

        # Sinhala base -> {gn_division}
        self.si_index: dict[str, set[str]] = defaultdict(set)
        for r in self.raw.itertuples():
            if isinstance(r.gn_name_si, str) and r.gn_name_si.strip():
                self.si_index[si_base(r.gn_name_si.strip())].add(r.base_name)

        # Aliases come from two named lists: hand-written entries with a stated
        # rationale, and entries produced by step 1c where a geocoded point was
        # shown to fall inside a GN division polygon.
        self.alias: dict[str, str] = {}
        self.alias_source: dict[str, str] = {}
        for fname in ("gn_alias_geocoded.csv", "gn_alias_manual.csv"):
            path = REF / fname
            if not path.exists():
                continue
            adf = pd.read_csv(path)
            if adf.empty:
                continue
            for r in adf.itertuples():
                if not isinstance(r.alias, str) or not isinstance(r.gn_division, str):
                    continue
                if r.gn_division not in self.names:
                    raise SystemExit(f"{fname}: {r.gn_division!r} is not a GN division")
                key = (
                    r.alias.strip()
                    if SINHALA_RE.search(r.alias)
                    else latin_key(r.alias)
                )
                if not key:
                    continue
                self.alias[key] = r.gn_division  # manual file loaded last, so it wins
                self.alias_source[key] = fname

        # Places that a listing may name but that lie outside Colombo district.
        # Populated by step 1c from geocoded evidence; listings matching one of
        # these are reported as out-of-district rather than silently mapped.
        self.out_of_district: dict[str, str] = {}
        ood_path = REF / "geocode_out_of_district.csv"
        if ood_path.exists():
            odf = pd.read_csv(ood_path)
            for r in odf.itertuples():
                if isinstance(r.alias, str):
                    key = (
                        r.alias.strip()
                        if SINHALA_RE.search(r.alias)
                        else latin_key(r.alias)
                    )
                    if key:
                        self.out_of_district[key] = str(getattr(r, "osm_district", ""))

        # Decisive other-district place names, checked *before* any in-district
        # match.  Needed because a few GN base names occur in more than one
        # district ("Kurunduwatta, Kirindiwela" is in Gampaha), so naming the
        # other district's town has to win.  Hand-curated, one rationale each.
        self.ood_priority: dict[str, str] = {}
        prio_path = REF / "out_of_district_manual.csv"
        if prio_path.exists():
            pdf = pd.read_csv(prio_path)
            for r in pdf.itertuples():
                if isinstance(r.place, str):
                    self.ood_priority[latin_key(r.place)] = str(r.district)
        self.ood_keys = list(self.ood_priority.keys())

        self.fuzzy_pool = list(self.key_index.keys())

        bad_postal = {n: g for n, g in POSTAL_TO_GN.items() if g not in self.names}
        if bad_postal:
            raise SystemExit(
                f"POSTAL_TO_GN maps to names that are not GN divisions: {bad_postal}"
            )

    def ds_candidates(self, name: str):
        return self.by_ds.get(name, [])


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def find_postal(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    for m in POSTAL_RE.finditer(text):
        n = int(m.group(1))
        if n in POSTAL_TO_GN:
            return POSTAL_TO_GN[n]
    return None


OUT_OF_DISTRICT = "__OUT_OF_DISTRICT__"


def is_all_filler(tokens: list[str]) -> bool:
    """True when every token is generic address vocabulary."""
    return all(t.lower() in NON_PLACE_WORDS for t in tokens)


def scan_out_of_district(text, idx: GNIndex):
    """Return (district, matched_token, how) if `text` names a town outside Colombo.

    Two passes:
      exact  the normalised token is in the named out-of-district list
      fuzzy  the token is within OOD_FUZZY_CUTOFF of a listed name AND is not
             itself an exact GN division name.  This catches spelling variants
             such as "mellawagedara" for the listed "Mallawagedara"; the
             GN-name guard stops "Biyagama" (Gampaha) from swallowing the
             Colombo division "Diyagama", which it resembles at 0.875.

    Only fires when the name is used as a location, not as a road name:
    "Horana Rd, Padukka" and "Kesbewa Bandaragama Rd" are Colombo addresses
    that merely reference the road towards a town in the next district.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    latin = [t for t in tokenize(text) if re.search(r"[a-zA-Z]", t)]
    for pos, tok in enumerate(latin):
        if tok.lower() in NON_PLACE_WORDS:
            continue
        k = sound_key(tok)
        how = None
        if k in idx.ood_priority:
            district, how = idx.ood_priority[k], "exact"
        elif len(k) >= 6 and k not in idx.key_index:
            close = difflib.get_close_matches(
                k, idx.ood_keys, n=1, cutoff=OOD_FUZZY_CUTOFF
            )
            if close:
                district, how = idx.ood_priority[close[0]], f"fuzzy~{close[0]}"
        if how is None:
            continue
        lookahead = {t.lower() for t in latin[pos + 1 : pos + 3]}
        if lookahead & THOROUGHFARE_WORDS:
            continue
        return district, tok, how
    return None


def find_in_text(text: str, idx: GNIndex):
    """Return (gn_division, stage, matched_text) or None."""
    if not isinstance(text, str) or not text.strip():
        return None

    toks = tokenize(text)
    latin = [t for t in toks if re.search(r"[a-zA-Z]", t)]
    keys = [sound_key(t) for t in latin]

    # Decisive other-district town names beat everything else.
    ood = scan_out_of_district(text, idx)
    if ood:
        return OUT_OF_DISTRICT, "out_of_district", f"{ood[0]} [{ood[1]}/{ood[2]}]"

    # whole-string alias first: "Piliyandala Town" must beat any single token
    whole = latin_key(text)
    if whole and whole in idx.alias:
        return idx.alias[whole], "alias", text

    # alias (latin)
    for n in range(min(MAX_NGRAM, len(keys)), 0, -1):
        for i in range(len(keys) - n + 1):
            span = latin[i : i + n]
            if is_all_filler(span):
                continue
            gram = " ".join(keys[i : i + n])
            if gram in idx.alias:
                return idx.alias[gram], "alias", " ".join(span)

    # alias (sinhala) + official sinhala names
    if SINHALA_RE.search(text):
        for al, gn in idx.alias.items():
            if SINHALA_RE.search(al) and al in text:
                return gn, "alias", al
        best = None
        for si_name, gns in idx.si_index.items():
            if len(si_name) >= 3 and si_name in text:
                if best is None or len(si_name) > len(best[0]):
                    best = (si_name, gns)
        if best:
            gn = min(best[1], key=lambda g: idx.area.get(g, 1e9))
            return gn, "sinhala", best[0]

    # exact n-gram
    hits = []
    for n in range(min(MAX_NGRAM, len(keys)), 0, -1):
        for i in range(len(keys) - n + 1):
            span = latin[i : i + n]
            if is_all_filler(span):
                continue
            gram = " ".join(keys[i : i + n])
            if gram in idx.key_index:
                for gn in idx.key_index[gram]:
                    hits.append((n, idx.area.get(gn, 1e9), gn, " ".join(span)))
        if hits:
            break
    if hits:
        hits.sort(key=lambda h: (-h[0], h[1]))
        return hits[0][2], "exact", hits[0][3]

    # Postal area.  Deliberately ranked *below* a named GN division: "Sri
    # Siddhartha Rd, Kirulapone, Colombo 05" belongs to Kirulapone, not to the
    # default division of the Colombo 5 postal area.
    postal = find_postal(text)
    if postal:
        return postal, "postal", text

    # fuzzy
    for n in range(min(MAX_NGRAM, len(keys)), 0, -1):
        cands = []
        for i in range(len(keys) - n + 1):
            span = latin[i : i + n]
            # a fuzzy match is only trusted when no token is generic filler
            if any(t.lower() in NON_PLACE_WORDS for t in span):
                continue
            gram = " ".join(keys[i : i + n])
            if len(gram) < 5:
                continue
            close = difflib.get_close_matches(gram, idx.fuzzy_pool, n=1, cutoff=FUZZY_CUTOFF)
            if close:
                ratio = difflib.SequenceMatcher(None, gram, close[0]).ratio()
                for gn in idx.key_index[close[0]]:
                    cands.append((ratio, n, -idx.area.get(gn, 0), gn, " ".join(span)))
        if cands:
            cands.sort(reverse=True)
            best = cands[0]
            return best[3], "fuzzy", f"{best[4]} ~ {best[3]}"

    # Nothing in Colombo district was found in this text.  If it names a place
    # that step 1c geocoded to another district, say so - the caller treats that
    # as terminal so a coarse Town tag cannot pull a Kalutara or Gampaha
    # advertisement back into the study area.
    if whole and whole in idx.out_of_district:
        return OUT_OF_DISTRICT, "out_of_district", idx.out_of_district[whole]
    for n in range(min(MAX_NGRAM, len(keys)), 0, -1):
        for i in range(len(keys) - n + 1):
            span = latin[i : i + n]
            if is_all_filler(span):
                continue
            gram = " ".join(keys[i : i + n])
            if gram in idx.out_of_district:
                return OUT_OF_DISTRICT, "out_of_district", idx.out_of_district[gram]
    return None


def resolve_collision(gn: str, idx: GNIndex, anchor: tuple[float, float] | None):
    """Pick the DS-specific instance of a colliding base name."""
    cands = idx.ds_candidates(gn)
    if len(cands) <= 1:
        return (cands[0][0] if cands else None), False
    if anchor is None:
        return None, True
    best = min(cands, key=lambda c: haversine(anchor[0], anchor[1], c[1], c[2]))
    return best[0], False


def main() -> int:
    INTERIM.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    idx = GNIndex()
    n_multi = int((idx.lookup.n_ds_divisions > 1).sum())
    print(f"GN divisions, counted within DS : {len(idx.merged)}")
    print(f"distinct GN division names      : {len(idx.names)} "
          f"({n_multi} names span >1 DS division, "
          f"accounting for the {len(idx.merged) - len(idx.names)} row difference)")
    print(f"aliases loaded                  : {len(idx.alias)}")

    df = pd.read_csv(LISTINGS, low_memory=False, usecols=LISTING_COLS)
    # Stable link back to the source file.  Listing_ID is the advert slug
    # truncated to 24 characters and is NOT unique (3526 distinct over 8669
    # rows), so it cannot be used to identify a row.
    df["source_row_id"] = df.index
    total = len(df)
    df = df[df.District.str.lower().str.strip() == DISTRICT].copy()
    print(f"\nlistings in file            : {total}")
    print(f"Colombo district listings   : {len(df)}")

    gn_col, stage_col, src_col, hit_col, ds_col = [], [], [], [], []
    ambiguous = []

    for row in df.itertuples():
        # anchor from the Town field, used only to resolve name collisions
        anchor = None
        town_hit = find_in_text(getattr(row, "Town", None), idx)
        if town_hit:
            anchor = idx.coord.get(town_hit[0])

        # R12 - if the seller's own text (Address_Raw) or the advert Title
        # names a town outside Colombo district, the listing is excluded
        # regardless of what the coarse ikman Town tag says and regardless of
        # which stage would otherwise have matched.  This runs before the field
        # cascade so it outranks exact and alias matches taken from Town, not
        # only postal ones.  The road-name guard still applies inside
        # scan_out_of_district.
        ood_hit, ood_field = None, None
        for fname in ("Address_Raw", "Title"):
            hit = scan_out_of_district(getattr(row, fname, None), idx)
            if hit:
                ood_hit, ood_field = hit, fname
                break
        if ood_hit:
            district, tok, how = ood_hit
            gn_col.append(None); stage_col.append("out_of_district")
            src_col.append(ood_field)
            hit_col.append(f"{district} [{tok}/{how}]")
            ds_col.append(None)
            continue

        result, field = None, None
        for fname in ("Address_Raw", "Town", "Title"):
            hit = find_in_text(getattr(row, fname, None), idx)
            if hit:
                result, field = hit, fname
                break

        if result is None:
            gn_col.append(None); stage_col.append("unmapped")
            src_col.append(None); hit_col.append(None); ds_col.append(None)
            continue

        gn, stage, matched = result
        # R7 (Title outranks a postal match from Town) is subsumed by the R12
        # pre-pass above, which scans Address_Raw and Title before any field
        # match is accepted.
        if gn is OUT_OF_DISTRICT or gn == OUT_OF_DISTRICT:
            gn_col.append(None); stage_col.append("out_of_district")
            src_col.append(field); hit_col.append(matched); ds_col.append(None)
            continue
        ds, amb = resolve_collision(gn, idx, anchor)
        if amb:
            ambiguous.append((row.Listing_ID, gn))
        gn_col.append(gn); stage_col.append(stage)
        src_col.append(field); hit_col.append(matched); ds_col.append(ds)

    df["gn_division"] = gn_col
    df["match_stage"] = stage_col
    df["match_field"] = src_col
    df["match_text"] = hit_col
    df["ds_division"] = ds_col
    df = df.merge(
        idx.lookup[["gn_division", "lat", "lon"]].rename(
            columns={"lat": "gn_lat", "lon": "gn_lon"}
        ),
        on="gn_division",
        how="left",
    )

    mapped = df.gn_division.notna()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    ood = df.match_stage == "out_of_district"
    unmapped_only = (~mapped) & (~ood)

    emit("\n" + "=" * 72)
    emit("STEP 1 - GN DIVISION MAPPING REPORT")
    emit("=" * 72)
    emit(f"listings tagged Colombo district : {len(df)}")
    emit(f"mapped to a GN division          : {mapped.sum()}  ({mapped.mean()*100:.2f}%)")
    emit(f"excluded, geocode outside Colombo: {ood.sum()}  ({ood.mean()*100:.2f}%)")
    emit(f"unmapped                         : {unmapped_only.sum()}  ({unmapped_only.mean()*100:.2f}%)")
    emit("")
    emit("by matching stage:")
    for stage, n in df.match_stage.value_counts().items():
        emit(f"  {stage:10s} {n:6d}  ({n/len(df)*100:5.2f}%)")
    emit("")
    emit("by field the match came from:")
    for f, n in df.match_field.value_counts().items():
        emit(f"  {f:12s} {n:6d}")
    emit("")
    emit(f"GN division names represented : {df.gn_division.nunique()} of "
         f"{len(idx.names)} distinct names ({len(idx.merged)} divisions counted within DS)")
    emit(f"ambiguous DS collisions       : {len(ambiguous)}")

    fuzzy = df[df.match_stage == "fuzzy"]
    if len(fuzzy):
        emit(f"\nfuzzy matches to audit ({len(fuzzy)} rows, {fuzzy.match_text.nunique()} distinct):")
        for t, n in fuzzy.match_text.value_counts().head(60).items():
            emit(f"  {n:5d}  {t}")

    if ood.sum():
        emit(f"\nEXCLUDED - location geocodes outside Colombo district ({ood.sum()} listings):")
        for (addr, dist), n in (
            df[ood].groupby([df.Address_Raw.fillna("<no address>"), df.match_text]).size().items()
        ):
            emit(f"  {n:5d}  {str(dist):20s}  {addr}")

    unmapped = df[unmapped_only]
    emit(f"\nUNMAPPED LISTINGS ({len(unmapped)}) - distinct location text:")
    umd = (
        unmapped.groupby(unmapped.Address_Raw.fillna("<no address>"))
        .agg(n=("Listing_ID", "size"), town=("Town", "first"))
        .sort_values("n", ascending=False)
    )
    for addr, r in umd.iterrows():
        emit(f"  {r['n']:5d}  town={str(r['town'])[:20]:20s}  {addr}")

    emit("\ntop 30 GN divisions by listing count:")
    for g, n in df.gn_division.value_counts().head(30).items():
        emit(f"  {n:6d}  {g}")

    df.to_csv(INTERIM / "listings_gn_mapped.csv", index=False, encoding="utf-8")
    unmapped.to_csv(RESULTS / "step1_unmapped_listings.csv", index=False, encoding="utf-8")
    df[ood].to_csv(
        RESULTS / "step1_out_of_district_listings.csv", index=False, encoding="utf-8"
    )
    (RESULTS / "step1_mapping_report.txt").write_text("\n".join(lines), encoding="utf-8")
    emit(f"\nWrote {INTERIM / 'listings_gn_mapped.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
