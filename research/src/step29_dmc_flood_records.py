"""Step 29 - Colombo district flood records from the Disaster Management
Centre's DesInventar database (Disaster Information Management System in Sri
Lanka, DMC with UNDP; records since 1974).

SOURCE.  The DMC database as exported on desinventar.net
(DI_export_lka.zip, retrieved 2026-09-23; the DMC's own server
desinventar.lk:8081 returned empty query lists that day). The 1.2 GB XML is
streamed straight out of the zip; nothing is extracted to disk.

GEOGRAPHY.  level1 = district, level2 = DS division - the finest coded level.
GN divisions appear only as free text in `lugar` (place), and the extension
field `no_gn_affe` gives the NUMBER of GNs affected, not which ones. So this
source supports a DS-level flood history, not a GN-level one.

EVENTS KEPT.  FLOOD, FLASH FLOOD, CYCLONE & FLOOD. HEAVY RAINS is counted
separately and reported, not merged, because a rain record does not say that
land went under water.

Outputs
  data/processed/dmc_flood_records_colombo.csv   one row per record
  data/processed/dmc_flood_by_ds.csv             one row per Colombo DS division
"""

import os
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "data", "raw", "dmc_desinventar", "DI_export_lka.zip")
PROC = os.path.join(ROOT, "data", "processed")
FLOOD = {"FLOOD", "FLASH FLOOD", "CYCLONE & FLOOD"}
RAIN = {"HEAVY RAINS"}
KEEP = ["serial", "level1", "level2", "name1", "name2", "evento", "lugar", "fechano",
        "fechames", "fechadia", "muertos", "afectados", "vivdest", "vivafec",
        "evacuados", "fuentes", "clave"]


def main():
    f = zipfile.ZipFile(ZIP).open("DI_export_lka.xml")
    lev2, recs, ext = {}, [], {}
    stack = []
    for ev, el in ET.iterparse(f, events=("start", "end")):
        if ev == "start":
            stack.append(el.tag)
            continue
        stack.pop()
        if el.tag != "TR" or not stack:
            if len(stack) <= 1:
                el.clear()
            continue
        sec = stack[-1]
        if sec == "lev2":
            lev2[el.findtext("lev2_cod")] = (el.findtext("lev2_name"), el.findtext("lev2_lev1"))
        elif sec == "fichas":
            evento = (el.findtext("evento") or "").strip().upper()
            if el.findtext("name1", "").strip().lower() == "colombo" and evento in FLOOD | RAIN:
                recs.append({k: (el.findtext(k) or "").strip() for k in KEEP})
        elif sec == "extension":
            ext[el.findtext("clave_ext")] = el.findtext("no_gn_affe")
        el.clear()

    df = pd.DataFrame(recs)
    df["no_gn_affected"] = pd.to_numeric(df.clave.map(ext), errors="coerce")
    for c in ("fechano", "muertos", "afectados", "vivdest", "vivafec", "evacuados"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ds_division"] = df.name2.replace("", pd.NA)
    df["event_class"] = df.evento.str.upper().map(lambda e: "flood" if e in FLOOD else "heavy_rain")
    df.to_csv(os.path.join(PROC, "dmc_flood_records_colombo.csv"), index=False)

    fl = df[df.event_class == "flood"]
    by = (fl.dropna(subset=["ds_division"]).groupby("ds_division")
          .agg(flood_records=("serial", "count"),
               flood_years=("fechano", lambda s: int(s.nunique())),
               first_year=("fechano", "min"), last_year=("fechano", "max"),
               people_affected=("afectados", "sum"),
               houses_damaged=("vivafec", "sum"))
          .sort_values("flood_records", ascending=False))
    ds_all = sorted(n for n, p in lev2.values() if p == "lka001001")
    by = by.reindex(sorted(set(ds_all) | set(by.index))).fillna(0)
    by.to_csv(os.path.join(PROC, "dmc_flood_by_ds.csv"))

    print(f"Colombo records: {len(df)} ({(df.event_class == 'flood').sum()} flood, "
          f"{(df.event_class == 'heavy_rain').sum()} heavy rain)")
    print(f"years {int(df.fechano.min())}-{int(df.fechano.max())}; "
          f"flood records without a DS division: {fl.ds_division.isna().sum()}")
    print(f"'lugar' filled on {(fl.lugar != '').sum()} flood records")
    print(by.to_string())


if __name__ == "__main__":
    main()
