"""Tests for predictor.py.

    python test_predictor.py        (or: pytest test_predictor.py)

Nothing here retrains or refits - the saved tuned XGBoost model is used as it stands.
"""

import math

import predictor

FLOOR = 100_000
CEILING = 20_000_000

KNOWN = ["nugegoda", "maharagama", "battaramulla"]


def test_known_divisions_predict_a_sensible_price():
    for division in KNOWN:
        assert division in predictor.DIVISION_VALUES.index, f"{division} is not one of the 194"
        out = predictor.predict(division=division, perches=10,
                                land_type="Land_type_Residential")
        price = out["price_per_perch"]
        assert FLOOR < price < CEILING, f"{division}: {price:,.0f} outside the expected range"
        assert out["known_division"] is True
        assert math.isclose(out["total_price"], price * 10, rel_tol=1e-9)
        print(f"  {division:14s} LKR {price:>12,.0f} per perch")


def test_land_types_come_from_the_feature_list():
    assert predictor.LAND_TYPE_COLUMNS
    assert all(c in predictor.FEATURE_ORDER for c in predictor.LAND_TYPE_COLUMNS)
    assert len(predictor.FEATURE_ORDER) == 50           # 49 + ndvi (step27)


def test_feature_row_matches_the_model_column_order():
    out = predictor.predict(division="nugegoda", perches=10,
                            land_type="Land_type_Residential")
    assert out["perches"] == 10
    assert list(predictor.DIVISION_VALUES.columns) + ["Address_target_enc"] != []
    # exactly one land type flag is set
    row = predictor.DIVISION_VALUES.loc["nugegoda"].copy()
    for col in predictor.LAND_TYPE_COLUMNS:
        row[col] = 0.0
    row["Land_type_Residential"] = 1.0
    assert sum(row[c] for c in predictor.LAND_TYPE_COLUMNS) == 1.0


def test_unknown_division_falls_back_to_colombo_wide_values():
    out = predictor.predict(division="not-a-real-division", perches=10,
                            land_type="Land_type_Residential")
    assert out["known_division"] is False
    assert FLOOR < out["price_per_perch"] < CEILING
    print(f"  unknown        LKR {out['price_per_perch']:>12,.0f} per perch (Colombo-wide)")


def test_point_lookup_resolves_to_a_division():
    # Nugegoda town centre
    found = predictor.locate(6.8649, 79.8997)
    assert found["division"] is not None
    assert found["match"] in ("polygon", "nearest")
    print(f"  (6.8649, 79.8997) -> {found['division']} [{found['match']}]")


def test_polygon_area_is_geodesic():
    # a ~100 m square at Colombo's latitude
    d = 100 / 111_320
    lat, lon = 6.9271, 79.8612
    ring = [(lat, lon), (lat, lon + d / math.cos(math.radians(lat))),
            (lat + d, lon + d / math.cos(math.radians(lat))), (lat + d, lon)]
    area = predictor.polygon_area_m2(ring)
    assert 9_500 < area < 10_500, f"expected ~10,000 m2, got {area:,.0f}"
    perches = predictor.m2_to_perches(area)
    assert 370 < perches < 420, f"expected ~395 perches, got {perches:.1f}"
    print(f"  100 m square -> {area:,.0f} m2 = {perches:.1f} perches")


def test_flood_profile_is_present_and_read_only():
    out = predictor.predict(division="nugegoda", perches=10,
                            land_type="Land_type_Residential")
    for field in predictor.FLOOD_FIELDS:
        assert field in out["flood"]
    # flood values come from the division table, never from caller input
    assert out["flood"]["water_occurrence_pct"] == \
        predictor.DIVISION_VALUES.loc["nugegoda", "water_occurrence_pct"]


def test_flood_record_is_official_and_descriptive():
    # Kelanimulla sits on the Kelani floodplain: inside both Survey Department maps
    rec = predictor.predict(division="kelanimulla", perches=10)["flood_record"]
    assert rec["indication"] == "High"
    assert rec["flooded_pct_2016"] > 50 and rec["flooded_pct_2025"] > 50
    assert len(rec["sources"]) == 4
    assert rec["flooded_pct_2018"] is not None
    # Nugegoda is inside the 2016 map's frame and was not flooded in it
    nug = predictor.flood_record("nugegoda")
    assert nug["flooded_pct_2016"] == 0
    # the record never changes the price
    a = predictor.predict(division="kelanimulla", perches=10)["price_per_perch"]
    saved, predictor.FLOOD_RECORDS = predictor.FLOOD_RECORDS, {}
    try:
        b = predictor.predict(division="kelanimulla", perches=10)
    finally:
        predictor.FLOOD_RECORDS = saved
    # the forest sums trees in parallel, so compare to float noise, not bits
    assert b["flood_record"] is None and abs(b["price_per_perch"] - a) < 1e-6 * a


def test_flood_record_follows_the_clicked_gn_not_the_fallback():
    # a point in an unmodelled GN is priced from its neighbour, but its flood
    # record must be its own polygon's
    import json
    feats = json.load(open(predictor.os.path.join(predictor.DEPLOY, "gn_boundaries.geojson"),
                           encoding="utf-8"))["features"]
    other = next(f["properties"] for f in feats
                 if not f["properties"]["modelled"]
                 and f["properties"]["address"] in predictor.FLOOD_RECORDS)
    out = predictor.predict(perches=10, lat=other["gn_lat"], lon=other["gn_lon"])
    if out.get("located", {}).get("gn_address") == other["address"]:
        assert out["flood_record"] == predictor.flood_record(other["address"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"{fn.__name__} ...")
            fn()
            print("  ok")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
