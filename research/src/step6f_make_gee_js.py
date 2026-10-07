"""Step 6f - generate a Google Earth Engine *Code Editor* script.

The Python client needs `earthengine authenticate`, which is an interactive
browser login. The Code Editor at code.earthengine.google.com is already
authenticated in the browser, so pasting a script there needs no install, no
token and no asset upload - provided the 194 centroids travel inside the script
itself. That is what this generates.

Output: scripts/gee_code_editor_hand_floodfreq.js
"""

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
OUT = os.path.join(ROOT, "scripts", "gee_code_editor_hand_floodfreq.js")

HEADER = """// ===================================================================
// Colombo land-price replication - hazard extension
// HAND and flood frequency for the 194 GN division centroids
//
// HOW TO RUN
//   1. Open https://code.earthengine.google.com
//   2. Paste this whole file into the editor
//   3. Click Run
//   4. Open the Tasks tab (right-hand panel), click RUN on
//      "gee_hand_floodfreq", confirm
//   5. The CSV lands in your Google Drive under the folder
//      "colombo_replication". Download it to
//      data/processed/gee_hand_floodfreq.csv
//   6. Back in the project:  python src/step6e_screen_and_join.py
//
// WHAT IT MEASURES
//   hand_m           MERIT Hydro 'hnd', height above nearest drainage, ~90 m.
//                    Metres a point sits above the stream that drains it - the
//                    single best flood-proneness predictor available free.
//   flood_frequency  Global Flood Database v1: each image is one mapped flood
//                    event, the 'flooded' band is 1 where it inundated the
//                    pixel. Summed over 2000-2018 = number of observed floods.
//
// GEOMETRY
//   Mean over a 1 km circular buffer around each centroid - the same geometry
//   as every other hazard variable in this dataset. Never a single pixel.
//
// NULLS
//   flood_frequency is unmasked to 0: absent from the flood archive means
//   never observed flooded, which is a real zero.
//   hand_m is left MASKED. Where MERIT has no value (open water, coastline)
//   the cell exports empty, so a gap shows up as a gap rather than as a
//   confident 0 m above drainage. Do not fill these with zeros.
// ===================================================================

var BUFFER_M = 1000;

// [Address, latitude, longitude] - generated from data/processed/gn_centroids.csv
var PTS = [
"""

FOOTER = """];

// ---------------------------------------------------------------- features
var fc = ee.FeatureCollection(PTS.map(function (p) {
  return ee.Feature(
    ee.Geometry.Point([p[2], p[1]]).buffer(BUFFER_M),
    {Address: p[0]}
  );
}));

// ------------------------------------------------------------------ images
var hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd').rename('hand_m');

var floodFreq = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1')
  .select('flooded')
  .sum()
  .unmask(0)
  .rename('flood_frequency');

var stack = hand.addBands(floodFreq);

// ------------------------------------------------------------------ sample
var sampled = stack.reduceRegions({
  collection: fc,
  reducer: ee.Reducer.mean(),
  scale: 90,          // MERIT's native ~90 m; the 250 m flood layer is averaged
  tileScale: 4        // raise to 8 or 16 if the task fails on memory
});

// ------------------------------------------------------------- quick look
print('features sampled:', sampled.size());
print('first 5 rows:', sampled.limit(5));

var hv = sampled.aggregate_array('hand_m');
var ff = sampled.aggregate_array('flood_frequency');
print('hand_m  min / mean / max:',
      hv.reduce(ee.Reducer.min()), hv.reduce(ee.Reducer.mean()),
      hv.reduce(ee.Reducer.max()));
print('hand_m  non-null count (194 expected):', hv.length());
print('flood_frequency  min / mean / max:',
      ff.reduce(ee.Reducer.min()), ff.reduce(ee.Reducer.mean()),
      ff.reduce(ee.Reducer.max()));
print('flood_frequency  distinct values:',
      ee.List(ff).distinct().sort());

// see the buffers on the map, as a sanity check that they sit over Colombo
Map.centerObject(fc, 10);
Map.addLayer(hand, {min: 0, max: 40, palette: ['blue', 'white', 'brown']}, 'HAND (m)');
Map.addLayer(fc, {color: 'red'}, '1 km buffers');

// ------------------------------------------------------------------ export
Export.table.toDrive({
  collection: sampled,
  description: 'gee_hand_floodfreq',
  folder: 'colombo_replication',
  fileNamePrefix: 'gee_hand_floodfreq',
  fileFormat: 'CSV',
  selectors: ['Address', 'hand_m', 'flood_frequency']
});

// ===================================================================
// OPTIONAL cross-check: the five layers already extracted without Earth
// Engine, so the two methods can be compared. Uncomment, Run, and a second
// task appears. Column names match cog_hazard_194.csv.
// ===================================================================
// var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic();
// var check = dem.rename('elevation_m')
//   .addBands(ee.Terrain.slope(dem.setDefaultProjection('EPSG:4326', null, 30))
//             .rename('slope_deg'))
//   .addBands(ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
//             .unmask(0).rename('water_occurrence_pct'))
//   .addBands(ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
//             .eq(50).rename('built_up_fraction'));
// Export.table.toDrive({
//   collection: check.reduceRegions({collection: fc,
//                                    reducer: ee.Reducer.mean(),
//                                    scale: 30, tileScale: 4}),
//   description: 'gee_crosscheck',
//   folder: 'colombo_replication',
//   fileNamePrefix: 'gee_crosscheck',
//   fileFormat: 'CSV',
//   selectors: ['Address', 'elevation_m', 'slope_deg',
//               'water_occurrence_pct', 'built_up_fraction']
// });
"""


def main():
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))

    lines = []
    for i, p in enumerate(pts):
        name = p["Address"].replace("\\", "").replace("'", "\\'")
        comma = "," if i < len(pts) - 1 else ""
        lines.append(f"  ['{name}', {float(p['gn_lat']):.6f}, "
                     f"{float(p['gn_lon']):.6f}]{comma}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEADER + "\n".join(lines) + "\n" + FOOTER)

    print(f"wrote {OUT}")
    print(f"  {len(pts)} centroids embedded, "
          f"{os.path.getsize(OUT) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
