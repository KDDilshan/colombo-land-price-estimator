"""Step 6i - generate the ATTEMPT B Earth Engine Code Editor script:
Sentinel-1 SAR change detection for the verified Colombo flood events.

Attempt A (JRC water recurrence, step6h) failed its geography test: the Kelani
floodplain did rank above the dry uplands, but the coast still dominated, so
the permanent-water mask did not do its job. Recorded, not tuned.

WHY SAR.  Sentinel-1 is radar: 10 m, and it sees through monsoon cloud - which
is what defeated the optical archives behind the rejected MODIS column.

PASS SELECTION IS AUTOMATIC, PER EVENT
  The first run hard-coded DESCENDING and lost the 2016 event. The diagnostic
  showed 2016 has 36 IW+VV scenes over Colombo and DESCENDING leaves 20, with
  Roanu's own windows sitting on the other pass. So the script no longer
  assumes: for each event it evaluates BOTH passes, keeps only passes with
  scenes in BOTH the baseline and the flood window - a baseline on one pass and
  a flood image on the other is not a comparison - and among the qualifying
  passes takes the one whose flood acquisition lands closest to the documented
  peak, preferring on-or-after the peak over before it. Every option is printed
  with its counts, dates and lag, so the choice is auditable rather than
  asserted. An event with no qualifying pass is skipped and named.

  Consistency rule: one pass within an event, different passes between events.

FLOOD WINDOWS ARE NEVER WIDENED. A scene from after the water receded measures
nothing. Losing an event beats inventing one.

KNOWN BIAS, RECORDED NOT FIXED (NOTES.md 6.12b)
  Each flood window resolves to a single acquisition DATE - the multiple
  "scenes" are frames of one pass - and that date falls ~2-3 days after peak.
  The variable therefore measures water STILL STANDING about two days after the
  peak, not peak inundation, which biases it toward slow-draining low ground.
  That is arguably the more relevant quantity for land value, but it is not
  what the variable name suggests, so it is written down.

THRESHOLDS - fixed before any result is seen, not adjusted afterwards:
  DROP_DB  = -3     at least 3 dB below the pixel's own baseline
  WATER_DB = -15    and dark enough in absolute terms to be open water

MASKS:
  JRC occurrence > 80%   permanent water: sea, river, lake
  MERIT HAND > 15 m      high ground, where radar shadow imitates water

Writes: scripts/gee_code_editor_sar_flood.js
"""

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
OUT = os.path.join(ROOT, "scripts", "gee_code_editor_sar_flood.js")

HEADER = """// ===================================================================
// ATTEMPT B - Sentinel-1 SAR flood frequency, 194 Colombo GN divisions
//
// HOW TO RUN
//   1. https://code.earthengine.google.com
//   2. Paste this whole file, click Run
//   3. READ THE CONSOLE FIRST. For every event it prints both orbit passes
//      with their baseline count, flood count, flood acquisition dates and
//      the lag from the documented peak, then the pass it chose and why.
//   4. Only then: Tasks tab -> RUN on "gee_sar_flood"
//   5. Download from Drive/colombo_replication to
//      data/processed/gee_sar_flood.csv
//   6. python src/step6j_validate_sar.py
//
// PASS SELECTION IS AUTOMATIC. A pass qualifies only if it has scenes in BOTH
// the baseline and the flood window - a baseline on one pass compared against
// a flood image on the other is not a comparison. Among qualifying passes the
// script takes the flood acquisition closest to the documented peak, and
// prefers on-or-after the peak over before it. Force a pass with PASS_OVERRIDE
// below if you need to.
//
// FLOOD WINDOWS ARE NEVER WIDENED. A scene taken after the water receded
// measures nothing.
// ===================================================================

var BUFFER_M   = 1000;
var DROP_DB    = -3;      // fall below the pixel's own baseline
var WATER_DB   = -15;     // absolute darkness of open water
var PERM_OCC   = 80;      // JRC occurrence % above which water is permanent
var HAND_MAX_M = 15;      // above this, dark pixels are radar shadow

var PASSES = ['DESCENDING', 'ASCENDING'];

// force a pass for an event, e.g. {roanu_2016: 'ASCENDING'}; normally empty
var PASS_OVERRIDE = {};

// [label, baselineStart, baselineEnd, floodStart, floodEnd, documentedPeak]
//   peak dates, from the sources in NOTES.md 6.12:
//   roanu_2016   350 mm on the Kelani basin 15-17 May; RISAT-1 mapped the
//                flood extent on 19 May -> peak taken as 18 May
//   june_2021    125 mm on the upper catchment 5 June, red alert -> 6 June
//   oct_2024     Kelani at flood level in Colombo district on 12 October
//   ditwah_2025  Kelani above MAJOR flood level at Hanwella and Mahawatta
//                on 30 November
var EVENTS = [
  ['roanu_2016',  '2016-01-15', '2016-03-15', '2016-05-17', '2016-05-25', '2016-05-18'],
  ['june_2021',   '2021-01-15', '2021-03-15', '2021-06-04', '2021-06-11', '2021-06-06'],
  ['oct_2024',    '2024-01-15', '2024-03-15', '2024-10-11', '2024-10-18', '2024-10-12'],
  ['ditwah_2025', '2025-01-15', '2025-03-15', '2025-11-29', '2025-12-06', '2025-11-30']
];

// [Address, latitude, longitude] - from data/processed/gn_centroids.csv
var PTS = [
"""

FOOTER = """];

// ---------------------------------------------------------------- geometry
var fc = ee.FeatureCollection(PTS.map(function (p) {
  return ee.Feature(ee.Geometry.Point([p[2], p[1]]).buffer(BUFFER_M),
                    {Address: p[0]});
}));
var aoi = fc.geometry().bounds();

// ------------------------------------------------------------------ masks
var occurrence = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
                   .select('occurrence').unmask(0);
var hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd').unmask(0);
var eligible = occurrence.lte(PERM_OCC).and(hand.lte(HAND_MAX_M));

// ------------------------------------------------------------- Sentinel-1
var S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filterBounds(aoi)
  .select('VV');

// speckle is what makes single SAR pixels unreliable; a 50 m focal median is
// the standard suppression, applied to both sides of the comparison
function composite(coll) {
  return coll.median().focal_median(50, 'circle', 'meters');
}

function distinctDates(coll) {
  return coll.aggregate_array('system:time_start')
             .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
             .distinct().sort().getInfo();
}

function dayDiff(a, b) {
  return Math.round((Date.parse(a) - Date.parse(b)) / 86400000);
}

// ------------------------------------------- evaluate both passes per event
function evaluate(ev, pass) {
  var coll = S1.filter(ee.Filter.eq('orbitProperties_pass', pass));
  var base = coll.filterDate(ev[1], ev[2]);
  var flow = coll.filterDate(ev[3], ev[4]);
  var nBase = base.size().getInfo();
  var nFlow = flow.size().getInfo();
  var fDates = nFlow > 0 ? distinctDates(flow) : [];
  var bDates = nBase > 0 ? distinctDates(base) : [];

  // lag of each flood acquisition from the documented peak, in days
  var lags = fDates.map(function (d) { return dayDiff(d, ev[5]); });
  // prefer on-or-after the peak; a scene before it may predate the water
  var after = lags.filter(function (l) { return l >= 0; });
  var best = after.length > 0
    ? Math.min.apply(null, after)
    : (lags.length > 0 ? Math.max.apply(null, lags) : null);

  return {pass: pass, nBase: nBase, nFlow: nFlow, fDates: fDates,
          bDates: bDates, lags: lags, bestLag: best,
          qualifies: nBase > 0 && nFlow > 0,
          bestIsAfterPeak: after.length > 0};
}

var bands = [], used = [], skipped = [], summary = [];

print('=== PASS SELECTION AND SCENE COUNTS - READ BEFORE EXPORTING ===');

EVENTS.forEach(function (ev) {
  var label = ev[0];
  print('--------------------------------------------------');
  print(label + '   flood window ' + ev[3] + ' .. ' + ev[4] +
        '   documented peak ' + ev[5]);

  var options = PASSES.map(function (p) { return evaluate(ev, p); });

  options.forEach(function (o) {
    print('   ' + o.pass +
          '   baseline: ' + o.nBase + ' scenes on ' + (o.bDates.length || 0) +
          ' date(s)   flood: ' + o.nFlow + ' scenes on ' +
          (o.fDates.length || 0) + ' date(s)' +
          (o.qualifies ? '' : '   <- does not qualify'));
    if (o.fDates.length) {
      print('        flood dates: ' + o.fDates.join(', ') +
            '   lag from peak (days): ' + o.lags.join(', '));
    }
    if (o.bDates.length) {
      print('        baseline dates: ' + o.bDates.join(', '));
    }
  });

  var qual = options.filter(function (o) { return o.qualifies; });
  var forced = PASS_OVERRIDE[label];
  var chosen = null;

  if (forced) {
    chosen = qual.filter(function (o) { return o.pass === forced; })[0] || null;
    print('   PASS_OVERRIDE requests ' + forced +
          (chosen ? '' : ' - but it does not qualify, ignoring the override'));
  }
  if (!chosen && qual.length > 0) {
    // closest flood acquisition to the peak; on-or-after beats before
    qual.sort(function (a, b) {
      if (a.bestIsAfterPeak !== b.bestIsAfterPeak) {
        return a.bestIsAfterPeak ? -1 : 1;
      }
      return Math.abs(a.bestLag) - Math.abs(b.bestLag);
    });
    chosen = qual[0];
  }

  if (!chosen) {
    var reason;
    var withBase = options.filter(function (o) { return o.nBase > 0; })
                          .map(function (o) { return o.pass; });
    var withFlow = options.filter(function (o) { return o.nFlow > 0; })
                          .map(function (o) { return o.pass; });
    if (withBase.length && withFlow.length) {
      reason = 'baseline exists only on ' + withBase.join('/') +
               ' but the flood window only on ' + withFlow.join('/') +
               ' - cross-pass comparison is invalid, so the event is dropped';
    } else if (!withFlow.length) {
      reason = 'no scene on any pass inside the flood window';
    } else {
      reason = 'no scene on any pass inside the baseline window';
    }
    print('   >>> SKIPPED: ' + reason);
    skipped.push(label + ': ' + reason);
    return;
  }

  print('   >>> CHOSEN: ' + chosen.pass +
        '   flood image ' + chosen.fDates.join('/') +
        '   lag ' + chosen.bestLag + ' day(s) from peak' +
        (chosen.bestIsAfterPeak ? '' : '   WARNING: before the peak'));
  if (chosen.fDates.length === 1) {
    print('        NOTE: one acquisition date - the ' + chosen.nFlow +
          ' scenes are frames of a single pass. This measures water still ' +
          'standing ' + chosen.bestLag + ' day(s) after peak, not peak ' +
          'inundation. Biased toward slow-draining low ground. NOTES.md 6.12b.');
  }

  var coll = S1.filter(ee.Filter.eq('orbitProperties_pass', chosen.pass));
  var baseline = composite(coll.filterDate(ev[1], ev[2]));
  var during   = composite(coll.filterDate(ev[3], ev[4]));

  bands.push(during.subtract(baseline)
                   .lt(DROP_DB)
                   .and(during.lt(WATER_DB))
                   .and(eligible)
                   .rename(label));
  used.push(label);
  summary.push(label + ': ' + chosen.pass + ', flood ' +
               chosen.fDates.join('/') + ', lag ' + chosen.bestLag + 'd');
});

print('==================================================');
print('events used:', summary.length ? summary : 'none');
print('events skipped:', skipped.length ? skipped : 'none');

if (bands.length === 0) {
  print('>>> NO USABLE EVENTS. Nothing to export. Run ' +
        'scripts/gee_diagnose_s1_2016.js to see what coverage exists.');
} else {
  var stack = ee.Image.cat(bands);

  var sampled = stack.reduceRegions({
    collection: fc,
    reducer: ee.Reducer.mean(),   // mean of a 0/1 mask = flooded fraction
    scale: 10,
    tileScale: 8
  });

  print('first 5 rows:', sampled.limit(5));

  Map.centerObject(fc, 11);
  Map.addLayer(bands[0].selfMask(), {palette: ['red']}, 'flooded ' + used[0]);
  if (bands.length > 1) {
    Map.addLayer(bands[bands.length - 1].selfMask(), {palette: ['blue']},
                 'flooded ' + used[used.length - 1], false);
  }
  Map.addLayer(fc, {color: 'yellow'}, '1 km buffers', false);

  Export.table.toDrive({
    collection: sampled,
    description: 'gee_sar_flood',
    folder: 'colombo_replication',
    fileNamePrefix: 'gee_sar_flood',
    fileFormat: 'CSV',
    selectors: ['Address'].concat(used)
  });
}
"""


def main():
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))
    lines = []
    for i, p in enumerate(pts):
        name = p["Address"].replace("\\", "").replace("'", "\\'")
        end = "," if i < len(pts) - 1 else ""
        lines.append(f"  ['{name}', {float(p['gn_lat']):.6f}, "
                     f"{float(p['gn_lon']):.6f}]{end}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEADER + "\n".join(lines) + "\n" + FOOTER)
    print(f"wrote {OUT}  ({len(pts)} centroids, "
          f"{os.path.getsize(OUT) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
