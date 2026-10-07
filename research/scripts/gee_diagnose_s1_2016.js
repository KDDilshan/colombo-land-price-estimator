// ===================================================================
// DIAGNOSTIC - why did the 2016 Roanu event return 0 scenes?
//
// Paste into https://code.earthengine.google.com and click Run.
// Nothing is exported. Read the Console panel top to bottom.
//
// The earlier "0 scenes" was counted on a collection that ALREADY had
// IW + VV + DESCENDING applied, so it could not distinguish "no data" from
// "no data matching my filters". This adds the filters back one at a time.
// ===================================================================

// Colombo district bounds, from data/processed/gn_centroids.csv
var AOI = ee.Geometry.Rectangle([79.80, 6.70, 80.25, 7.05]);
Map.centerObject(AOI, 10);
Map.addLayer(AOI, {color: 'yellow'}, 'Colombo district bounds');

var RAW = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(AOI);

function dates(coll) {
  return coll.aggregate_array('system:time_start')
             .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
             .distinct().sort();
}

// ---- 1. all of 2016, NO mode / polarisation / pass filter ----------
var y2016 = RAW.filterDate('2016-01-01', '2017-01-01');
print('=== 1. ALL Sentinel-1 GRD over Colombo, 2016, no filters ===');
print('scene count:', y2016.size());
print('orbitProperties_pass:', y2016.aggregate_histogram('orbitProperties_pass'));
print('instrumentMode:', y2016.aggregate_histogram('instrumentMode'));
print('platform:', y2016.aggregate_histogram('platform_number'));
print('resolution_meters:', y2016.aggregate_histogram('resolution_meters'));
print('acquisition dates 2016:', dates(y2016));

// ---- 2. which filter removes what, cumulatively -------------------
var f_iw = y2016.filter(ee.Filter.eq('instrumentMode', 'IW'));
var f_vv = f_iw.filter(
  ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'));
var f_desc = f_vv.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
var f_asc = f_vv.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));

print('=== 2. cumulative filter effect, 2016 ===');
print('  all              :', y2016.size());
print('  + IW             :', f_iw.size());
print('  + VV             :', f_vv.size());
print('  + DESCENDING     :', f_desc.size());
print('  + ASCENDING      :', f_asc.size());
print('  DESCENDING dates :', dates(f_desc));
print('  ASCENDING dates  :', dates(f_asc));

// ---- 3. the two windows that failed -------------------------------
function window_report(label, start, end) {
  var w = RAW.filterDate(start, end);
  var wIwVv = w.filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains(
                 'transmitterReceiverPolarisation', 'VV'));
  print('--- ' + label + '  ' + start + ' .. ' + end + ' ---');
  print('   any scene      :', w.size());
  print('   IW + VV        :', wIwVv.size());
  print('   pass histogram :', wIwVv.aggregate_histogram('orbitProperties_pass'));
  print('   dates          :', dates(wIwVv));
}

print('=== 3. the failed Roanu windows ===');
window_report('Roanu baseline', '2016-01-15', '2016-03-15');
window_report('Roanu flood   ', '2016-05-17', '2016-05-25');

// Wider look ONLY to see what exists nearby - this is diagnosis, not a
// licence to widen the flood window. A scene after the water receded
// measures nothing.
window_report('around Roanu  ', '2016-04-15', '2016-07-15');

// ---- 4. the three events that did work, for comparison ------------
print('=== 4. the other three events ===');
window_report('2021 flood    ', '2021-06-04', '2021-06-11');
window_report('2024 flood    ', '2024-10-11', '2024-10-18');
window_report('Ditwah flood  ', '2025-11-29', '2025-12-06');

// ---- 5. year by year, so the coverage gap is visible ---------------
print('=== 5. IW+VV scene count per year over Colombo ===');
[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
  .forEach(function (y) {
    var c = RAW.filterDate(y + '-01-01', (y + 1) + '-01-01')
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains(
                 'transmitterReceiverPolarisation', 'VV'));
    print('  ' + y + ':', c.size(),
          c.aggregate_histogram('orbitProperties_pass'));
  });
