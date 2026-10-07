// ===================================================================
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
  ['athurugiriya', 6.869298, 79.997290],
  ['homagama', 6.851505, 80.002128],
  ['kolamunna', 6.795273, 79.922606],
  ['nugegoda', 6.877149, 79.890369],
  ['battaramulla', 6.897827, 79.921915],
  ['thalawathugoda', 6.873058, 79.938004],
  ['kottawa', 6.850883, 79.963825],
  ['malabe', 6.906037, 79.967377],
  ['moratuwella', 6.771168, 79.885079],
  ['dehiwala', 6.857536, 79.865477],
  ['boralesgamuwa', 6.839342, 79.902325],
  ['pannipitiya', 6.847977, 79.945919],
  ['thunadahena', 6.903608, 79.987338],
  ['pitakotte', 6.888291, 79.906531],
  ['maharagama', 6.847244, 79.932103],
  ['kaduwela', 6.930546, 79.987041],
  ['kothalawala', 6.922262, 79.979872],
  ['rajagiriya', 6.906753, 79.896825],
  ['habarakada', 6.878025, 80.017682],
  ['nawala', 6.892625, 79.888939],
  ['halpita', 6.779366, 79.956228],
  ['meegoda', 6.860787, 80.053102],
  ['diyagama', 6.805962, 79.997382],
  ['mount lavinia', 6.839240, 79.865035],
  ['mattegoda', 6.814394, 79.969905],
  ['padukka', 6.838686, 80.090046],
  ['kahathuduwa', 6.783388, 79.993907],
  ['pitipana', 6.822699, 80.030547],
  ['panagoda', 6.858353, 80.021112],
  ['kiriwattuduwa', 6.796261, 80.005807],
  ['hokandara', 6.879535, 79.969031],
  ['avissawella', 6.954105, 80.210286],
  ['nawagamuwa', 6.913583, 80.016075],
  ['havelock town', 6.886026, 79.866288],
  ['rukmale', 6.856241, 79.979098],
  ['rathmalana', 6.811498, 79.875406],
  ['kalalgoda', 6.867500, 79.949642],
  ['kesbewa', 6.788849, 79.936055],
  ['thalahena', 6.913194, 79.952092],
  ['koswatta', 6.895701, 80.051280],
  ['bokundara', 6.817149, 79.920260],
  ['kohuwala', 6.863895, 79.887450],
  ['makumbura', 6.838693, 79.976142],
  ['depanama', 6.858506, 79.947756],
  ['godagama', 6.849116, 80.031893],
  ['kalubovila', 6.862987, 79.877531],
  ['rattanapitiya', 6.848953, 79.901427],
  ['mirihana', 6.878991, 79.907808],
  ['rathmaldeniya', 6.839464, 79.949940],
  ['watareka', 6.850216, 80.066865],
  ['shanthalokagama', 6.892657, 79.996848],
  ['pepiliyana', 6.857012, 79.888461],
  ['vihara', 6.819504, 79.869634],
  ['wattegedara', 6.847364, 79.918252],
  ['welivita', 6.936036, 79.955809],
  ['belagama', 6.935090, 79.922797],
  ['beruketiya', 6.800569, 80.051008],
  ['borella', 6.917253, 79.879778],
  ['mahalwarawa', 6.836531, 79.955889],
  ['uyana', 6.785686, 79.878759],
  ['malapalla', 6.847808, 79.982287],
  ['arangala', 6.886052, 79.962674],
  ['pothuarawa', 6.895796, 79.952530],
  ['hanwella', 6.904215, 80.084297],
  ['polwatta', 6.858956, 79.938969],
  ['madiwela', 6.877724, 79.923097],
  ['subhoothipura', 6.906223, 79.914144],
  ['kalu aggala', 6.930511, 80.109207],
  ['boralugoda', 6.873984, 79.980051],
  ['ethulkotte', 6.898754, 79.906347],
  ['wellampitiya', 6.939398, 79.894398],
  ['atigala', 6.895964, 80.060318],
  ['rilawala', 6.790639, 79.979044],
  ['neelammahara', 6.831064, 79.918788],
  ['wellawatta', 6.874556, 79.861785],
  ['wijerama', 6.853631, 79.908709],
  ['bellanvila', 6.847911, 79.888105],
  ['henpita', 6.904049, 80.051726],
  ['katuwawala', 6.828525, 79.913787],
  ['thalapathpitiya', 6.869894, 79.922315],
  ['thumbovila', 6.811720, 79.911853],
  ['weniwelkola', 6.770961, 79.985767],
  ['pore', 6.887305, 79.984516],
  ['maradana', 6.926081, 79.864187],
  ['udahamulla', 6.868281, 79.916107],
  ['attidiya', 6.830750, 79.884509],
  ['gothatuwa', 6.924925, 79.907462],
  ['makuludoowa', 6.813581, 79.933599],
  ['mattakkuliya', 6.972178, 79.876962],
  ['navinna', 6.855039, 79.914557],
  ['pahalawela', 6.884875, 79.927089],
  ['pamunuwa', 6.858010, 79.932248],
  ['piriwena', 6.825309, 79.873721],
  ['kirulapone', 6.880490, 79.877633],
  ['kudamaduwa', 6.801195, 79.958307],
  ['nedimala', 6.855425, 79.880484],
  ['pathiragoda', 6.856932, 79.920788],
  ['thaldiyawala', 6.871449, 79.988222],
  ['himbutana', 6.921805, 79.935369],
  ['katubedda', 6.800430, 79.896381],
  ['kolonnawa', 6.932882, 79.896880],
  ['magammana', 6.820246, 79.997575],
  ['narahenpita', 6.902588, 79.878252],
  ['oruwala', 6.883930, 80.001009],
  ['dematagoda', 6.935812, 79.879950],
  ['divulpitiya', 6.851063, 79.894370],
  ['egodawatta', 6.848857, 79.907936],
  ['kosgama', 6.944192, 80.139605],
  ['madapatha', 6.766690, 79.929199],
  ['pahathgama', 6.896185, 80.083634],
  ['siyambalagoda', 6.796665, 79.969334],
  ['thalangama', 6.908870, 79.938109],
  ['walawwatta', 6.901053, 80.071382],
  ['dutugemunu', 6.868053, 79.881694],
  ['galavilawatta', 6.840212, 79.993731],
  ['grandpass', 6.944006, 79.874586],
  ['jambugasmulla', 6.861819, 79.895667],
  ['katuwana', 6.834662, 80.002418],
  ['malwatta', 6.853281, 79.870692],
  ['mullegama', 6.878982, 80.011579],
  ['mulleriyawa', 6.937436, 79.929681],
  ['ranala', 6.916568, 80.033991],
  ['watarappala', 6.840023, 79.870777],
  ['ambathale', 6.941375, 79.938348],
  ['kalapaluwawa', 6.914251, 79.912725],
  ['kaldemulla', 6.806441, 79.879407],
  ['kawdana', 6.844532, 79.876874],
  ['kirigampamunuwa', 6.800689, 79.980463],
  ['kotahena', 6.949180, 79.861862],
  ['kotikawatta', 6.930625, 79.917244],
  ['siddamulla', 6.818377, 79.958582],
  ['sri saranankara', 6.868695, 79.868276],
  ['uduwana', 6.823770, 80.007645],
  ['borupana', 6.812892, 79.891829],
  ['dampe', 6.792585, 79.983597],
  ['fort', 6.944053, 79.835533],
  ['gangodavila', 6.863109, 79.903775],
  ['henawatta', 6.859342, 80.040409],
  ['honnanthara', 6.796873, 79.948581],
  ['idama', 6.780022, 79.886174],
  ['kahapola', 6.754090, 79.924853],
  ['kajugahawatta', 6.925972, 79.910742],
  ['koralawella', 6.758354, 79.889167],
  ['kumaragewatta', 6.889279, 79.946021],
  ['kurunduwatta', 6.893996, 79.909542],
  ['madulawa', 6.826948, 80.066287],
  ['maligagodella', 6.925692, 79.951295],
  ['sedawatta', 6.953925, 79.884183],
  ['slave island', 6.925370, 79.850397],
  ['suwarapola', 6.796214, 79.914119],
  ['udumulla', 6.867602, 80.026295],
  ['wennawatta', 6.942410, 79.904029],
  ['wewala', 6.800905, 79.909313],
  ['wickramasinghapura', 6.882379, 79.941746],
  ['angampitiya', 6.845999, 80.119028],
  ['asiri uyana', 6.895002, 79.928505],
  ['bambalapitiya', 6.902504, 79.854628],
  ['batakettara', 6.781910, 79.924476],
  ['bomiriya', 6.932123, 80.000775],
  ['bopetta', 6.932687, 79.904619],
  ['dedigamuwa', 6.897403, 80.026211],
  ['diddeniya', 6.885115, 80.107682],
  ['galwala', 6.861272, 79.868928],
  ['ganegoda', 6.844484, 80.109226],
  ['gorakapitiya', 6.814296, 79.948813],
  ['hewagama', 6.930477, 79.977056],
  ['jalthara', 6.901725, 80.042327],
  ['karagampitiya', 6.847960, 79.871602],
  ['kelanimulla', 6.945143, 79.920380],
  ['kotuvila', 6.949412, 79.891826],
  ['lakshapathiya', 6.797161, 79.882040],
  ['liyanagoda', 6.858737, 79.959157],
  ['madampitiya', 6.961778, 79.875497],
  ['mahadeniya', 6.925706, 79.961457],
  ['mahawatta', 6.957463, 79.874186],
  ['maligakanda', 6.928646, 79.867851],
  ['mawathgama', 6.833269, 80.013648],
  ['modara', 6.964661, 79.871736],
  ['molpe', 6.792152, 79.901040],
  ['nawalamulla', 6.871782, 80.043602],
  ['neluwattuduwa', 6.897988, 80.115822],
  ['pagoda', 6.877827, 79.898297],
  ['paligedara', 6.818973, 79.932759],
  ['salamulla', 6.926873, 79.896956],
  ['seethawaka', 6.954259, 80.217257],
  ['thelawala', 6.807437, 79.893757],
  ['thummodara', 6.864092, 80.169001],
  ['thunnana', 6.883229, 80.085407],
  ['udyanaya', 6.855652, 79.875600],
  ['walpola', 6.916917, 79.922696],
  ['wedikanda', 6.825114, 79.865463],
  ['welipillewa', 6.884017, 80.031506],
  ['weragala', 6.828735, 80.120209],
  ['werahera', 6.825283, 79.904485]
];

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
