// ===================================================================
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
