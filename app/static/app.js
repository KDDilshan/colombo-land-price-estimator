/* Colombo land price estimator - map interface.
 *
 * The map does two things: it turns a click into a point the server can resolve
 * to a GN division, and it turns a drawn polygon into an area in perches. It
 * never sets any model input other than location, size and land type.
 *
 * On top of the original tool this also drives the paid "nearby amenities"
 * panel and surfaces free-tier quota / upgrade messaging from the API.
 */

const COLOMBO = [6.9271, 79.8612];
const EARTH_RADIUS_M = 6371008.8;
const PERCH_M2 = window.PERCH_M2;
const DIVISION_COUNT = window.DIVISION_COUNT;
const CSRF_TOKEN = document.querySelector('meta[name="csrf-token"]')?.content;

const map = L.map('map').setView(COLOMBO, 11);
window.map = map;   // handy from the browser console when checking a location

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

/* ---------------------------------------------------------------- layers */

const drawnItems = new L.FeatureGroup().addTo(map);
let boundaryLayer = null;
let marker = null;
let amenityLayers = {};      // category key -> L.layerGroup, so each can be toggled
let amenityLegend = null;    // the Leaflet control listing categories with checkboxes

const drawControl = new L.Control.Draw({
  edit: { featureGroup: drawnItems, remove: true },
  draw: {
    polygon: { allowIntersection: false, showArea: false },
    rectangle: false,
    polyline: false,
    circle: false,
    circlemarker: false,
    marker: false
  }
});
map.addControl(drawControl);

/* ------------------------------------------------------------------ state */

const state = { lat: null, lon: null, source: null };

/* Leaflet still fires map 'click' for each vertex while leaflet-draw is running,
 * which would drop a marker and clear the drawing. Gate the click handler on it. */
let drawing = false;

const $ = (id) => document.getElementById(id);

/* --------------------------------------------------------------- geometry */

/* Geodesic area by spherical excess. Each edge contributes its true spherical
 * wedge, so the result is correct at Colombo's latitude - a naive lat/lon area
 * would be wrong by roughly the cos(latitude) factor. Mirrors
 * predictor.polygon_area_m2 on the server. */
function polygonAreaM2(latlngs) {
  if (latlngs.length < 3) return 0;
  const pts = latlngs.slice();
  pts.push(pts[0]);

  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const l1 = pts[i].lng * Math.PI / 180;
    const l2 = pts[i + 1].lng * Math.PI / 180;
    const p1 = pts[i].lat * Math.PI / 180;
    const p2 = pts[i + 1].lat * Math.PI / 180;
    total += (l2 - l1) * (2 + Math.sin(p1) + Math.sin(p2));
  }
  return Math.abs(total * EARTH_RADIUS_M * EARTH_RADIUS_M / 2);
}

function centroidOf(latlngs) {
  let lat = 0, lng = 0;
  latlngs.forEach((p) => { lat += p.lat; lng += p.lng; });
  return [lat / latlngs.length, lng / latlngs.length];
}

/* ------------------------------------------------------------ interaction */

function setPoint(lat, lon, source) {
  state.lat = lat;
  state.lon = lon;
  state.source = source;

  if (marker) map.removeLayer(marker);
  marker = L.marker([lat, lon]).addTo(map);

  $('division').value = '';
  $('division-search').value = '';
  $('division-sub').textContent = '';
  if (source !== 'ratnapura-point') $('ratnapura-point').value = '';
  $('location-hint').textContent =
    `Point set at ${lat.toFixed(5)}, ${lon.toFixed(5)} — the division is resolved on the server.`;
}

map.on(L.Draw.Event.DRAWSTART, () => { drawing = true; });
map.on(L.Draw.Event.DRAWSTOP, () => { drawing = false; });

map.on('click', (e) => {
  if (drawing) return;
  drawnItems.clearLayers();
  $('perches-sub').textContent = `1 perch = ${PERCH_M2} m²`;
  setPoint(e.latlng.lat, e.latlng.lng, 'click');
});

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);

  const ring = e.layer.getLatLngs()[0];
  const area = polygonAreaM2(ring);
  const perches = area / PERCH_M2;

  $('perches').value = perches.toFixed(2);
  $('perches-sub').textContent =
    `${area.toFixed(1)} m² from the drawn polygon ÷ ${PERCH_M2} m² per perch.`;

  const [lat, lon] = centroidOf(ring);
  setPoint(lat, lon, 'polygon');
  $('location-hint').textContent =
    'Plot drawn — area filled in below, division taken from the plot centre.';
});

map.on(L.Draw.Event.DELETED, () => {
  $('perches-sub').textContent = `1 perch = ${PERCH_M2} m²`;
});

/* Choosing a division by hand overrides the clicked point. */
$('division').addEventListener('change', () => {
  if (!$('division').value) return;
  state.lat = null;
  state.lon = null;
  state.source = 'division';
  if (marker) { map.removeLayer(marker); marker = null; }
  $('location-hint').textContent = 'Division chosen from the list.';
});

/* Searchable GN division box: typing filters the suggestion list; once the text
 * matches a division exactly, it is copied into the hidden select, which the
 * rest of the estimator reads. */
const DIVISION_NAMES = Array.from($('division').options).map((o) => o.value).filter(Boolean);
$('division-search').addEventListener('input', () => {
  const typed = $('division-search').value.trim().toLowerCase();
  const match = DIVISION_NAMES.find((d) => d.toLowerCase() === typed);
  if (match && $('division').value !== match) {
    $('division').value = match;
    $('division').dispatchEvent(new Event('change'));
  } else if (!match && $('division').value) {
    $('division').value = '';
  }
});

/* Ratnapura has no GN divisions to pick from, but it does have these 10 real
 * reference points - picking one sets a point exactly like clicking the map
 * would, since each already carries its own lat/lon. */
$('ratnapura-point').addEventListener('change', () => {
  const idx = $('ratnapura-point').value;
  if (idx === '') return;
  const point = (window.RATNAPURA_POINTS || [])[Number(idx)];
  if (!point) return;
  drawnItems.clearLayers();
  setPoint(point.lat, point.lon, 'ratnapura-point');
  map.panTo([point.lat, point.lon]);
  $('location-hint').textContent = `Reference point chosen: ${point.label}.`;
});

function resetColomboPlot() {
  drawnItems.clearLayers();
  if (marker) { map.removeLayer(marker); marker = null; }
  if (boundaryLayer) { map.removeLayer(boundaryLayer); boundaryLayer = null; }
  clearAmenityLayers();
  state.lat = state.lon = state.source = null;
  $('division').value = '';
  $('division-search').value = '';
  $('division-sub').textContent = '';
  $('ratnapura-point').value = '';
  $('results').hidden = true;
  $('compare-card').hidden = true;
  $('compare-results').innerHTML = '';
  lastEstimate = null;
  $('amenities-card').hidden = true;
  $('error').hidden = true;
  $('quota-note').hidden = true;
  document.querySelectorAll('#land-type-group input').forEach((box) => {
    box.checked = box.value === 'Residential';
  });
  $('perches-sub').textContent = `1 perch = ${PERCH_M2} m²`;
  $('location-hint').textContent =
    'Click a point on the map, or use the polygon tool to draw the plot.';
}

$('reset').addEventListener('click', () => {
  resetColomboPlot();
  map.setView(COLOMBO, 11);
});

/* -------------------------------------------------------------- district */

const RATNAPURA_CENTER = [6.72, 80.44];
let ratnapuraLayer = null;

function showRatnapuraPoints() {
  ratnapuraLayer = L.layerGroup().addTo(map);
  (window.RATNAPURA_POINTS || []).forEach((p) => {
    L.circleMarker([p.lat, p.lon], {
      radius: 6, color: '#94a3b8', fillColor: '#94a3b8', fillOpacity: 0.6, weight: 1
    }).bindTooltip(p.label).addTo(ratnapuraLayer);
  });
  map.setView(RATNAPURA_CENTER, 10);
}

function hideRatnapuraPoints() {
  if (ratnapuraLayer) { map.removeLayer(ratnapuraLayer); ratnapuraLayer = null; }
}

/* Ratnapura has no known GN divisions to pick from - only Colombo does - so
 * that one field is hidden there, but the map (click or draw), land size,
 * land type and submit all work exactly as they do for Colombo. The estimate
 * that comes back is always the same Colombo-wide fallback the model already
 * uses for any click outside Colombo district (see render()'s handling of
 * `unknown` below) - there is no real Ratnapura listing data behind it. */
$('district').addEventListener('change', () => {
  resetColomboPlot();
  if ($('district').value === 'ratnapura') {
    $('division-field').hidden = true;
    $('ratnapura-field').hidden = false;
    showRatnapuraPoints();
  } else {
    $('division-field').hidden = false;
    $('ratnapura-field').hidden = true;
    hideRatnapuraPoints();
    map.setView(COLOMBO, 11);
  }
});

/* ------------------------------------------------------------- formatting */

const lkr = (v) => 'LKR ' + Math.round(v).toLocaleString('en-US');

/* Official past-flooding record for the clicked GN (Survey Department maps,
 * DMC DesInventar). Descriptive - it never enters the price. */
const FLOOD_BADGE = {
  'High':               ['#fde2e1', '#9b1c1c'],
  'Medium':             ['#fef3c7', '#92400e'],
  'Low':                ['#dcfce7', '#166534'],
  'No official record': ['#e5e7eb', '#374151'],
};

function renderFloodRecord(rec, gnName) {
  $('flood-record-wrap').hidden = !rec;
  if (!rec) return;
  const [bg, fg] = FLOOD_BADGE[rec.indication] || FLOOD_BADGE['No official record'];
  const badge = $('fr-badge');
  badge.textContent = rec.indication;
  badge.style.background = bg;
  badge.style.color = fg;
  $('fr-gn').textContent = gnName ? `GN division: ${gnName}` : '';

  const mapCell = (pct) => pct === null || pct === undefined
    ? 'outside the mapped area'
    : pct > 0 ? `${pct < 1 ? '<1' : Math.round(pct)}% of the GN inside the flood extent`
              : 'mapped - not flooded';
  $('fr-2016').textContent = mapCell(rec.flooded_pct_2016);
  $('fr-2018').textContent = mapCell(rec.flooded_pct_2018);
  $('fr-2025').textContent = mapCell(rec.flooded_pct_2025);
  $('fr-dmc').textContent = rec.dmc_years.length
    ? `flooding recorded in ${rec.dmc_years.join(', ')}`
    : 'none naming this GN';
  $('fr-ds').textContent = rec.dmc_ds_flood_records
    ? `${rec.dmc_ds_flood_records} (${rec.ds_division})` : '0';
}

function num(v, digits, suffix) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits) + (suffix || '');
}

/* ------------------------------------------------------------- land type */

/* The model was trained with land type as one category per plot, not
 * independently combinable flags - "Commercial + Residential" is its own
 * trained class, not a blend of two. This canonical order matches how the
 * combined category names were built (see predictor.py / LAND_TYPE_VALUES),
 * so joining whichever base types are checked, in this order, reproduces the
 * exact trained category name. */
const LAND_TYPE_ORDER = ['Agricultural', 'Commercial', 'Residential', 'Other'];
const LAND_TYPE_VALUES = new Set(window.LAND_TYPE_VALUES || []);

/* Returns { key } on a valid combination, or { error } otherwise - never both. */
function resolveLandType() {
  const checked = LAND_TYPE_ORDER.filter((t) => {
    const box = document.querySelector(`#land-type-group input[value="${t}"]`);
    return box && box.checked;
  });
  if (!checked.length) {
    return { error: 'Pick at least one land type.' };
  }
  const key = 'Land_type_' + checked.join('_');
  if (!LAND_TYPE_VALUES.has(key)) {
    return { error: `${checked.join(' + ')} isn't a combination the model was trained on - try a different mix.` };
  }
  return { key };
}

/* ---------------------------------------------------------------- request */

async function submit() {
  const perches = parseFloat($('perches').value);
  const division = $('division').value;

  $('error').hidden = true;
  $('quota-note').hidden = true;

  if (!(perches > 0)) {
    showError('Enter a land size greater than zero.');
    return;
  }
  if (!division && state.lat === null) {
    showError('Click a point on the map, draw a plot, or pick a division.');
    return;
  }

  const landType = resolveLandType();
  if (landType.error) {
    showError(landType.error);
    return;
  }

  const body = {
    perches: perches,
    land_type: landType.key
  };
  if (division) {
    body.division = division;
  } else {
    body.lat = state.lat;
    body.lon = state.lon;
  }

  $('submit').disabled = true;
  $('submit').textContent = 'Estimating…';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF_TOKEN },
      body: JSON.stringify(body)
    });
    const data = await res.json();

    if (res.status === 402) {
      showQuotaNotice(data);
      return;
    }
    if (!res.ok) {
      showError(data.error || 'Prediction failed.');
      return;
    }
    render(data);
  } catch (err) {
    showError('Could not reach the server.');
  } finally {
    $('submit').disabled = false;
    $('submit').textContent = 'Estimate price';
  }
}

$('submit').addEventListener('click', submit);

function showError(msg) {
  $('error').textContent = msg;
  $('error').hidden = false;
  $('results').hidden = true;
  $('amenities-card').hidden = true;
}

function showQuotaNotice(data) {
  $('results').hidden = true;
  $('amenities-card').hidden = true;
  const note = $('quota-note');
  note.innerHTML = (data.message || 'Free plan limit reached.') +
    (data.upgrade_url ? ` <a href="${data.upgrade_url}">Upgrade to Pro</a>.` : '');
  note.hidden = false;
}

/* ----------------------------------------------------------------- render */

function render(data) {
  $('per-perch').textContent = lkr(data.price_per_perch);
  $('total').textContent = lkr(data.total_price);

  const loc = data.located;
  const unknown = !data.known_division;

  /* Only name a division when its own values were actually used. For an unknown
   * location the estimate comes from Colombo-wide medians, so naming the nearest
   * division here would imply a precision the estimate does not have. */
  const plot = `${data.perches} perches of ${data.land_type_label.toLowerCase()} land`;
  $('basis').textContent = unknown
    ? `${plot}, priced from Colombo-wide average values.`
    : `${plot} in ${data.division}.`;

  $('small-plot-warning').hidden = !data.small_plot;

  $('unknown-warning').hidden = !unknown;
  if (unknown) {
    let detail = '';
    /* A point outside Colombo district entirely (adm4_name null, e.g. Ratnapura)
     * gets a much stronger warning than a point inside Colombo but just outside
     * the modelled divisions - the former has literally no real listing data
     * anywhere near it, the latter is at least geographically in-domain. */
    let severe = false;
    if (loc && loc.modelled_reason) {
      /* This division was modelled in an earlier run and got dropped by a later
       * retrain (e.g. outlier trimming) - a materially different situation from
       * never having been in the training data at all. */
      detail = `${loc.adm4_name} was modelled in an earlier version of this tool ` +
                `but is not one of the current ${DIVISION_COUNT} divisions ` +
                `(${loc.modelled_reason}). `;
    } else if (loc && loc.adm4_name) {
      detail = `The point falls in ${loc.adm4_name}, which is not one of the ${DIVISION_COUNT} divisions in the training data. `;
    } else if (loc) {
      detail = 'This point is outside Colombo district entirely - there is no real ' +
                'land-listing data anywhere near it. The number above is a Colombo-wide ' +
                'average with no real connection to local land values here; treat it as ' +
                'illustrative only, not a genuine estimate. ';
      severe = true;
    }
    $('unknown-detail').textContent = detail;
    $('unknown-warning').classList.toggle('alert-error', severe);
    $('unknown-warning').classList.toggle('alert-warn', !severe);
  }

  /* Listing-count basis and confidence, shown only when a division's own values
   * were used - an unknown-division estimate already gets its own warning above. */
  $('listing-basis').hidden = unknown;
  $('low-confidence-warning').hidden = true;
  if (!unknown) {
    const n = data.listing_count;
    const plural = n === 1 ? '' : 's';
    $('listing-basis').textContent =
      `Based on ${n} listing${plural} in ${data.division} (confidence: ${data.confidence}).`;
    if (data.confidence === 'low' || data.confidence === 'medium') {
      $('low-confidence-warning').hidden = false;
      $('low-confidence-detail').textContent =
        `Only ${n} listing${plural} back this division's estimate - treat it with more ` +
        `caution than a division with many listings.`;
    }
  }

  $('flood-heading').textContent = unknown
    ? 'Environmental and spatial profile — Colombo-wide medians'
    : `Environmental and spatial profile for ${data.division}`;

  // occurrence runs well below 1% in the drier divisions, so 1 dp would read as zero
  $('f-water').textContent = num(data.flood.water_occurrence_pct, 2, '%');
  $('f-hand').textContent = num(data.flood.hand_m, 1, ' m');
  $('f-stream').textContent = num(data.flood.dist_to_stream_km, 2, ' km');
  $('f-ndvi').textContent = num(data.flood.ndvi, 2, '');

  renderFloodRecord(data.flood_record, loc ? loc.adm4_name : data.division);

  if (loc && loc.match === 'polygon') {
    $('division-sub').textContent = `Resolved from the map: ${loc.adm4_name}.`;
  } else if (loc && loc.nearest_division) {
    $('division-sub').textContent =
      `Nearest modelled division is ${loc.nearest_division}, ${loc.distance_km} km away.`;
  }

  if (boundaryLayer) { map.removeLayer(boundaryLayer); boundaryLayer = null; }
  if (data.boundary) {
    boundaryLayer = L.geoJSON(data.boundary, {
      style: { color: '#1d4ed8', weight: 2, fillOpacity: 0.08 }
    }).addTo(map);
    if (drawnItems.getLayers().length === 0) {
      map.fitBounds(boundaryLayer.getBounds(), { padding: [30, 30] });
    }
  }

  const pdfWrap = $('pdf-link-wrap');
  if (data.is_pro && data.estimate_id) {
    $('pdf-link').href = `/dashboard/estimate/${data.estimate_id}/pdf`;
    pdfWrap.hidden = false;
  } else {
    pdfWrap.hidden = true;
  }

  $('results').hidden = false;

  prepareCompare(data);
  loadAmenities(data);
}

/* ---------------------------------------------------------------- compare */

let lastEstimate = null;

function prepareCompare(data) {
  lastEstimate = data;
  $('compare-card').hidden = false;
  $('compare-results').innerHTML = '';
  $('compare-btn').textContent = 'Compare all land types for this plot';
  if (data.is_pro) {
    $('compare-btn').hidden = false;
    $('compare-teaser').hidden = true;
  } else {
    $('compare-btn').hidden = true;
    $('compare-teaser').hidden = false;
  }
}

$('compare-btn').addEventListener('click', runCompare);

async function runCompare() {
  if (!lastEstimate) return;

  $('compare-btn').disabled = true;
  $('compare-btn').textContent = 'Comparing…';
  $('compare-results').innerHTML = '<span class="skeleton"></span><span class="skeleton"></span><span class="skeleton"></span>';

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF_TOKEN },
      /* When known_division is false (e.g. a Ratnapura click, or any point
       * outside Colombo), `division` is still filled in with the nearest
       * modelled division as a location hint - but predictor.predict() did NOT
       * price off that division's real data, it used the Colombo-wide
       * fallback. Compare has to key off known_division, not division's
       * truthiness, or it would silently re-price using that nearby
       * division's real listings instead of matching the original estimate. */
      body: JSON.stringify({
        perches: lastEstimate.perches,
        division: lastEstimate.known_division ? lastEstimate.division : null,
        lat: lastEstimate.known_division ? undefined : lastEstimate.anchor?.lat,
        lon: lastEstimate.known_division ? undefined : lastEstimate.anchor?.lon
      })
    });
    const payload = await res.json();

    if (res.status === 402) {
      $('compare-btn').hidden = true;
      $('compare-teaser').hidden = false;
      return;
    }
    if (!res.ok) {
      $('compare-results').innerHTML = `<p class="amenity-empty">${payload.error || 'Could not compare.'}</p>`;
      return;
    }
    renderCompare(payload);
    $('compare-btn').hidden = true;
  } catch (err) {
    $('compare-results').innerHTML = '<p class="amenity-empty">Could not reach the server.</p>';
  } finally {
    $('compare-btn').disabled = false;
    $('compare-btn').textContent = 'Compare all land types for this plot';
  }
}

function renderCompare(payload) {
  const list = $('compare-results');
  list.innerHTML = '';
  const max = Math.max(...payload.results.map((r) => r.price_per_perch));

  payload.results.forEach((r) => {
    const row = document.createElement('div');
    row.className = 'compare-row';
    if (lastEstimate && r.land_type === lastEstimate.land_type) row.classList.add('compare-row-current');

    const label = document.createElement('span');
    label.className = 'compare-label';
    label.textContent = r.land_type_label;

    const barWrap = document.createElement('span');
    barWrap.className = 'compare-bar-wrap';
    const bar = document.createElement('span');
    bar.className = 'compare-bar';
    bar.style.width = `${((r.price_per_perch / max) * 100).toFixed(1)}%`;
    barWrap.appendChild(bar);

    const value = document.createElement('span');
    value.className = 'compare-value';
    value.textContent = lkr(r.price_per_perch);

    row.appendChild(label);
    row.appendChild(barWrap);
    row.appendChild(value);
    list.appendChild(row);
  });
}

/* -------------------------------------------------------------- amenities */

const AMENITY_ICONS = {
  school: '🎓', hospital: '🏥', supermarket: '🛒',
  bank: '🏦', bus_stop: '🚌', railway_station: '🚆', park: '🌳'
};

function amenityDivIcon(key) {
  return L.divIcon({
    className: 'amenity-marker',
    html: AMENITY_ICONS[key] || '📍',
    iconSize: [26, 26],
    iconAnchor: [13, 13]
  });
}

function clearAmenityLayers() {
  Object.values(amenityLayers).forEach((layer) => map.removeLayer(layer));
  amenityLayers = {};
  if (amenityLegend) { map.removeControl(amenityLegend); amenityLegend = null; }
}

async function loadAmenities(data) {
  const card = $('amenities-card');
  const proBox = $('amenities-pro');
  const teaser = $('amenities-teaser');

  clearAmenityLayers();

  const anchor = data.anchor;
  if (!anchor) { card.hidden = true; return; }

  if (data.is_pro === false) {
    card.hidden = false;
    proBox.hidden = true;
    teaser.hidden = false;
    return;
  }

  card.hidden = false;
  teaser.hidden = true;
  proBox.hidden = false;
  $('amenities-list').innerHTML = '<span class="skeleton"></span><span class="skeleton"></span><span class="skeleton"></span>';

  try {
    const res = await fetch('/api/amenities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF_TOKEN },
      body: JSON.stringify({ lat: anchor.lat, lon: anchor.lon })
    });
    const payload = await res.json();

    if (res.status === 402) {
      proBox.hidden = true;
      teaser.hidden = false;
      return;
    }
    if (!res.ok) {
      $('amenities-list').innerHTML = '<p class="amenity-empty">Could not load amenities.</p>';
      return;
    }

    renderAmenities(payload.categories, anchor);
  } catch (err) {
    $('amenities-list').innerHTML = '<p class="amenity-empty">Could not reach the server.</p>';
  }
}

function renderAmenities(categories, anchor) {
  const list = $('amenities-list');
  list.innerHTML = '';

  const legendRows = [];

  Object.entries(categories).forEach(([key, group]) => {
    const section = document.createElement('div');
    section.className = 'amenity-group';

    const heading = document.createElement('h4');
    heading.textContent = `${AMENITY_ICONS[key] || ''} ${group.label}`;
    section.appendChild(heading);

    const layer = L.layerGroup();
    amenityLayers[key] = layer;

    if (!group.items.length) {
      const empty = document.createElement('p');
      empty.className = 'amenity-empty';
      empty.textContent = 'None found within 1.5 km.';
      section.appendChild(empty);
    } else {
      const ul = document.createElement('ul');
      group.items.slice(0, 5).forEach((item) => {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.textContent = item.name;
        const dist = document.createElement('span');
        dist.className = 'dist';
        dist.textContent = item.distance_m < 1000
          ? `${item.distance_m} m`
          : `${(item.distance_m / 1000).toFixed(1)} km`;
        li.appendChild(name);
        li.appendChild(dist);
        li.addEventListener('click', () => map.panTo([item.lat, item.lon]));
        ul.appendChild(li);

        L.marker([item.lat, item.lon], { icon: amenityDivIcon(key) })
          .bindTooltip(`${item.name} (${dist.textContent})`)
          .addTo(layer);
      });
      section.appendChild(ul);
      layer.addTo(map);
      legendRows.push({ key, label: group.label, count: group.items.length });
    }

    list.appendChild(section);
  });

  if (legendRows.length) {
    amenityLegend = buildAmenityLegend(legendRows);
    amenityLegend.addTo(map);
  }
}

function buildAmenityLegend(rows) {
  const Legend = L.Control.extend({
    options: { position: 'bottomleft' },
    onAdd() {
      const box = L.DomUtil.create('div', 'amenity-legend');
      rows.forEach(({ key, label, count }) => {
        const row = document.createElement('label');
        row.className = 'amenity-legend-row';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', () => {
          const layer = amenityLayers[key];
          if (!layer) return;
          if (checkbox.checked) { layer.addTo(map); } else { map.removeLayer(layer); }
        });

        const icon = document.createElement('span');
        icon.textContent = AMENITY_ICONS[key] || '📍';

        const text = document.createElement('span');
        text.textContent = `${label} (${count})`;

        row.appendChild(checkbox);
        row.appendChild(icon);
        row.appendChild(text);
        box.appendChild(row);
      });
      L.DomEvent.disableClickPropagation(box);
      return box;
    }
  });
  return new Legend();
}
