/* SolarSoiled Homeowner Dashboard — dashboard.js
 *
 * Responsibilities:
 *  1. Initialize Leaflet map and load array GeoJSON (from backend or embedded data)
 *  2. Handle QR code URL param ?id=<array_id> — focus + highlight the array
 *  3. "Your Array" detail panel — populate on map click
 *  4. Recalculate button — calls /recommend-quick on the backend
 *  5. Energy/money calculator — pure client-side
 */

// ── config ────────────────────────────────────────────────────────────────────

const API_BASE   = 'https://solarsoiled-api.onrender.com';
const API_KEY    = '';                               // set if your backend requires one
const PARTNER_ID = 'santa-cruz-outreach-v1';
const ARRAYS_URL = `${API_BASE}/results/${PARTNER_ID}/arrays`;
const RECALC_URL = `${API_BASE}/recommend-quick`;

// Santa Cruz center
const SC_CENTER = [36.974, -122.030];
const SC_ZOOM   = 13;

// Color mapping (mirrors viz.py palette)
const RISK_COLORS = [
  { threshold: 0.65, color: '#ef4444', label: 'High',   cls: 'badge-high'   },
  { threshold: 0.40, color: '#eab308', label: 'Medium', cls: 'badge-medium' },
  { threshold: 0.00, color: '#22c55e', label: 'Low',    cls: 'badge-low'    },
];

function riskColor(score) {
  for (const { threshold, color } of RISK_COLORS) {
    if (score >= threshold) return color;
  }
  return '#94a3b8';
}
function riskLabel(score) {
  return RISK_COLORS.find(r => score >= r.threshold)?.label ?? 'Unknown';
}
function riskBadgeCls(score) {
  return RISK_COLORS.find(r => score >= r.threshold)?.cls ?? '';
}

// Soiling rate estimate from risk score (linear rough mapping)
function soilingPct(score) {
  const pct = score * 15;
  return { lo: Math.max(1, (pct * 0.6).toFixed(1)), hi: (pct * 1.4).toFixed(1) };
}

// ── state ─────────────────────────────────────────────────────────────────────
let _map, _geojsonLayer, _features = [], _selectedLayer = null, _selectedId = null;

// ── init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadArrays();
  initCalculator();
  handleQrParam();
});

function initMap() {
  _map = L.map('map', { zoomControl: true }).setView(SC_CENTER, SC_ZOOM);

  // Basemap: ESRI satellite
  L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Tiles © Esri', maxZoom: 20 }
  ).addTo(_map);

  // Legend
  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = () => {
    const div = L.DomUtil.create('div', 'map-legend');
    div.innerHTML = `
      <h4>Soiling Risk</h4>
      <div class="legend-row"><div class="legend-dot" style="background:#ef4444"></div> High (≥0.65)</div>
      <div class="legend-row"><div class="legend-dot" style="background:#eab308"></div> Medium (0.40–0.64)</div>
      <div class="legend-row"><div class="legend-dot" style="background:#22c55e"></div> Low (&lt;0.40)</div>
    `;
    return div;
  };
  legend.addTo(_map);
}

async function loadArrays() {
  setLoading(true);
  try {
    const headers = API_KEY ? { 'X-API-Key': API_KEY } : {};
    const res = await fetch(ARRAYS_URL, { headers });
    if (!res.ok) throw new Error(`${res.status}`);
    const fc = await res.json();
    _features = fc.features || [];
    renderArrays(fc);
    updateStats();
  } catch (err) {
    console.warn('Could not load arrays from backend:', err.message);
    // Use embedded fallback data if backend unreachable
    if (window.FALLBACK_ARRAYS) {
      _features = window.FALLBACK_ARRAYS.features || [];
      renderArrays(window.FALLBACK_ARRAYS);
      updateStats();
    } else {
      document.getElementById('map-loading').innerHTML =
        '<p style="color:#fca5a5;font-size:.85rem;text-align:center;padding:1rem">' +
        'Map data unavailable. Backend may be starting up — try again in 30 seconds.</p>';
      return;
    }
  }
  setLoading(false);
}

function renderArrays(fc) {
  if (_geojsonLayer) _map.removeLayer(_geojsonLayer);

  _geojsonLayer = L.geoJSON(fc, {
    style: feat => {
      const score = feat.properties?.risk_score ?? 0;
      return {
        color:       '#1e293b',
        weight:      1,
        fillColor:   riskColor(score),
        fillOpacity: 0.65,
      };
    },
    onEachFeature: (feat, layer) => {
      layer.on('click', () => selectArray(feat, layer));
      const p = feat.properties || {};
      const score = p.risk_score ?? 0;
      layer.bindTooltip(
        `<b>Array #${p.array_id}</b><br>Risk: ${riskLabel(score)} (${score.toFixed(2)})<br>Area: ${p.area_m2} m²`,
        { sticky: true }
      );
    },
  }).addTo(_map);
}

function selectArray(feat, layer) {
  // Reset previous highlight
  if (_selectedLayer) {
    _geojsonLayer.resetStyle(_selectedLayer);
  }
  _selectedLayer = layer;
  _selectedId    = feat.properties?.array_id;

  layer.setStyle({ weight: 3, color: '#F7B731', fillOpacity: 0.8 });
  layer.bringToFront();

  populateDetail(feat);
  document.getElementById('array-detail').classList.add('visible');
  document.getElementById('detail-placeholder').style.display = 'none';
  document.getElementById('array-detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function populateDetail(feat) {
  const p   = feat.properties || {};
  const id  = p.array_id ?? '?';
  const score = p.risk_score ?? 0;
  const area  = p.area_m2  ?? '?';
  const sl    = soilingPct(score);

  document.getElementById('detail-id').textContent    = `Array #${id}`;
  document.getElementById('detail-badge').textContent  = riskLabel(score);
  document.getElementById('detail-badge').className    = `detail-badge ${riskBadgeCls(score)}`;
  document.getElementById('detail-score').textContent  = score.toFixed(2);
  document.getElementById('detail-area').textContent   = `${area} m²`;
  document.getElementById('detail-soiling').textContent= `${sl.lo}–${sl.hi}%`;

  const fill = document.querySelector('.risk-gauge-fill');
  fill.style.width      = `${Math.round(score * 100)}%`;
  fill.style.background = riskColor(score);

  // Pre-fill recalc threshold
  document.getElementById('threshold-val').value =
    document.getElementById('threshold-display').textContent = '0.60';

  // Auto-update calculator soiling rate from this array
  document.getElementById('calc-soiling').value = (score * 15).toFixed(1);
  calcUpdate();
}

// ── recalculate ───────────────────────────────────────────────────────────────
document.getElementById('btn-recalc')?.addEventListener('click', async () => {
  if (_selectedId === null) return;
  const lastCleaned = document.getElementById('last-cleaned').value;
  const threshold   = document.getElementById('threshold-val').value;
  if (!lastCleaned) { alert('Please select a last-cleaned date.'); return; }

  const btn = document.getElementById('btn-recalc');
  btn.disabled = true;
  btn.textContent = 'Recalculating…';

  try {
    const params = new URLSearchParams({
      array_id:       _selectedId,
      last_cleaned:   lastCleaned,
      partner_id:     PARTNER_ID,
      risk_threshold: threshold,
    });
    const headers = API_KEY ? { 'X-API-Key': API_KEY } : {};
    const res = await fetch(`${RECALC_URL}?${params}`, { headers });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();

    const arr = data.array_recommendation || {};
    const aoi = data.aoi_recommendation   || {};
    const win = arr.cleaning_window || aoi.window_start
      ? `${aoi.window_start ?? ''} → ${aoi.window_end ?? ''}`
      : 'Not recommended at current risk level';

    document.getElementById('recalc-result').innerHTML =
      `<strong>Updated Recommendation</strong>
       Cleaning window: <b>${win}</b><br>
       Confidence: <b>${aoi.confidence ?? '—'}</b><br>
       Priority: <b>${arr.priority ?? '—'}</b>`;
    document.getElementById('recalc-result').classList.add('visible');
  } catch (err) {
    document.getElementById('recalc-result').innerHTML =
      `<span style="color:#b91c1c">Could not reach backend: ${err.message}. Backend may be sleeping — wait 30s and retry.</span>`;
    document.getElementById('recalc-result').classList.add('visible');
  }

  btn.disabled = false;
  btn.textContent = 'Recalculate';
});

// Threshold slider label sync
document.getElementById('threshold-val')?.addEventListener('input', e => {
  document.getElementById('threshold-display').textContent = parseFloat(e.target.value).toFixed(2);
});

// ── stats overview ────────────────────────────────────────────────────────────
function updateStats() {
  const n = _features.length;
  const high   = _features.filter(f => (f.properties?.risk_score ?? 0) >= 0.65).length;
  const medium = _features.filter(f => {
    const s = f.properties?.risk_score ?? 0;
    return s >= 0.40 && s < 0.65;
  }).length;
  const avgArea = n
    ? (_features.reduce((s, f) => s + (f.properties?.area_m2 ?? 0), 0) / n).toFixed(0)
    : '—';

  document.getElementById('stat-total').textContent  = n;
  document.getElementById('stat-high').textContent   = high;
  document.getElementById('stat-medium').textContent = medium;
  document.getElementById('stat-avg-area').textContent = avgArea + ' m²';
}

// ── calculator ────────────────────────────────────────────────────────────────
function initCalculator() {
  ['calc-size', 'calc-rate', 'calc-sun', 'calc-soiling'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', calcUpdate);
  });
  calcUpdate();
}

function calcUpdate() {
  const size    = parseFloat(document.getElementById('calc-size')?.value)    || 6;
  const rate    = parseFloat(document.getElementById('calc-rate')?.value)    || 0.25;
  const sun     = parseFloat(document.getElementById('calc-sun')?.value)     || 5.5;
  const soiling = parseFloat(document.getElementById('calc-soiling')?.value) || 5;

  // Annual generation without soiling
  const annualKwh    = size * sun * 365;
  // Loss due to soiling
  const lossKwh      = annualKwh * (soiling / 100);
  const lossDollars  = lossKwh * rate;
  // Monthly figures
  const monthlyLoss  = lossDollars / 12;

  document.getElementById('calc-annual-kwh').textContent   = lossKwh.toFixed(0) + ' kWh';
  document.getElementById('calc-annual-usd').textContent   = '$' + lossDollars.toFixed(0);
  document.getElementById('calc-monthly-usd').textContent  = '$' + monthlyLoss.toFixed(0);
  document.getElementById('calc-recovery').textContent     = '$' + (lossDollars * 0.85).toFixed(0);
}

// ── QR code param handling ────────────────────────────────────────────────────
function handleQrParam() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('id');
  if (!id) return;

  // Wait for data to load, then focus the array
  const tryFocus = (attempts = 0) => {
    const feat = _features.find(f => String(f.properties?.array_id) === String(id));
    if (!feat) {
      if (attempts < 20) setTimeout(() => tryFocus(attempts + 1), 300);
      return;
    }

    // Find and select the layer
    _geojsonLayer?.eachLayer(layer => {
      if (String(layer.feature?.properties?.array_id) === String(id)) {
        const center = layer.getBounds().getCenter();
        _map.setView(center, 17);
        selectArray(layer.feature, layer);

        // Pulse the outline
        layer.getElement()?.classList.add('qr-highlight');

        // Scroll sidebar to detail
        setTimeout(() => {
          document.getElementById('array-detail')?.scrollIntoView({ behavior: 'smooth' });
        }, 500);
      }
    });
  };

  setTimeout(tryFocus, 800);
}

// ── utility ───────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('map-loading')?.classList.toggle('hidden', !on);
}
