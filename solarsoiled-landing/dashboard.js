/* SolarSoiled Homeowner Dashboard — dashboard.js
 *
 * Responsibilities:
 *  1. Initialize Leaflet map and load array GeoJSON (from backend or embedded data)
 *  2. Handle QR code URL param ?id=<array_id> — focus + highlight the array
 *  3. "Your Array" detail panel — populate on map click
 *  4. Recalculate button — calls /recommend-quick on the backend
 *  5. Energy/money calculator — pure client-side
 *  6. Area loss banner — always-visible dollar estimate across all arrays
 *  7. Address search — Nominatim geocoding + "report your array" callout
 *  8. Coverage boundary — dashed rectangle showing pilot area extent
 *  9. Circle markers — visible at low zoom so arrays are easy to spot
 * 10. Weather widget — Open-Meteo current conditions, top-right Leaflet control
 */

// ── config ────────────────────────────────────────────────────────────────────

const API_BASE   = 'https://solarsoiled-api.onrender.com';
const API_KEY    = '';
const PARTNER_ID = 'santa-cruz-outreach-v1';
const ARRAYS_URL = `${API_BASE}/results/${PARTNER_ID}/arrays`;
const RECALC_URL = `${API_BASE}/recommend-quick`;

const SC_CENTER = [36.974, -122.030];
const SC_ZOOM   = 13;

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

function soilingPct(score) {
  const pct = score * 15;
  return { lo: Math.max(1, (pct * 0.6).toFixed(1)), hi: (pct * 1.4).toFixed(1) };
}

// ── model config ──────────────────────────────────────────────────────────────
const MODEL_CONFIGS = {
  xgb: {
    key:   'risk_score',
    label: 'ML Model (XGBoost)',
    desc:  'Spatial-CV AUC 0.728 — trained on 1,000 NREL station-years across 15 states using weather, location, and structural features.',
  },
  somos: {
    key:   'somos_score',
    label: 'SOMOSclean Physics',
    desc:  'ENEL SOMOSclean trajectory model — exponential soiling accumulation with PM10-driven dust days and rain reset.',
  },
  kimber: {
    key:   'kimber_score',
    label: 'Kimber 2007',
    desc:  'Linear PM2.5 deposition model — accumulates proportional to daily PM2.5 and resets fully on any rain ≥ 1 mm.',
  },
};
let _activeModel = 'xgb';

// ── state ─────────────────────────────────────────────────────────────────────
let _map, _geojsonLayer, _features = [], _selectedLayer = null, _selectedId = null;
let _circleLayer = null, _circleMarkers = new Map(), _circleZoomHandler = null;
let _searchMarker = null;

// ── init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadArrays();
  initCalculator();
  initModelTabs();
  handleQrParam();
  initWeatherWidget();
  initSearch();
  document.getElementById('fit-btn')?.addEventListener('click', fitAllArrays);
});

// ── map init ──────────────────────────────────────────────────────────────────
function initMap() {
  _map = L.map('map', { zoomControl: true }).setView(SC_CENTER, SC_ZOOM);

  L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Tiles © Esri', maxZoom: 20 }
  ).addTo(_map);

  // Legend (bottom-right)
  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = () => {
    const div = L.DomUtil.create('div', 'map-legend');
    div.innerHTML = `
      <h4>Soiling Risk</h4>
      <div class="legend-row"><div class="legend-dot" style="background:#ef4444"></div> High (≥0.65)</div>
      <div class="legend-row"><div class="legend-dot" style="background:#eab308"></div> Medium (0.40–0.64)</div>
      <div class="legend-row"><div class="legend-dot" style="background:#22c55e"></div> Low (&lt;0.40)</div>
      <div class="legend-row" style="margin-top:.3rem;border-top:1px solid #e2e8f0;padding-top:.3rem">
        <div style="width:22px;height:0;border:1.5px dashed #60a5fa;flex-shrink:0;margin-right:.1rem"></div>
        <span>Coverage area</span>
      </div>
    `;
    return div;
  };
  legend.addTo(_map);

}

// ── data loading ──────────────────────────────────────────────────────────────
async function loadArrays() {
  setLoading(true);

  // Load embedded data immediately so the map is never blocked on the backend.
  if (window.FALLBACK_ARRAYS) {
    _features = window.FALLBACK_ARRAYS.features || [];
    renderArrays(window.FALLBACK_ARRAYS);
    updateStats();
    setLoading(false);
    fitAllArrays();
  }

  // Silently try the backend in the background — if it responds it may have
  // fresher scores; if it's cold-starting on Render we just skip it.
  try {
    const headers = API_KEY ? { 'X-API-Key': API_KEY } : {};
    const res = await fetch(ARRAYS_URL, { headers, signal: AbortSignal.timeout(8000) });
    if (!res.ok) throw new Error(`${res.status}`);
    const fc = await res.json();
    // Only re-render if the backend returned more arrays than the embedded file
    if ((fc.features?.length ?? 0) > _features.length) {
      _features = fc.features;
      renderArrays(fc);
      updateStats();
    }
  } catch {
    // Backend unavailable or timed out — embedded data already shown, nothing to do
  }

  setLoading(false);
}

function getScore(props) {
  const key = MODEL_CONFIGS[_activeModel].key;
  const v = props?.[key];
  return (v != null && !isNaN(v)) ? v : (props?.risk_score ?? 0);
}

// ── map rendering ─────────────────────────────────────────────────────────────
function renderArrays(fc) {
  if (_geojsonLayer) _map.removeLayer(_geojsonLayer);

  _geojsonLayer = L.geoJSON(fc, {
    style: feat => {
      const score = getScore(feat.properties);
      return {
        color:       '#1e293b',
        weight:      2,
        fillColor:   riskColor(score),
        fillOpacity: 0.65,
      };
    },
    onEachFeature: (feat, layer) => {
      layer.on('click', () => selectArray(feat, layer));
      const p = feat.properties || {};
      const score = getScore(p);
      layer.bindTooltip(
        `<b>Array #${p.array_id}</b><br>Risk: ${riskLabel(score)} (${score.toFixed(2)})<br>Area: ${p.area_m2} m²`,
        { sticky: true }
      );
    },
  }).addTo(_map);

  addCoverageBoundary();
  buildCircleLayer(fc);
}

function addCoverageBoundary() {
  if (!_geojsonLayer) return;
  const bounds = _geojsonLayer.getBounds();
  if (!bounds.isValid()) return;
  const pad = 0.003;
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  L.rectangle(
    [[sw.lat - pad, sw.lng - pad], [ne.lat + pad, ne.lng + pad]],
    { color: '#60a5fa', weight: 2, dashArray: '6 4', fill: false, interactive: false }
  ).addTo(_map);
}

function buildCircleLayer(fc) {
  if (_circleLayer) {
    if (_map.hasLayer(_circleLayer)) _map.removeLayer(_circleLayer);
    _circleLayer.clearLayers();
  }
  _circleLayer = L.layerGroup();
  _circleMarkers.clear();

  fc.features.forEach(feat => {
    const p = feat.properties || {};
    const coords = feat.geometry?.coordinates?.[0] ?? [];
    if (!coords.length) return;
    const clat = coords.reduce((s, c) => s + c[1], 0) / coords.length;
    const clon = coords.reduce((s, c) => s + c[0], 0) / coords.length;
    const score = getScore(p);

    const marker = L.circleMarker([clat, clon], {
      radius: 7,
      color: '#fff',
      weight: 1.5,
      fillColor: riskColor(score),
      fillOpacity: 0.9,
    });
    marker.on('click', () => {
      _geojsonLayer?.eachLayer(layer => {
        if (layer.feature === feat) selectArray(feat, layer);
      });
    });
    marker.bindTooltip(
      `<b>Array #${p.array_id}</b><br>Risk: ${riskLabel(score)} (${score.toFixed(2)})<br>Area: ${p.area_m2} m²`,
      { sticky: true }
    );
    _circleMarkers.set(String(p.array_id), marker);
    _circleLayer.addLayer(marker);
  });

  // Toggle circle visibility based on zoom — circles help at low zoom,
  // polygons are legible at zoom 15+
  if (_circleZoomHandler) _map.off('zoomend', _circleZoomHandler);
  _circleZoomHandler = () => {
    const z = _map.getZoom();
    if (z < 15) {
      if (!_map.hasLayer(_circleLayer)) _circleLayer.addTo(_map);
    } else {
      if (_map.hasLayer(_circleLayer)) _map.removeLayer(_circleLayer);
    }
  };
  _map.on('zoomend', _circleZoomHandler);
  _circleZoomHandler(); // set initial state
}

function refreshMapColors() {
  if (!_geojsonLayer) return;
  _geojsonLayer.eachLayer(layer => {
    if (layer === _selectedLayer) return;
    const score = getScore(layer.feature?.properties);
    layer.setStyle({ fillColor: riskColor(score), fillOpacity: 0.65, weight: 2, color: '#1e293b' });
    const p = layer.feature?.properties || {};
    const s = getScore(p);
    layer.setTooltipContent(
      `<b>Array #${p.array_id}</b><br>Risk: ${riskLabel(s)} (${s.toFixed(2)})<br>Area: ${p.area_m2} m²`
    );
  });
  if (_selectedLayer) {
    const score = getScore(_selectedLayer.feature?.properties);
    _selectedLayer.setStyle({ weight: 3, color: '#F7B731', fillColor: riskColor(score), fillOpacity: 0.8 });
  }

  // Refresh circle markers
  _circleMarkers.forEach((marker, id) => {
    const feat = _features.find(f => String(f.properties?.array_id) === id);
    if (!feat) return;
    const score = getScore(feat.properties);
    marker.setStyle({ fillColor: riskColor(score) });
    const p = feat.properties || {};
    marker.setTooltipContent(
      `<b>Array #${p.array_id}</b><br>Risk: ${riskLabel(score)} (${score.toFixed(2)})<br>Area: ${p.area_m2} m²`
    );
  });
}

// ── array selection + detail panel ───────────────────────────────────────────
function selectArray(feat, layer) {
  if (_selectedLayer && _selectedLayer !== layer) {
    const prevScore = getScore(_selectedLayer.feature?.properties);
    _selectedLayer.setStyle({ weight: 2, color: '#1e293b', fillColor: riskColor(prevScore), fillOpacity: 0.65 });
  }
  _selectedLayer = layer;
  _selectedId    = feat.properties?.array_id;

  const score = getScore(feat.properties);
  layer.setStyle({ weight: 3, color: '#F7B731', fillColor: riskColor(score), fillOpacity: 0.8 });
  layer.bringToFront();

  populateDetail(feat);
  document.getElementById('array-detail').classList.add('visible');
  document.getElementById('detail-placeholder').style.display = 'none';
  document.getElementById('recalc-section').style.display = '';
  document.getElementById('array-detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function populateDetail(feat) {
  const p     = feat.properties || {};
  const id    = p.array_id ?? '?';
  const score = getScore(p);
  const area  = p.area_m2 ?? '?';
  const sl    = soilingPct(score);
  const cfg   = MODEL_CONFIGS[_activeModel];

  document.getElementById('detail-id').textContent     = `Array #${id}`;
  document.getElementById('detail-badge').textContent   = riskLabel(score);
  document.getElementById('detail-badge').className     = `detail-badge ${riskBadgeCls(score)}`;
  document.getElementById('detail-score').textContent   = score.toFixed(2);
  document.getElementById('detail-area').textContent    = `${area} m²`;
  document.getElementById('detail-soiling').textContent = `${sl.lo}–${sl.hi}%`;

  const fill = document.querySelector('.risk-gauge-fill');
  fill.style.width      = `${Math.round(score * 100)}%`;
  fill.style.background = riskColor(score);

  const modelLabel = document.getElementById('detail-model-label');
  if (modelLabel) modelLabel.textContent = cfg.label;

  const scoreTable = document.getElementById('detail-all-scores');
  if (scoreTable) {
    const xgb    = (p.risk_score   != null) ? (+p.risk_score).toFixed(2)   : '—';
    const somos  = (p.somos_score  != null) ? (+p.somos_score).toFixed(2)  : '—';
    const kimber = (p.kimber_score != null) ? (+p.kimber_score).toFixed(2) : '—';
    scoreTable.innerHTML =
      `<tr><td class="lbl">ML (XGBoost)</td><td class="val">${xgb}</td></tr>` +
      `<tr><td class="lbl">SOMOSclean</td><td class="val">${somos}</td></tr>` +
      `<tr><td class="lbl">Kimber 2007</td><td class="val">${kimber}</td></tr>`;
  }

  // Address (from pre-baked lookup; only shown when available)
  const addrRow = document.getElementById('detail-address-row');
  const addrEl  = document.getElementById('detail-address');
  const addr = (typeof ADDRESS_LOOKUP !== 'undefined') ? ADDRESS_LOOKUP[String(id)] : null;
  if (addr && addrRow && addrEl) {
    addrEl.textContent = addr;
    addrRow.style.display = '';
  } else if (addrRow) {
    addrRow.style.display = 'none';
  }

  document.getElementById('threshold-val').value =
    document.getElementById('threshold-display').textContent = '0.60';

  document.getElementById('calc-soiling').value = (score * 15).toFixed(1);
  calcUpdate();

  // Update impact card header to name this array
  const tag = document.getElementById('impact-array-tag');
  if (tag) tag.textContent = `Array #${id}`;
  const note = document.getElementById('impact-note');
  if (note) note.textContent = 'Adjust inputs in the calculator below to personalize';

  // Update the always-visible banner to show this array's specific estimate
  renderArrayLossBanner(score, id);
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

document.getElementById('threshold-val')?.addEventListener('input', e => {
  document.getElementById('threshold-display').textContent = parseFloat(e.target.value).toFixed(2);
});

// ── stats overview ────────────────────────────────────────────────────────────
function updateStats() {
  const n      = _features.length;
  const high   = _features.filter(f => getScore(f.properties) >= 0.65).length;
  const medium = _features.filter(f => {
    const s = getScore(f.properties);
    return s >= 0.40 && s < 0.65;
  }).length;
  const avgArea = n
    ? (_features.reduce((s, f) => s + (f.properties?.area_m2 ?? 0), 0) / n).toFixed(0)
    : '—';

  document.getElementById('stat-total').textContent    = n;
  document.getElementById('stat-high').textContent     = high;
  document.getElementById('stat-medium').textContent   = medium;
  document.getElementById('stat-avg-area').textContent = avgArea + ' m²';

  renderAggregateLossBanner();
}

// ── area loss banner ──────────────────────────────────────────────────────────

function arrayLossUsd(score, kw = 6, sunH = 5.5, rate = 0.25) {
  const sl     = soilingPct(score);
  const annual = kw * sunH * 365;
  return {
    lo: Math.round(annual * (sl.lo / 100) * rate),
    hi: Math.round(annual * (sl.hi / 100) * rate),
  };
}

function fmtDollars(n) {
  return n >= 1000 ? `$${(n / 1000).toFixed(0)}K` : `$${n}`;
}

function renderAggregateLossBanner() {
  const el = document.getElementById('area-loss-content');
  if (!el || !_features.length) return;

  let totalLo = 0, totalHi = 0;
  const highRisk = _features.filter(f => getScore(f.properties) >= 0.65);
  let highHi = 0;

  _features.forEach(f => {
    const { lo, hi } = arrayLossUsd(getScore(f.properties));
    totalLo += lo;
    totalHi += hi;
  });
  highRisk.forEach(f => {
    highHi += arrayLossUsd(getScore(f.properties)).hi;
  });

  el.innerHTML = `
    <div class="area-loss-agg">
      <div class="area-loss-total">
        <span class="loss-range">${fmtDollars(totalLo)}–${fmtDollars(totalHi)}</span>
        <span class="loss-per-yr">/yr est. dry-season loss</span>
      </div>
      <div class="area-loss-sub">
        Across ${_features.length} detected arrays · ~6 kW avg assumption<br>
        <em>Rain resets panels — estimate assumes no rainfall.</em>
      </div>
      ${highRisk.length ? `
      <div class="area-loss-high-row">
        ⚠ <b>${highRisk.length} high-risk arrays</b> — up to ${fmtDollars(highHi)}/yr recoverable
      </div>` : ''}
      <div class="area-loss-cta">↓ Click any array to see your estimate</div>
    </div>
  `;
}

function renderArrayLossBanner(score, id) {
  const el = document.getElementById('area-loss-content');
  if (!el) return;
  const { lo, hi } = arrayLossUsd(score);
  el.innerHTML = `
    <div class="area-loss-array">
      <div class="loss-headline">Array #${id} &mdash; <span class="loss-val">${fmtDollars(lo)}–${fmtDollars(hi)}/yr</span></div>
      <div class="loss-reco">${localCleaningReco(score)}</div>
      <button class="area-loss-back" onclick="renderAggregateLossBanner()">← All arrays</button>
    </div>
  `;
}

function localCleaningReco(score) {
  if (score >= 0.65) {
    return '⚠ High soiling risk — a hose-down before the next dry stretch may be worth the 30 minutes. Professional cleaning is rarely cost-effective for residential systems (UC San Diego study).';
  }
  if (score >= 0.40) {
    return 'Moderate soiling — a rinse at the end of summer can recover some output. Rain typically resets most of this.';
  }
  return '✓ Low soiling — rain should handle this. Per the UC San Diego study, most CA homeowners don\'t recover enough to justify cleaning.';
}

// ── weather widget ────────────────────────────────────────────────────────────
function initWeatherWidget() {
  const WeatherControl = L.Control.extend({
    options: { position: 'topright' },
    onAdd() {
      const div = L.DomUtil.create('div', 'weather-widget');
      div.innerHTML = '<div class="ww-title">Santa Cruz · Weather</div><div class="ww-loading">Loading…</div>';
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      this._el = div;
      return div;
    },
  });

  const ctrl = new WeatherControl();
  ctrl.addTo(_map);

  const fetchWeather = async () => {
    try {
      const url =
        'https://api.open-meteo.com/v1/forecast' +
        '?latitude=36.974&longitude=-122.030' +
        '&current=temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m' +
        '&temperature_unit=fahrenheit&wind_speed_unit=mph';
      const res = await fetch(url);
      if (!res.ok) throw new Error(res.status);
      const data = await res.json();
      const c    = data.current || {};
      const temp  = c.temperature_2m  != null ? c.temperature_2m.toFixed(0)  : '—';
      const wind  = c.wind_speed_10m  != null ? c.wind_speed_10m.toFixed(0)  : '—';
      const humid = c.relative_humidity_2m != null ? c.relative_humidity_2m.toFixed(0) : '—';
      const rain  = c.precipitation   != null ? c.precipitation.toFixed(2)   : '0.00';

      let advisory = '';
      if (parseFloat(wind) > 15) {
        advisory = '<div class="ww-advisory warn">⚠ Windy — dust likely accumulating</div>';
      } else if (parseFloat(rain) > 0) {
        advisory = '<div class="ww-advisory good">🌧 Rain — panels self-cleaning!</div>';
      }

      ctrl._el.innerHTML = `
        <div class="ww-title">Santa Cruz · Now</div>
        <div class="ww-row"><span>${temp}°F</span><span>Wind ${wind} mph</span></div>
        <div class="ww-row"><span>Humidity ${humid}%</span><span>Rain ${rain}"</span></div>
        ${advisory}
      `;
    } catch {
      if (ctrl._el) {
        ctrl._el.innerHTML =
          '<div class="ww-title">Santa Cruz · Weather</div>' +
          '<div class="ww-loading">Unavailable</div>';
      }
    }
  };

  fetchWeather();
  setInterval(fetchWeather, 30 * 60 * 1000);
}

// ── address search ────────────────────────────────────────────────────────────
function initSearch() {
  const input = document.getElementById('search-input');
  const btn   = document.getElementById('search-btn');
  if (!input || !btn) return;

  let _suggestTimer = null;
  let _activeIdx    = -1;

  // ── suggestions dropdown ────────────────────────────────────────────────────
  const hideSuggestions = () => {
    const list = document.getElementById('search-suggestions');
    list?.classList.add('hidden');
    _activeIdx = -1;
  };

  const showSuggestions = results => {
    const list = document.getElementById('search-suggestions');
    if (!list || !results.length) { hideSuggestions(); return; }

    list.innerHTML = results.map((r, i) => {
      // Trim Nominatim's very long display names to first 4 comma-parts
      const label = r.display_name.split(', ').slice(0, 4).join(', ');
      return `<li data-idx="${i}" data-display="${r.display_name.replace(/"/g, '&quot;')}"
                  data-lat="${r.lat}" data-lon="${r.lon}">${label}</li>`;
    }).join('');
    list.classList.remove('hidden');
    _activeIdx = -1;

    list.querySelectorAll('li').forEach(li => {
      li.addEventListener('mousedown', e => {
        e.preventDefault(); // prevent blur before we read the value
        input.value = li.dataset.display;
        hideSuggestions();
        doSearch();
      });
    });
  };

  const moveSuggestion = delta => {
    const list = document.getElementById('search-suggestions');
    if (!list || list.classList.contains('hidden')) return;
    const items = list.querySelectorAll('li');
    if (!items.length) return;
    items[_activeIdx]?.classList.remove('active');
    _activeIdx = (_activeIdx + delta + items.length) % items.length;
    const active = items[_activeIdx];
    active.classList.add('active');
    input.value = active.dataset.display;
  };

  // Debounced fetch as user types
  input.addEventListener('input', () => {
    clearTimeout(_suggestTimer);
    const q = input.value.trim();
    if (q.length < 3) { hideSuggestions(); return; }
    _suggestTimer = setTimeout(async () => {
      try {
        // Bias results toward Santa Cruz with a loose viewbox
        const url = `https://nominatim.openstreetmap.org/search?format=json` +
          `&q=${encodeURIComponent(q)}&limit=5&countrycodes=us` +
          `&viewbox=-122.15,36.92,-121.85,36.99&bounded=0`;
        const res  = await fetch(url, { headers: { 'Accept-Language': 'en' } });
        const data = await res.json();
        showSuggestions(data);
      } catch { hideSuggestions(); }
    }, 280);
  });

  // Keyboard: arrows to navigate, Enter to select, Escape to dismiss
  input.addEventListener('keydown', e => {
    const list = document.getElementById('search-suggestions');
    const open = list && !list.classList.contains('hidden');
    if (e.key === 'ArrowDown')  { e.preventDefault(); open ? moveSuggestion(1)  : null; return; }
    if (e.key === 'ArrowUp')    { e.preventDefault(); open ? moveSuggestion(-1) : null; return; }
    if (e.key === 'Escape')     { hideSuggestions(); return; }
    if (e.key === 'Enter')      { hideSuggestions(); doSearch(); }
  });

  input.addEventListener('blur', () => setTimeout(hideSuggestions, 150));

  // ── geocode + map action ────────────────────────────────────────────────────
  const doSearch = async () => {
    const q = input.value.trim();
    if (!q) return;
    btn.disabled = true;
    if (_searchMarker) { _map.removeLayer(_searchMarker); _searchMarker = null; }
    hideSearchCallout();

    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=1&countrycodes=us`;
      const res  = await fetch(url, { headers: { 'Accept-Language': 'en' } });
      const data = await res.json();

      if (!data.length) {
        showSearchCallout('Address not found. Try a more complete address (street, city, state).');
        return;
      }

      const latN = parseFloat(data[0].lat), lonN = parseFloat(data[0].lon);

      _searchMarker = L.marker([latN, lonN], {
        icon: L.divIcon({
          className: 'search-pin',
          html: '<div class="search-pin-dot"></div>',
          iconSize: [18, 18],
          iconAnchor: [9, 9],
        }),
      }).addTo(_map);

      // Find nearest array centroid
      let nearest = null, nearestDist = Infinity;
      _features.forEach(feat => {
        const coords = feat.geometry?.coordinates?.[0] ?? [];
        if (!coords.length) return;
        const clat = coords.reduce((s, c) => s + c[1], 0) / coords.length;
        const clon = coords.reduce((s, c) => s + c[0], 0) / coords.length;
        const d = haversineMeters(latN, lonN, clat, clon);
        if (d < nearestDist) { nearestDist = d; nearest = feat; }
      });

      if (nearest && nearestDist < 100) {
        _map.setView([latN, lonN], 17);
        _geojsonLayer?.eachLayer(layer => {
          if (layer.feature === nearest) selectArray(nearest, layer);
        });
      } else if (nearest && nearestDist < 600) {
        _map.setView([latN, lonN], 16);
        const body = encodeURIComponent(`Address: ${q}\nI have solar panels here but they were not detected.`);
        showSearchCallout(
          `Nearest detected array is ${Math.round(nearestDist)} m away — click it on the map. ` +
          `Not yours? <a href="mailto:solarsoil.app@gmail.com?subject=Report+my+array&body=${body}">Report your panels →</a>`
        );
      } else {
        _map.setView([latN, lonN], 16);
        const body = encodeURIComponent(`Address: ${q}\nI have solar panels here but they were not detected.`);
        showSearchCallout(
          `No solar array detected near this address. Have panels? ` +
          `<a href="mailto:solarsoil.app@gmail.com?subject=Report+my+array&body=${body}">Let us know →</a> ` +
          `— we may add your area to a future analysis.`
        );
      }
    } catch {
      showSearchCallout('Search unavailable — check your connection and try again.');
    } finally {
      btn.disabled = false;
    }
  };

  btn.addEventListener('click', doSearch);
  document.getElementById('search-dismiss')?.addEventListener('click', hideSearchCallout);
}

function showSearchCallout(msg) {
  const el    = document.getElementById('search-callout');
  const msgEl = document.getElementById('search-callout-msg');
  if (!el || !msgEl) return;
  msgEl.innerHTML = msg;
  el.classList.remove('hidden');
}

function hideSearchCallout() {
  document.getElementById('search-callout')?.classList.add('hidden');
}

function haversineMeters(lat1, lon1, lat2, lon2) {
  const R  = 6371000;
  const φ1 = lat1 * Math.PI / 180, φ2 = lat2 * Math.PI / 180;
  const Δφ = (lat2 - lat1) * Math.PI / 180;
  const Δλ = (lon2 - lon1) * Math.PI / 180;
  const a  = Math.sin(Δφ / 2) ** 2 + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
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

  const annualKwh   = size * sun * 365;
  const lossKwh     = annualKwh * (soiling / 100);
  const lossDollars = lossKwh * rate;
  const monthlyLoss = lossDollars / 12;

  document.getElementById('calc-annual-kwh').textContent  = lossKwh.toFixed(0) + ' kWh';
  document.getElementById('calc-annual-usd').textContent  = '$' + lossDollars.toFixed(0);
  document.getElementById('calc-monthly-usd').textContent = '$' + monthlyLoss.toFixed(0);
  document.getElementById('calc-recovery').textContent    = '$' + (lossDollars * 0.85).toFixed(0);
}

// ── QR code param handling ────────────────────────────────────────────────────
function handleQrParam() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('id');
  if (!id) return;

  const tryFocus = (attempts = 0) => {
    const feat = _features.find(f => String(f.properties?.array_id) === String(id));
    if (!feat) {
      if (attempts < 20) setTimeout(() => tryFocus(attempts + 1), 300);
      return;
    }

    _geojsonLayer?.eachLayer(layer => {
      if (String(layer.feature?.properties?.array_id) === String(id)) {
        const center = layer.getBounds().getCenter();
        _map.setView(center, 17);
        selectArray(layer.feature, layer);
        layer.getElement()?.classList.add('qr-highlight');
        setTimeout(() => {
          document.getElementById('array-detail')?.scrollIntoView({ behavior: 'smooth' });
        }, 500);
      }
    });
  };

  setTimeout(tryFocus, 800);
}

// ── model tabs ────────────────────────────────────────────────────────────────
function initModelTabs() {
  document.querySelectorAll('.model-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const model = btn.dataset.model;
      if (model === _activeModel) return;
      _activeModel = model;

      document.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      refreshMapColors();
      updateStats();

      if (_selectedLayer) populateDetail(_selectedLayer.feature);

      const descEl = document.getElementById('model-description');
      if (descEl) descEl.textContent = MODEL_CONFIGS[model].desc;
    });
  });
}

// ── utility ───────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('map-loading')?.classList.toggle('hidden', !on);
}

function fitAllArrays() {
  if (!_features.length) return;
  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  _features.forEach(f => {
    (f.geometry?.coordinates?.[0] ?? []).forEach(([lng, lat]) => {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
    });
  });
  if (isFinite(minLat)) _map.fitBounds([[minLat, minLng], [maxLat, maxLng]], { padding: [30, 30] });
}
