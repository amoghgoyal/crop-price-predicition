/* ═══════════════════════════════════════════════════════════════════════
   CropSignal — Dashboard JavaScript
   ═══════════════════════════════════════════════════════════════════════ */

let allPredictions = [];
let currentFilter = '';
let chart = null;
let weatherData = null;

// ═══ BOOT ════════════════════════════════════════════════════════════════
async function boot() {
  try {
    const [sigRes, weatherRes] = await Promise.all([
      fetch('/api/signals'),
      fetch('/api/weather?region=average'),
    ]);

    if (sigRes.ok) {
      const data = await sigRes.json();
      allPredictions = data.predictions || [];
      document.getElementById('updated').textContent = 'Updated ' + (data.generated_at || '').slice(0, 10);
      document.getElementById('model-type').textContent = data.model_type || '';
      updateStats();
      renderCropList(allPredictions);
      buildTicker();
      populateCropSelect();
      populateStates();
      if (allPredictions.length) selectCrop(allPredictions[0]);
    } else {
      showError('Run ml_pipeline.py first to generate predictions.');
    }

    if (weatherRes.ok) {
      weatherData = await weatherRes.json();
      updateWeather();
    }
  } catch (e) {
    showError('Cannot connect. Is app.py running?');
  }
}

function showError(msg) {
  document.getElementById('crop-list').innerHTML =
    `<div style="padding:1.5rem;color:var(--red);font-size:.85rem;">⚠ ${msg}</div>`;
}

// ═══ STATS ═══════════════════════════════════════════════════════════════
function updateStats() {
  const buy = allPredictions.filter(p => p.signal === 'BUY').length;
  const sell = allPredictions.filter(p => p.signal === 'SELL').length;
  const hold = allPredictions.filter(p => p.signal === 'HOLD').length;
  animateCounter('s-all', allPredictions.length);
  animateCounter('s-buy', buy);
  animateCounter('s-sell', sell);
  animateCounter('s-hold', hold);
}

function animateCounter(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  let current = 0;
  const step = Math.max(1, Math.ceil(target / 30));
  const interval = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = current;
    if (current >= target) clearInterval(interval);
  }, 30);
}

// ═══ WEATHER ═════════════════════════════════════════════════════════════
function updateWeather() {
  if (!weatherData) return;
  const el = document.getElementById('weather-info');
  if (el) {
    const temp = weatherData.temperature || '--';
    const desc = weatherData.description || '';
    const rain = weatherData.rain_7d_avg || 0;
    el.innerHTML = `<span class="temp">${temp}°C</span> ${desc} · Rain avg: ${rain}mm`;
  }
}

// ═══ TICKER ══════════════════════════════════════════════════════════════
function buildTicker() {
  const track = document.getElementById('ticker-track');
  if (!track || !allPredictions.length) return;
  const top = allPredictions.slice(0, 30);
  const items = top.map(p => {
    const cls = p.pct_change >= 0 ? 'up' : 'down';
    const arrow = p.pct_change >= 0 ? '▲' : '▼';
    return `<div class="ticker-item">
      <span class="name">${p.commodity}</span>
      <span class="price">₹${p.current_price.toLocaleString()}</span>
      <span class="change ${cls}">${arrow}${Math.abs(p.pct_change).toFixed(1)}%</span>
    </div>`;
  }).join('');
  track.innerHTML = items + items; // duplicate for seamless loop
}

// ═══ CROP LIST ═══════════════════════════════════════════════════════════
function renderCropList(items) {
  const el = document.getElementById('crop-list');
  if (!items.length) {
    el.innerHTML = '<div style="padding:1.5rem;color:var(--text-dim);font-size:.85rem;">No crops found</div>';
    return;
  }
  el.innerHTML = items.map(p => `
    <div class="crop-item" data-crop="${p.commodity}" onclick='selectCropByName("${p.commodity.replace(/'/g, "\\'")}");'>
      <div>
        <div class="crop-name">${p.commodity}</div>
        <div class="crop-price">₹${p.current_price.toLocaleString()}/qtl</div>
      </div>
      <div class="crop-meta">
        <span class="badge ${p.signal}">${p.signal}</span>
        <span class="change-sm ${p.pct_change >= 0 ? 'up' : 'down'}">
          ${p.pct_change >= 0 ? '▲' : '▼'}${Math.abs(p.pct_change).toFixed(1)}%
        </span>
      </div>
    </div>
  `).join('');
}

// ═══ FILTER & SEARCH ═════════════════════════════════════════════════════
function filterSignal(sig, el) {
  currentFilter = sig;
  document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
  if (el) el.classList.add('active');
  applyFilters();
}

function applyFilters() {
  const q = (document.getElementById('search')?.value || '').toLowerCase();
  let items = allPredictions;
  if (currentFilter) items = items.filter(p => p.signal === currentFilter);
  if (q) items = items.filter(p => p.commodity.toLowerCase().includes(q));
  renderCropList(items);
}

// ═══ TABS ═════════════════════════════════════════════════════════════════
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`[data-tab="${tabId}"]`)?.classList.add('active');
  document.getElementById('tab-' + tabId)?.classList.add('active');
  if (tabId === 'crops') renderCropsTable();
}

// ═══ SELECT CROP ═════════════════════════════════════════════════════════
function selectCropByName(name) {
  const p = allPredictions.find(x => x.commodity === name);
  if (p) selectCrop(p);
}

function selectCrop(p) {
  // Highlight sidebar
  document.querySelectorAll('.crop-item').forEach(el => el.classList.remove('active'));
  const el = document.querySelector(`[data-crop="${p.commodity}"]`);
  if (el) { el.classList.add('active'); el.scrollIntoView({ block: 'nearest' }); }

  const pct = p.pct_change;
  const pClass = pct >= 0 ? 'up' : 'down';
  const arrow = pct >= 0 ? '▲' : '▼';
  const emoji = { BUY: '🟢', SELL: '🔴', HOLD: '🟡' }[p.signal] || '⚪';

  // Feature importance bars
  const featHtml = (p.feature_importance || []).map(f => {
    const maxImp = (p.feature_importance[0]?.importance || 1);
    const pctW = (f.importance / maxImp * 100).toFixed(0);
    const label = f.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    return `<div class="feat-bar-wrap">
      <div class="feat-bar-label"><span>${label}</span><span>${(f.importance * 100).toFixed(1)}%</span></div>
      <div class="feat-bar"><div class="feat-bar-fill" style="width:${pctW}%"></div></div>
    </div>`;
  }).join('');

  // Confidence
  const confLow = p.confidence_low ?? (p.predicted_price * 0.9);
  const confHigh = p.confidence_high ?? (p.predicted_price * 1.1);

  document.getElementById('detail').innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-title">${p.commodity}</div>
        <div class="detail-sub">Price Unit: ₹/Quintal · ${p.data_points.toLocaleString()} data points</div>
      </div>
      <div class="signal-card ${p.signal}">
        <div class="signal-emoji">${emoji}</div>
        <div>
          <div class="signal-label ${p.signal}">${p.signal}</div>
          <div class="signal-dates">${p.current_date} → ${p.predicted_date}</div>
        </div>
      </div>
    </div>

    <div class="metrics-grid">
      <div class="metric">
        <div class="m-label">Current Price</div>
        <div class="m-value">₹${p.current_price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
        <div class="m-sub">As of ${p.current_date}</div>
      </div>
      <div class="metric">
        <div class="m-label">7-Day Forecast</div>
        <div class="m-value ${pClass}">₹${p.predicted_price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
        <div class="m-sub">By ${p.predicted_date}</div>
      </div>
      <div class="metric">
        <div class="m-label">Expected Change</div>
        <div class="m-value ${pClass}">${arrow} ${Math.abs(pct).toFixed(2)}%</div>
        <div class="m-sub">${pct >= 0 ? 'Price rising' : 'Price falling'}</div>
      </div>
      <div class="metric">
        <div class="m-label">Confidence Range</div>
        <div class="m-value">₹${Math.round(confLow)} – ₹${Math.round(confHigh)}</div>
        <div class="m-sub">95% interval</div>
      </div>
      <div class="metric">
        <div class="m-label">Model R²</div>
        <div class="m-value">${(Math.max(0, p.r2) * 100).toFixed(1)}%</div>
        <div class="m-sub">Accuracy score</div>
      </div>
      <div class="metric">
        <div class="m-label">MAE</div>
        <div class="m-value">₹${p.mae.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
        <div class="m-sub">Mean absolute error</div>
      </div>
    </div>

    <div class="chart-wrap">
      <div class="chart-title">📈 Price History + 7-Day Forecast</div>
      <canvas id="price-chart"></canvas>
    </div>

    ${featHtml ? `<div class="chart-wrap"><div class="chart-title">🧠 Top Price Drivers</div>${featHtml}</div>` : ''}

    <div class="disclaimer">⚠ Predictions are based on historical patterns and ML models. This is not financial advice.</div>
  `;

  renderChart(p);
}

// ═══ CHART ═══════════════════════════════════════════════════════════════
function renderChart(p) {
  if (chart) chart.destroy();
  const hist = p.history || [];
  const labels = hist.map(h => h.report_date);
  const prices = hist.map(h => +h.modal_price.toFixed(2));

  labels.push(p.predicted_date);
  prices.push(null);

  const ctx = document.getElementById('price-chart').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 320);
  grad.addColorStop(0, 'rgba(34,211,238,0.2)');
  grad.addColorStop(1, 'rgba(34,211,238,0)');

  const sigColor = p.signal === 'BUY' ? '#4ade80' : p.signal === 'SELL' ? '#f87171' : '#fbbf24';

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Modal Price (₹/qtl)',
          data: prices,
          borderColor: '#22d3ee',
          backgroundColor: grad,
          borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
          fill: true, tension: 0.35, spanGaps: false,
        },
        {
          label: 'Forecast',
          data: [...Array(prices.length - 1).fill(null), prices[prices.length - 2], p.predicted_price],
          borderColor: sigColor,
          borderWidth: 2, borderDash: [6, 4],
          pointRadius: [0, 6],
          pointBackgroundColor: sigColor,
          fill: false, tension: 0,
        }
      ]
    },
    options: {
      responsive: true,
      animation: { duration: 800, easing: 'easeOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(6,10,19,0.95)',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          titleFont: { size: 12 }, bodyFont: { size: 12 },
          callbacks: { label: ctx => ` ₹${ctx.parsed.y?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#475569', maxTicksLimit: 8, font: { size: 10 } },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#475569', font: { size: 10 },
            callback: v => '₹' + v.toLocaleString('en-IN', { maximumFractionDigits: 0 })
          },
        }
      }
    }
  });
}

// ═══ ALL CROPS TABLE ═════════════════════════════════════════════════════
function renderCropsTable() {
  const wrap = document.getElementById('crops-table-body');
  if (!wrap) return;
  const q = (document.getElementById('table-search')?.value || '').toLowerCase();
  const sig = document.getElementById('table-filter')?.value || '';
  const sort = document.getElementById('table-sort')?.value || 'change';

  let items = [...allPredictions];
  if (q) items = items.filter(p => p.commodity.toLowerCase().includes(q));
  if (sig) items = items.filter(p => p.signal === sig);

  if (sort === 'change') items.sort((a, b) => Math.abs(b.pct_change) - Math.abs(a.pct_change));
  else if (sort === 'price') items.sort((a, b) => b.current_price - a.current_price);
  else if (sort === 'name') items.sort((a, b) => a.commodity.localeCompare(b.commodity));
  else if (sort === 'r2') items.sort((a, b) => b.r2 - a.r2);

  wrap.innerHTML = items.map(p => {
    const cls = p.pct_change >= 0 ? 'up' : 'down';
    const arrow = p.pct_change >= 0 ? '▲' : '▼';
    return `<tr>
      <td><span class="crop-link" onclick="switchTab('overview');selectCropByName('${p.commodity.replace(/'/g, "\\'")}')">${p.commodity}</span></td>
      <td><span class="badge ${p.signal}">${p.signal}</span></td>
      <td>₹${p.current_price.toLocaleString()}</td>
      <td class="${cls}">₹${p.predicted_price.toLocaleString()}</td>
      <td class="${cls}">${arrow} ${Math.abs(p.pct_change).toFixed(1)}%</td>
      <td>${(Math.max(0, p.r2) * 100).toFixed(1)}%</td>
      <td>₹${p.mae.toLocaleString()}</td>
      <td>${p.data_points.toLocaleString()}</td>
    </tr>`;
  }).join('');
}

// ═══ LIVE PREDICTION ═════════════════════════════════════════════════════
function populateCropSelect() {
  const sel = document.getElementById('predict-crop');
  if (!sel) return;
  sel.innerHTML = '<option value="">— Select a crop —</option>' +
    allPredictions.map(p => `<option value="${p.commodity}">${p.commodity}</option>`).join('');
}

function updateRainValue(val) {
  const v = parseFloat(val);
  document.getElementById('rain-value').textContent = v === 0 ? 'Auto' : val + ' mm';
}

async function populateStates() {
  try {
    const res = await fetch('/api/states');
    if (!res.ok) return;
    const states = await res.json();
    const sel = document.getElementById('predict-state');
    if (!sel) return;
    sel.innerHTML = states.map(s =>
      `<option value="${s.key}" ${s.key === 'average' ? 'selected' : ''}>${s.name}</option>`
    ).join('');
    onStateChange();  // fetch weather for default
  } catch (e) { console.warn('Failed to load states', e); }
}

async function onStateChange() {
  const region = document.getElementById('predict-state')?.value || 'average';
  const regionName = document.getElementById('predict-state')?.selectedOptions[0]?.textContent || region;
  const strip = document.getElementById('live-weather-strip');
  const rainEl = document.getElementById('state-rain');
  const tempEl = document.getElementById('state-temp');
  const weatherEl = document.getElementById('state-weather');
  if (!strip) return;

  // Show loading state with spinner
  strip.style.display = 'block';
  strip.style.opacity = '0.6';
  if (rainEl) rainEl.textContent = '...';
  if (tempEl) tempEl.textContent = '...';
  if (weatherEl) weatherEl.innerHTML = `<span style="color:var(--cyan)">Fetching ${regionName} weather...</span>`;

  try {
    const res = await fetch(`/api/weather?region=${region}`);
    if (!res.ok) throw new Error('API error');
    const w = await res.json();
    strip.style.opacity = '1';
    if (w.error && w.temperature === 28) {
      // Fallback data — API failed
      if (weatherEl) weatherEl.innerHTML = `<span style="color:var(--amber)">⚠ Using estimates (API slow)</span>`;
      if (rainEl) rainEl.textContent = w.rain_7d_avg ?? '--';
      if (tempEl) tempEl.textContent = w.temperature ?? '--';
    } else {
      if (rainEl) rainEl.textContent = w.rain_7d_avg ?? w.precipitation ?? '--';
      if (tempEl) tempEl.textContent = w.temperature ?? '--';
      if (weatherEl) weatherEl.textContent = w.description || '';
    }
  } catch (e) {
    strip.style.opacity = '1';
    if (weatherEl) weatherEl.innerHTML = `<span style="color:var(--red)">⚠ Connection failed</span>`;
  }
}

async function runPrediction() {
  const crop = document.getElementById('predict-crop')?.value;
  const rainVal = parseFloat(document.getElementById('predict-rain')?.value);
  const rainfall = rainVal > 0 ? rainVal : null;  // null = use live weather for state
  const month = parseInt(document.getElementById('predict-month')?.value);
  const region = document.getElementById('predict-state')?.value || 'average';
  const resultDiv = document.getElementById('predict-result');

  if (!crop) { resultDiv.innerHTML = '<div class="result-placeholder">⚠ Please select a crop</div>'; return; }

  resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Predicting...</div>';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ crop, rainfall, month, region }),
    });
    const data = await res.json();
    if (data.error) {
      resultDiv.innerHTML = `<div class="result-placeholder">⚠ ${data.error}</div>`;
      return;
    }

    const cls = data.pct_change >= 0 ? 'up' : 'down';
    const arrow = data.pct_change >= 0 ? '▲' : '▼';
    const emoji = { BUY: '🟢', SELL: '🔴', HOLD: '🟡' }[data.signal] || '⚪';

    resultDiv.innerHTML = `
      <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.5rem;">Prediction for ${data.commodity} · <span style="color:var(--cyan)">${data.inputs?.region || 'India'}</span></div>
      <div class="result-signal ${cls}" style="color:${data.signal === 'BUY' ? 'var(--green)' : data.signal === 'SELL' ? 'var(--red)' : 'var(--amber)'}">
        ${emoji} ${data.signal}
      </div>
      <div class="result-price ${cls}">₹${data.predicted_price.toLocaleString()}<span style="font-size:0.9rem;color:var(--text-muted)">/qtl</span></div>
      <div class="result-change ${cls}">${arrow} ${Math.abs(data.pct_change).toFixed(2)}% from ₹${data.current_price.toLocaleString()}</div>
      <div class="result-confidence">
        95% Confidence: ₹${data.confidence_low?.toLocaleString()} – ₹${data.confidence_high?.toLocaleString()}
      </div>
      <div style="margin-top:1rem;font-size:0.78rem;color:var(--text-muted);text-align:left;width:100%;max-width:320px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem 1rem;">
          <span>📅 Prediction for:</span><span style="color:var(--text)">${data.prediction_for || '--'}</span>
          <span>🗓 ${data.inputs?.day_of_week || '--'}, ${data.inputs?.month_name || '--'}</span><span style="color:var(--text)">${data.inputs?.season || '--'} season</span>
          <span>🌧 Rainfall:</span><span style="color:var(--cyan)">${data.inputs?.rainfall ?? 'N/A'} mm</span>
          <span>📊 vs Normal:</span><span style="color:${(data.inputs?.rain_deviation||0) > 0 ? 'var(--green)' : 'var(--red)'}">${data.inputs?.rain_deviation > 0 ? '+' : ''}${data.inputs?.rain_deviation?.toFixed(1) || '0'} mm</span>
          <span>🌡 Temperature:</span><span style="color:var(--amber)">${data.inputs?.temperature || '--'}°C</span>
          <span>☁ Weather:</span><span style="color:var(--text)">${data.inputs?.weather || '--'}</span>
        </div>
      </div>
      ${data.features_used ? `
      <div style="margin-top:0.8rem;font-size:0.7rem;color:var(--text-dim);text-align:left;width:100%;max-width:320px;background:rgba(255,255,255,0.03);padding:0.6rem 0.8rem;border-radius:8px;">
        <div style="font-weight:600;margin-bottom:0.3rem;color:var(--text-muted);">Features used (${data.features_used.history_points_used} history pts):</div>
        Lag 1d: ₹${data.features_used.price_lag_1?.toLocaleString()} · Lag 7d: ₹${data.features_used.price_lag_7?.toLocaleString()} · Lag 30d: ₹${data.features_used.price_lag_30?.toLocaleString()}<br>
        7d avg: ₹${data.features_used.roll_mean_7?.toLocaleString()} · 30d avg: ₹${data.features_used.roll_mean_30?.toLocaleString()} · Momentum: ${data.features_used.momentum_7d > 0 ? '+' : ''}₹${data.features_used.momentum_7d?.toLocaleString()}
      </div>` : ''}
    `;
  } catch (e) {
    resultDiv.innerHTML = `<div class="result-placeholder">⚠ Prediction failed: ${e.message}</div>`;
  }
}

// ═══ INIT ════════════════════════════════════════════════════════════════
boot();
