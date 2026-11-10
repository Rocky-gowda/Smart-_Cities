/* Police / authority dashboard: live map, real-time incidents, digital-ID verification,
 * hazard-zone management and the blockchain audit trail. */
'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const COLORS = { LOW: '#2e7d32', MODERATE: '#b7791f', HIGH: '#ef6c00', CRITICAL: '#c62828' };
const STATUS_PILL = { pending: 'bad', corroborated: 'warn', verified: 'ok', rejected: '' };
const S = { reports: [], photos: {}, token: null, map: null, incidents: [], tourists: [], zones: [], markers: {}, incMarkers: {}, ws: null, pick: null, scanner: null };
try { S.token = sessionStorage.getItem('tsm.police'); } catch { /* storage unavailable */ }
function toast(m, ms = 5000) { const t = $('#toast'); t.textContent = m; t.classList.remove('hidden'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add('hidden'), ms); }

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + S.token },
                                body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (r.status === 401) { logout(); throw new Error('login required'); }
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch { /* */ } throw new Error(m); }
  return r.json();
}

/* ---------------------------------------------------------------- login */
$('#loginForm').onsubmit = async e => {
  e.preventDefault();
  const f = Object.fromEntries(new FormData(e.target).entries());
  const r = await fetch('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(f) });
  if (!r.ok) { $('#loginMsg').textContent = 'Wrong username or password.'; return; }
  const d = await r.json(); S.token = d.token; try { sessionStorage.setItem('tsm.police', d.token); } catch { /* */ }
  start();
};
function logout() { try { sessionStorage.removeItem('tsm.police'); } catch { /* */ } location.reload(); }
$('#logoutBtn').onclick = logout;

async function boot() {
  const cfg = await (await fetch('/api/config')).json(); S.cfg = cfg;
  $('#llm').textContent = 'Cloud check: ' + cfg.llm;
  if (S.token) { try { await api('/api/me'); return start(); } catch { /* fall through */ } }
  $('#login').classList.remove('hidden');
}

async function start() {
  $('#login').classList.add('hidden'); $('#app').classList.remove('hidden');
  const me = await api('/api/me');
  $('#who').textContent = me.sub + ' (' + me.role + ')'; $('#who').classList.remove('hidden'); $('#logoutBtn').classList.remove('hidden');
  initMap(); tabs();
  await Promise.all([loadIncidents(), loadTourists(), loadZones(), loadReports()]);
  connect();
  setInterval(loadTourists, 30000);
}

/* ---------------------------------------------------------------- map */
function initMap() {
  S.map = L.map('map').setView([12.9763, 77.5929], 14);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(S.map);
  S.zoneLayer = L.layerGroup().addTo(S.map); S.repLayer = L.layerGroup().addTo(S.map); S.incLayer = L.layerGroup().addTo(S.map); S.tourLayer = L.layerGroup().addTo(S.map);
  S.map.on('click', e => {
    if ($('#tab-zones').classList.contains('hidden')) return;
    S.pick = e.latlng; $('#zPos').textContent = `Picked ${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`; $('#zSave').disabled = false;
    if (S.pickMarker) S.pickMarker.remove();
    S.pickMarker = L.circle(e.latlng, { radius: +$('#zRadius').value, color: '#6a3d9a', dashArray: '5 5' }).addTo(S.map);
  });
}

/* ---------------------------------------------------------------- tabs */
function tabs() {
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('active', x === b));
    document.querySelectorAll('.pane').forEach(p => p.classList.toggle('hidden', p.id !== 'tab-' + b.dataset.tab));
    if (b.dataset.tab === 'chain') loadChain();
    if (b.dataset.tab === 'tourists') loadTourists();
  });
  $('#filter').onchange = renderIncidents;
  $('#rFilter').onchange = renderReports;
}

/* ---------------------------------------------------------------- incidents */
async function loadIncidents() { S.incidents = await api('/api/incidents'); renderIncidents(); }
function renderIncidents() {
  const f = $('#filter').value, list = S.incidents.filter(i => !f || i.status === f);
  $('#kOpen').textContent = S.incidents.filter(i => i.status === 'open').length;
  $('#kCrit').textContent = S.incidents.filter(i => i.final_class === 'CRITICAL' && i.status !== 'resolved').length;
  $('#kCloud').textContent = S.incidents.filter(i => i.escalated).length;
  $('#incList').innerHTML = list.length ? list.map(i => `
    <div class="inc ${i.final_class} ${i._new ? 'new' : ''}" data-id="${i.id}">
      <div class="row"><span class="badge ${i.final_class}">${i.final_class}</span><b class="grow">${esc(i.name || i.tourist_id)}</b>
        <span class="pill ${i.status === 'open' ? 'bad' : i.status === 'resolved' ? 'ok' : 'warn'}">${i.status}</span></div>
      <div class="muted" style="margin-top:4px">${new Date(i.t).toLocaleString()} · ${i.source === 'sos' ? '🆘 SOS button' : i.source === 'report' ? '📢 tourist report: ' + esc(i.features?.label || '') : 'geofence alert'}${i.synced_late ? ' · sent after reconnecting' : ''}</div>
      <div style="margin-top:4px">${esc(i.action)}</div>
      <div class="muted">Zones: ${esc((i.zones || []).join(', ') || 'none')} · 📞 ${esc(i.phone || '')} · emergency ${esc(i.emergency_contact || '')}</div>
      <div class="muted">On-device: ${i.local_class} (${(+i.local_confidence).toFixed(2)}) → decided by <b>${esc(i.decided_by)}</b>${i.cloud_reason ? ': ' + esc(i.cloud_reason) : ''}</div>
      <div class="muted mono">⛓️ ${esc((i.chain_tx || '').slice(0, 24))}…</div>
      <div class="acts"><button class="secondary" data-act="map">Show</button>
        ${i.status === 'open' ? '<button data-act="ack">Acknowledge</button>' : ''}
        ${i.status !== 'resolved' ? '<button data-act="resolve">Resolve</button>' : '<button class="secondary" data-act="reopen">Reopen</button>'}</div>
    </div>`).join('') : '<p class="muted">No incidents.</p>';
  S.incLayer.clearLayers();
  S.incidents.filter(i => i.status !== 'resolved' && i.lat != null).forEach(i => {
    L.circleMarker([i.lat, i.lon], { radius: 11, color: '#fff', weight: 2, fillColor: COLORS[i.final_class], fillOpacity: .95 })
      .bindPopup(`<b>${i.final_class}</b> · ${esc(i.name || i.tourist_id)}<br>${esc(i.action)}`).addTo(S.incLayer);
  });
}
$('#incList').addEventListener('click', async e => {
  const b = e.target.closest('button'); if (!b) return;
  const id = +b.closest('.inc').dataset.id, inc = S.incidents.find(i => i.id === id);
  if (b.dataset.act === 'map') return S.map.setView([inc.lat, inc.lon], 17);
  try { const row = await api(`/api/incidents/${id}/${b.dataset.act}`, { method: 'POST', body: { notes: '' } }); upsert(row); }
  catch (err) { toast(err.message); }
});
function upsert(row, isNew = false) {
  const k = S.incidents.findIndex(i => i.id === row.id);
  if (k >= 0) S.incidents[k] = row; else S.incidents.unshift({ ...row, _new: isNew });
  renderIncidents();
}

/* ---------------------------------------------------------------- tourist incident reports */
async function loadReports() { S.reports = await api('/api/reports'); renderReports(); }
function upsertReport(r, isNew = false) {
  const k = S.reports.findIndex(x => x.id === r.id);
  if (k >= 0) S.reports[k] = r; else S.reports.unshift({ ...r, _new: isNew });
  S.reports.forEach(x => { if (x.id !== r.id) x._new = false; });
  renderReports();
}
function renderReports() {
  const f = $('#rFilter').value, list = S.reports.filter(r => !f || r.status === f);
  const pending = S.reports.filter(r => r.status === 'pending' || r.status === 'corroborated').length;
  $('#kRep').textContent = pending; $('#kRep').classList.toggle('hidden', !pending);
  $('#repList').innerHTML = list.length ? list.map(r => `
    <div class="rep ${r._new ? 'new' : ''}" data-id="${r.id}">
      <div class="row">${r.final_class ? `<span class="badge ${r.final_class}">${r.final_class}</span>` : ''}<b class="grow">${esc(catLabel(r.category))}</b>
        <span class="pill ${STATUS_PILL[r.status] || ''}">${esc(r.status)}</span></div>
      <div class="muted" style="margin-top:4px">${new Date(r.t).toLocaleString()} · by ${esc(r.name || r.tourist_id)} · 📞 ${esc(r.phone || '')}</div>
      ${r.description ? `<div style="margin-top:4px">“${esc(r.description)}”</div>` : ''}
      <div class="muted">${r.app_said_safe ? '⚠️ App showed this place as <b>safe</b>' : 'Inside a known hazard zone'}${r.zones?.length ? ' (' + esc(r.zones.join(', ')) + ')' : ''}
        · ${r.corroborations > 1 ? `<b>${r.corroborations} tourists</b> reported here` : '1 reporter so far'}${r.zone_id ? ' · caution zone active' : ''}</div>
      ${r.photo_file ? `<img data-photo="${r.id}" alt="Photo from the tourist">
        <div class="muted mono">photo SHA-256 ${esc(r.photo_sha256.slice(0, 24))}… (anchored on the blockchain)</div>` : '<div class="muted">No photo</div>'}
      <div class="muted mono">⛓️ ${esc((r.chain_tx || '').slice(0, 24))}…</div>
      <div class="acts"><button class="secondary" data-ract="map">Show</button>
        ${r.status !== 'verified' ? '<button data-ract="verify">Verify → caution zone</button>' : ''}
        ${r.status !== 'rejected' ? '<button class="secondary" data-ract="reject">Reject</button>' : ''}</div>
    </div>`).join('') : '<p class="muted">No reports.</p>';
  list.filter(r => r.photo_file).forEach(r => loadPhoto(r.id));
  S.repLayer.clearLayers();
  S.reports.filter(r => r.status !== 'rejected').forEach(r => L.marker([r.lat, r.lon], { icon: L.divIcon({ className: '', html: '<div class="rep-pin">!</div>', iconSize: [26, 26], iconAnchor: [13, 13] }) })
    .bindPopup(`<b>📢 ${esc(catLabel(r.category))}</b><br>${esc(r.description || '')}<br><span class="muted">${new Date(r.t).toLocaleString()} · ${esc(r.status)}</span>`).addTo(S.repLayer));
}
const catLabel = c => (S.cfg?.report_categories || {})[c] || c;
// photos need the login token, so fetch them as blobs instead of plain <img src>
async function loadPhoto(id) {
  const img = document.querySelector(`img[data-photo="${id}"]`); if (!img) return;
  if (!S.photos[id]) {
    try { const r = await fetch(`/api/reports/${id}/photo`, { headers: { Authorization: 'Bearer ' + S.token } });
          if (!r.ok) return; S.photos[id] = URL.createObjectURL(await r.blob()); } catch { return; }
  }
  img.src = S.photos[id];
}
$('#repList').addEventListener('click', async e => {
  if (e.target.dataset.photo) return window.open(e.target.src, '_blank');
  const b = e.target.closest('button'); if (!b) return;
  const id = +b.closest('.rep').dataset.id, r = S.reports.find(x => x.id === id);
  if (b.dataset.ract === 'map') return S.map.setView([r.lat, r.lon], 17);
  try { upsertReport(await api(`/api/reports/${id}/${b.dataset.ract}`, { method: 'POST' }));
        toast(b.dataset.ract === 'verify' ? 'Verified: caution zone pushed to all tourist phones.' : 'Report rejected.'); }
  catch (err) { toast(err.message); }
});

/* ---------------------------------------------------------------- tourists */
async function loadTourists() {
  S.tourists = await api('/api/tourists'); $('#kTour').textContent = S.tourists.length;
  $('#tourList').innerHTML = S.tourists.map(t => `<tr><td><b>${esc(t.name)}</b><br><span class="mono">${t.tourist_id}</span><br>${esc(t.phone)}</td>
    <td>${t.last_seen ? new Date(t.last_seen).toLocaleTimeString() : '—'}</td>
    <td>${t.last_risk ? `<span class="badge ${t.last_risk}">${t.last_risk}</span>` : '—'}</td>
    <td>${t.lat != null ? `<button class="secondary" data-loc="${t.lat},${t.lon}" style="padding:4px 8px">Map</button>` : ''}</td></tr>`).join('');
  S.tourists.forEach(t => t.lat != null && placeTourist(t.tourist_id, t.lat, t.lon, t.name, t.last_risk));
}
$('#tourList').addEventListener('click', e => { const b = e.target.closest('button[data-loc]'); if (b) S.map.setView(b.dataset.loc.split(',').map(Number), 17); });
function placeTourist(id, lat, lon, name, risk) {
  const color = COLORS[risk] || '#1565c0';
  if (S.markers[id]) S.markers[id].setLatLng([lat, lon]).setStyle({ fillColor: color });
  else S.markers[id] = L.circleMarker([lat, lon], { radius: 7, color: '#fff', weight: 2, fillColor: color, fillOpacity: 1 })
    .bindTooltip(esc(name || id)).addTo(S.tourLayer);
}

/* ---------------------------------------------------------------- live feed */
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(S.token)}`);
  S.ws = ws;
  ws.onopen = () => { $('#live').textContent = '● live'; $('#live').style.color = '#7ee08a'; S.ping = setInterval(() => ws.readyState === 1 && ws.send('ping'), 20000); };
  ws.onclose = () => { $('#live').textContent = 'reconnecting…'; $('#live').style.color = ''; clearInterval(S.ping); setTimeout(connect, 3000); };
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === 'incident') {
      upsert(m.incident, true); beep(m.incident.final_class);
      toast(`🚨 ${m.incident.final_class}: ${m.incident.name || m.incident.tourist_id} — ${m.incident.action}`);
      placeTourist(m.incident.tourist_id, m.incident.lat, m.incident.lon, m.incident.name, m.incident.final_class);
    } else if (m.type === 'incident_update') upsert(m.incident);
    else if (m.type === 'report') {
      upsertReport(m.report, true); upsert(m.incident, true); beep(m.incident.final_class);
      toast(`📢 New tourist report: ${m.incident.features?.label || m.report.category} — ${m.report.name || m.report.tourist_id}`);
    } else if (m.type === 'report_update') upsertReport(m.report);
    else if (m.type === 'position') { const t = S.tourists.find(x => x.tourist_id === m.tourist_id); placeTourist(m.tourist_id, m.lat, m.lon, t?.name, t?.last_risk); }
    else if (m.type === 'zones') { S.zones = m.zones; drawZones(); }
    else if (m.type === 'tourist') { toast(`New tourist registered: ${m.name}`); loadTourists(); }
  };
}
function beep(level) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)(), o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = level === 'CRITICAL' ? 880 : 660; o.connect(g); g.connect(ctx.destination); g.gain.value = .15;
    o.start(); o.stop(ctx.currentTime + (level === 'CRITICAL' ? .8 : .3));
  } catch { /* audio not allowed yet */ }
}

/* ---------------------------------------------------------------- zones */
async function loadZones() { S.zones = await api('/api/zones'); drawZones(); }
function drawZones() {
  S.zoneLayer.clearLayers();
  S.zones.forEach(z => {
    const st = { color: z.hazard ? (z.expires ? '#ef6c00' : '#c62828') : '#2e7d32', weight: 2, fillOpacity: .15, dashArray: z.expires ? '6 6' : null }, g = z.geometry;
    (z.kind === 'circle' ? L.circle([g.lat, g.lon], { radius: g.radius_m, ...st }) : L.polygon(g.vertices, st))
      .bindTooltip((z.hazard ? '⚠ ' : '✓ ') + esc(z.name)).addTo(S.zoneLayer);
  });
  $('#zList').innerHTML = S.zones.map(z => `<tr><td>${z.hazard ? '⚠️' : '✅'} ${esc(z.name)}</td><td class="muted">${z.kind}${z.geometry.radius_m ? ' · ' + z.geometry.radius_m + ' m' : ''}${z.expires ? `<br>${z.source === 'verified' ? 'police-verified' : 'from tourist reports'} · until ${new Date(z.expires.replace(' ', 'T') + 'Z').toLocaleString()}` : ''}</td>
    <td><button class="secondary" data-del="${z.id}" style="padding:4px 8px">Delete</button></td></tr>`).join('');
}
$('#zRadius').oninput = () => S.pickMarker && S.pickMarker.setRadius(+$('#zRadius').value);
$('#zSave').onclick = async () => {
  if (!S.pick || !$('#zName').value.trim()) return toast('Pick a location on the map and give the zone a name.');
  try {
    await api('/api/zones', { method: 'POST', body: { name: $('#zName').value.trim(), kind: 'circle', lat: S.pick.lat, lon: S.pick.lng,
      radius_m: +$('#zRadius').value, hazard: $('#zHazard').value === '1' } });
    toast('Zone saved. Tourist phones pick it up on their next sync.'); $('#zName').value = ''; S.pick = null; $('#zSave').disabled = true;
    if (S.pickMarker) S.pickMarker.remove(); await loadZones();
  } catch (e) { toast(e.message); }
};
$('#zList').addEventListener('click', async e => { const b = e.target.closest('button[data-del]'); if (b) { await api('/api/zones/' + b.dataset.del, { method: 'DELETE' }); loadZones(); } });

/* ---------------------------------------------------------------- digital ID verification */
$('#scanBtn').onclick = async () => {
  try {
    S.scanner = new Html5Qrcode('reader');
    await S.scanner.start({ facingMode: 'environment' }, { fps: 10, qrbox: 240 }, txt => { $('#credText').value = txt; stopScan(); verify(); });
  } catch (e) { toast('Camera not available: ' + e); }
};
async function stopScan() { if (S.scanner) { try { await S.scanner.stop(); } catch { /* */ } S.scanner = null; } }
$('#stopScan').onclick = stopScan;
$('#verifyBtn').onclick = verify;
async function verify() {
  try {
    const r = await api('/api/verify', { method: 'POST', body: { credential: $('#credText').value.trim() } });
    $('#verifyOut').innerHTML = `<div class="card" style="border-left:6px solid ${r.valid ? '#2e7d32' : '#c62828'}">
      <h3>${r.valid ? '✅ VALID' : '❌ REJECTED'}</h3><div>${esc(r.reason)}</div>
      <div style="margin-top:6px"><b>${esc(r.name)}</b> (${esc(r.nationality)}) · <span class="mono">${r.tourist_id}</span><br>
      Valid ${r.valid_from} → ${r.valid_to} · on-chain block ${r.block ?? '—'}<br>📞 ${esc(r.phone || '—')} · emergency ${esc(r.emergency_contact || '—')}</div>
      <div class="muted" style="margin-top:6px">Incidents on the blockchain: ${r.incidents.length}</div>
      ${r.valid ? `<button class="danger" id="revokeBtn" data-id="${r.tourist_id}" style="margin-top:8px">Revoke this ID</button>` : ''}</div>`;
  } catch (e) { $('#verifyOut').innerHTML = `<div class="card">❌ ${esc(e.message)}</div>`; }
}
$('#verifyOut').addEventListener('click', async e => {
  if (e.target.id !== 'revokeBtn') return;
  if (!e.target.dataset.armed) { e.target.dataset.armed = 1; e.target.textContent = 'Click again to confirm revocation'; return; }
  await api(`/api/tourists/${e.target.dataset.id}/revoke`, { method: 'POST', body: { reason: 'revoked by officer' } }); toast('ID revoked on the blockchain.'); verify();
});

/* ---------------------------------------------------------------- blockchain */
async function loadChain() {
  const c = await api('/api/chain');
  $('#chainOut').innerHTML = `<div class="card" style="border-left:6px solid ${c.intact ? '#2e7d32' : '#c62828'}">
      <b>${c.intact ? '✅ Chain intact' : '❌ Integrity problems'}</b> — ${c.height} blocks, ${c.transactions} transactions<br>
      <span class="muted">Authorities: ${c.authorities.map(esc).join(', ')}</span>${c.problems.map(p => '<br>⚠️ ' + esc(p)).join('')}</div>
    <table><thead><tr><th>#</th><th>Validator</th><th>Transactions</th><th>Hash</th></tr></thead><tbody>
    ${c.blocks.map(b => `<tr><td>${b.index}</td><td>${esc(b.validator)}</td><td>${b.transactions.map(t => t.kind).join(', ') || 'genesis'}</td>
      <td class="mono">${b.hash.slice(0, 16)}…</td></tr>`).join('')}</tbody></table>`;
}

boot();
