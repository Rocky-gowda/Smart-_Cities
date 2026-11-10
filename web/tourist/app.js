/* Tourist app: registration, real GPS tracking, on-device geofence + AI, offline queue, SOS. */
'use strict';
const $ = s => document.querySelector(s);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};
const S = {
  token: store.get('tsm.token', null), profile: store.get('tsm.profile', null), config: store.get('tsm.config', null),
  queue: store.get('tsm.queue', { fixes: [], events: [], sos: [] }), localAlerts: store.get('tsm.alerts', []),
  myReports: store.get('tsm.reports', []), nearby: store.get('tsm.nearby', []), warned: {},
  watch: null, tick: null, gate: null, lastFix: null, lastSent: null, online: navigator.onLine, map: null, layers: {},
};
if (!S.queue.reports) S.queue.reports = [];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const ago = t => { const m = Math.max(0, Math.round((Date.now() - Date.parse(t)) / 60000)); return m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`; };
const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : 'e' + Date.now() + Math.random().toString(16).slice(2));
const saveQueue = () => { store.set('tsm.queue', S.queue); const n = S.queue.events.length + S.queue.sos.length + S.queue.reports.length; $('#queue').textContent = S.queue.fixes.length ? `${n} alerts · ${S.queue.fixes.length} fixes` : String(n); };
function toast(msg, ms = 4000) { const t = $('#toast'); t.textContent = msg; t.classList.remove('hidden'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add('hidden'), ms); }

async function api(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(S.token ? { Authorization: 'Bearer ' + S.token } : {}) };
  const r = await fetch(path, { ...opts, headers, body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch { /* */ } const e = new Error(m); e.status = r.status; throw e; }
  return r.json();
}

/* ---------------------------------------------------------------- startup */
async function loadConfig() {
  try { S.config = await api('/api/config'); store.set('tsm.config', S.config); setNet(true); }
  catch { setNet(false); if (!S.config) toast('Cannot reach the server and no saved settings yet.'); }
}
function setNet(ok) { S.online = ok; const n = $('#net'); n.textContent = ok ? 'online' : 'offline'; n.className = 'pill ' + (ok ? 'ok' : 'warn'); }
window.addEventListener('online', () => { flush(); });
window.addEventListener('offline', () => setNet(false));

async function init() {
  await loadConfig();
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
  if (S.token && S.profile) showApp(); else showRegister();
  setInterval(flush, 10000);
}

/* ---------------------------------------------------------------- registration */
function showRegister() {
  $('#register').classList.remove('hidden');
  const spots = $('#spots');
  Object.keys(S.config?.spots || {}).forEach((name, i) => {
    const l = document.createElement('label');
    l.innerHTML = `<input type="checkbox" value="${name}" ${i < 2 ? 'checked' : ''}> ${name}`;
    spots.appendChild(l);
  });
  const f = $('#regForm'), today = new Date();
  f.valid_from.value = today.toISOString().slice(0, 10);
  f.valid_to.value = new Date(today.getTime() + 7 * 864e5).toISOString().slice(0, 10);
  f.addEventListener('submit', async e => {
    e.preventDefault();
    $('#regBtn').disabled = true; $('#regMsg').textContent = 'Issuing your digital ID on the blockchain…';
    const body = Object.fromEntries(new FormData(f).entries());
    body.itinerary = [...spots.querySelectorAll('input[value]:checked')].map(i => i.value);
    if ($('#useStart').checked) {
      try { const p = await currentPosition(); body.start = { lat: p.coords.latitude, lon: p.coords.longitude }; }
      catch { toast('Location not available — using the selected spots only.'); }
    }
    try {
      const r = await api('/api/tourists/register', { method: 'POST', body });
      S.token = r.token; S.profile = r; store.set('tsm.token', r.token); store.set('tsm.profile', r);
      $('#register').classList.add('hidden'); showApp(); toast(`Digital ID ${r.tourist_id} issued`);
    } catch (err) { $('#regMsg').textContent = 'Registration failed: ' + err.message; }
    $('#regBtn').disabled = false;
  });
}
const currentPosition = (timeout = 15000) => new Promise((ok, fail) => navigator.geolocation
  ? navigator.geolocation.getCurrentPosition(ok, fail, { enableHighAccuracy: true, timeout, maximumAge: 60000 }) : fail(new Error('no GPS')));

/* ---------------------------------------------------------------- main app */
function showApp() {
  $('#nav').classList.remove('hidden');
  document.querySelectorAll('#nav button').forEach(b => b.onclick = () => view(b.dataset.view));
  view('home'); renderId(); renderAlerts(); renderMyReports(); initMap(); saveQueue();
  refreshProfile(); loadNearby();
}
function view(v) {
  ['home', 'idview', 'alerts'].forEach(id => $('#' + id).classList.toggle('hidden', id !== v));
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b.dataset.view === v));
  if (v === 'home' && S.map) setTimeout(() => S.map.invalidateSize(), 50);
}
async function refreshProfile() {
  try {
    const me = await api('/api/tourists/me');
    S.profile = { ...S.profile, ...me }; store.set('tsm.profile', S.profile); renderId(); renderAlerts(me.incidents); setNet(true);
    renderMyReports(me.reports || []);
  } catch (e) { if (e.status === 401) { toast('This ID is no longer valid on the server. Please register again.'); } }
}
function renderId() {
  const p = S.profile, c = p.credential || {};
  $('#idName').textContent = c.name || p.name || '';
  $('#idMeta').innerHTML = `<b>${c.tourist_id || p.tourist_id}</b><br>Nationality ${c.nationality || ''} · valid ${c.valid_from} → ${c.valid_to}<br>Issued by ${c.issuer || ''}`;
  if (p.qr_png) $('#qr').src = 'data:image/png;base64,' + p.qr_png;
  $('#idChain').innerHTML = `Recorded in blockchain block <b>${p.block ?? '…'}</b>${p.revoked ? ' · <span class="pill bad">REVOKED</span>' : ' · <span class="pill ok">active</span>'}<br>Itinerary: ${(p.itinerary || []).join(' → ') || '—'}`;
}
function renderAlerts(server = []) {
  const items = [...S.localAlerts.filter(a => !server.some(s => s.client_event_id === a.client_event_id)), ...server]
    .sort((a, b) => String(b.t).localeCompare(String(a.t)));
  $('#alertList').innerHTML = items.length ? items.map(a => `<div class="alert-item"><span class="badge ${a.final_class}">${a.final_class}</span>
    <b> ${new Date(a.t).toLocaleString()}</b><br><span class="muted">${a.action || ''} · ${a.decided_by || ''}${a.pending ? ' · waiting for network' : ''}${a.status ? ' · ' + a.status : ''}</span></div>`).join('') : 'No alerts yet.';
}
document.addEventListener('click', e => { if (e.target.id === 'logout') { if (S.armedLogout) { ['tsm.token', 'tsm.profile', 'tsm.alerts'].forEach(store.del); location.reload(); } else { S.armedLogout = true; e.target.textContent = 'Tap again to confirm'; } } });

/* ---------------------------------------------------------------- map */
function initMap() {
  if (S.map) return;
  const route = S.profile.route || [];
  S.map = L.map('map', { zoomControl: true }).setView(route[0] || [12.9716, 77.5946], 15);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(S.map);
  if (route.length) L.polyline(route, { color: '#1f4e79', weight: 5, opacity: .8 }).addTo(S.map).bindTooltip('Planned route');
  drawZones(); drawNearby();
  S.layers.track = L.polyline([], { color: '#424242', weight: 2, dashArray: '4 6' }).addTo(S.map);
}
function drawZones() {
  if (S.layers.zones) S.layers.zones.remove();
  S.layers.zones = L.layerGroup().addTo(S.map);
  (S.config?.zones || []).forEach(z => {
    const style = { color: z.hazard ? (z.expires ? '#ef6c00' : '#c62828') : '#2e7d32', weight: 2, fillOpacity: .15, dashArray: z.expires ? '6 6' : null };
    const g = z.geometry, label = (z.hazard ? '⚠ ' : '✓ ') + z.name;
    (z.kind === 'circle' ? L.circle([g.lat, g.lon], { radius: g.radius_m, ...style }) : L.polygon(g.vertices, style))
      .bindTooltip(label).addTo(S.layers.zones);
  });
}
function showMe(fix) {
  if (!S.map) return;
  const ll = [fix.lat, fix.lon];
  if (!S.layers.me) {
    S.layers.me = L.circleMarker(ll, { radius: 9, color: '#fff', weight: 3, fillColor: '#1565c0', fillOpacity: 1 }).addTo(S.map).bindTooltip('You');
    S.layers.acc = L.circle(ll, { radius: fix.accuracy || 20, color: '#1565c0', weight: 1, fillOpacity: .08 }).addTo(S.map);
    S.map.setView(ll, 16);
  } else { S.layers.me.setLatLng(ll); S.layers.acc.setLatLng(ll).setRadius(fix.accuracy || 20);
    if (!S.map.getBounds().pad(-0.2).contains(ll)) S.map.panTo(ll); }
  S.layers.track.addLatLng(ll);
}

/* ---------------------------------------------------------------- tracking (Stage 1 on the phone) */
function newGate() {
  return new Geo.GeofenceGate(S.profile.route, S.config.zones, { idleMinutes: S.config.idle_minutes, timeScale: S.config.time_scale });
}
$('#trackBtn').onclick = () => (S.watch !== null ? stopTracking() : startTracking());
function startTracking() {
  if (!navigator.geolocation) return toast('This browser has no GPS access.');
  S.gate = newGate();
  S.watch = navigator.geolocation.watchPosition(onPosition, err => {
    toast(err.code === 1 ? 'Location permission denied. Allow location for this site.' : 'GPS error: ' + err.message);
    if (err.code === 1) stopTracking();
  }, { enableHighAccuracy: true, maximumAge: 5000, timeout: 30000 });
  // phones stop sending updates when you stand still, so re-check the last position every 15 s
  S.tick = setInterval(() => { if (S.lastFix) feed({ ...S.lastFix, t: new Date().toISOString() }, false); }, 15000);
  $('#trackBtn').textContent = 'Stop tracking'; $('#trackState').textContent = 'Tracking is on';
  $('#trackBtn').classList.add('secondary');
}
function stopTracking() {
  if (S.watch !== null) navigator.geolocation.clearWatch(S.watch);
  clearInterval(S.tick); S.watch = null;
  $('#trackBtn').textContent = 'Start tracking'; $('#trackState').textContent = 'Tracking is off'; $('#trackBtn').classList.remove('secondary');
}
function onPosition(pos) {
  const fix = { t: new Date(pos.timestamp || Date.now()).toISOString(), lat: pos.coords.latitude, lon: pos.coords.longitude,
                accuracy: Math.round(pos.coords.accuracy) };
  S.lastFix = fix; feed(fix, true);
}
function feed(fix, record, gate = S.gate) {
  if (record) {
    const last = S.queue.fixes[S.queue.fixes.length - 1] || S.lastSent;
    if (!last || Date.parse(fix.t) - Date.parse(last.t) >= 10000 || Geo.haversine(last.lat, last.lon, fix.lat, fix.lon) > 10) {
      S.queue.fixes.push(fix); if (S.queue.fixes.length > 2000) S.queue.fixes.shift(); saveQueue();
    }
    showMe(fix);
  }
  const ev = gate.update(fix);
  const p = [fix.lat, fix.lon], z = Geo.zonesAt(S.config.zones, p);
  $('#acc').textContent = fix.accuracy ? `±${fix.accuracy} m` : '–';
  $('#idle').textContent = `${Math.floor(gate.idleNow(Date.parse(fix.t)))} min`;
  $('#dev').textContent = `${Math.round(Geo.distanceToRoute(p, S.profile.route))} m`;
  $('#zoneNote').innerHTML = z.names.length ? `${z.hazard ? '⚠️ You are in a hazard zone' : '✓ Safe zone'}: ${z.names.join(', ')}` : '';
  if (record) warnNearby(fix);
  if (ev) onEvent(ev);
  if (record && S.queue.fixes.length >= 5) flush();
}

/* ---------------------------------------------------------------- Stage 2: on-device AI, then the server */
async function onEvent(ev) {
  const pred = Geo.predict(S.config.model, Geo.features(ev));
  const dec = Geo.escalationDecision(pred, S.config.theta, S.config.policy);
  const item = { ...ev, client_event_id: uuid(), local_class: pred.label, local_confidence: pred.confidence };
  showDecision({ final_class: pred.label, decided_by: 'on-device AI', action: Geo.ACTIONS[pred.label],
                 note: dec.escalate ? 'Checking with the cloud…' : 'Decided on your phone' });
  const local = { client_event_id: item.client_event_id, t: ev.t, final_class: pred.label, action: Geo.ACTIONS[pred.label], decided_by: 'on-device', pending: true };
  S.localAlerts.unshift(local); store.set('tsm.alerts', S.localAlerts.slice(0, 50)); renderAlerts();
  try {
    const r = await api('/api/events', { method: 'POST', body: { events: [item] } });
    const d = r.decisions[0]; setNet(true);
    Object.assign(local, d, { pending: false }); store.set('tsm.alerts', S.localAlerts.slice(0, 50)); renderAlerts();
    showDecision({ ...d, note: d.escalated ? `Verified by ${d.decided_by}` : 'Decided on your phone; police informed' });
  } catch {
    S.queue.events.push({ ...item, offline: true }); saveQueue(); setNet(false);
    showDecision({ final_class: pred.label, decided_by: 'on-device AI (offline)', action: Geo.ACTIONS[pred.label],
                   note: 'No network: saved on your phone and will be sent automatically.' });
  }
}
function showDecision(d) {
  const el = $('#decision'); el.className = 'card decision ' + d.final_class;
  el.innerHTML = `<div class="row"><span class="badge ${d.final_class}">${d.final_class}</span><b class="grow">${d.action || ''}</b></div>
    <div class="muted" style="margin-top:6px">${d.decided_by || ''} · ${d.note || ''}</div>`;
  if (navigator.vibrate && (d.final_class === 'HIGH' || d.final_class === 'CRITICAL')) navigator.vibrate([300, 100, 300]);
}

/* ---------------------------------------------------------------- sync (offline queue) */
let flushing = false;
async function flush() {
  if (flushing || !S.token) return; flushing = true;
  try {
    await api('/api/health'); setNet(true);
    if (!S.cfgAt || Date.now() - S.cfgAt > 60000) {          // pick up new hazard zones from the police
      const cfg = await api('/api/config'); S.cfgAt = Date.now();
      if (JSON.stringify(cfg.zones) !== JSON.stringify(S.config.zones)) {
        S.config = cfg; store.set('tsm.config', cfg); if (S.gate) S.gate.zones = cfg.zones; if (S.map) drawZones();
      }
    }
    if (!S.nearAt || Date.now() - S.nearAt > 60000) await loadNearby();
    while (S.queue.reports.length) { await sendReport(S.queue.reports[0], true); S.queue.reports.shift(); toast('Your saved incident report has been sent to the police.'); }
    if (S.queue.fixes.length) {
      const batch = S.queue.fixes.slice(0, 200);
      await api('/api/track', { method: 'POST', body: { fixes: batch } });
      S.lastSent = batch[batch.length - 1]; S.queue.fixes.splice(0, batch.length);
    }
    if (S.queue.events.length) {
      const r = await api('/api/events', { method: 'POST', body: { events: S.queue.events } });
      r.decisions.forEach((d, i) => { const a = S.localAlerts.find(x => x.client_event_id === S.queue.events[i].client_event_id); if (a) Object.assign(a, d, { pending: false }); });
      S.queue.events = []; store.set('tsm.alerts', S.localAlerts); renderAlerts(); toast('Saved alerts sent to the police.');
    }
    while (S.queue.sos.length) { await api('/api/sos', { method: 'POST', body: S.queue.sos[0] }); S.queue.sos.shift(); toast('Your SOS has been delivered.'); }
  } catch { setNet(false); }
  saveQueue(); flushing = false;
}

/* ---------------------------------------------------------------- SOS */
$('#sos').onclick = async () => {
  const b = $('#sos');
  if (!b.classList.contains('armed')) { b.classList.add('armed'); b.textContent = 'TAP AGAIN TO SEND SOS'; setTimeout(() => { b.classList.remove('armed'); b.textContent = 'SOS — EMERGENCY'; }, 4000); return; }
  b.classList.remove('armed'); b.textContent = 'SOS — EMERGENCY';
  showDecision({ final_class: 'CRITICAL', decided_by: 'SOS', action: Geo.ACTIONS.CRITICAL, note: 'Getting your location…' });
  let fix = S.lastFix, approx = false;
  if (!fix) {
    try { const p = await currentPosition(5000); fix = { lat: p.coords.latitude, lon: p.coords.longitude }; }
    catch { const r = S.profile.route[0]; fix = r ? { lat: r[0], lon: r[1] } : null; approx = true; }
  }
  if (!fix) return toast('No location available for the SOS.');
  const sos = { client_event_id: uuid(), t: new Date().toISOString(), lat: fix.lat, lon: fix.lon,
                message: approx ? 'SOS from tourist app (location approximate: GPS unavailable)' : 'SOS from tourist app' };
  showDecision({ final_class: 'CRITICAL', decided_by: 'SOS', action: Geo.ACTIONS.CRITICAL, note: 'Sending…' });
  try { await api('/api/sos', { method: 'POST', body: sos }); showDecision({ final_class: 'CRITICAL', decided_by: 'SOS', action: Geo.ACTIONS.CRITICAL, note: 'Police have received your SOS.' }); }
  catch { S.queue.sos.push({ ...sos, offline: true }); saveQueue(); showDecision({ final_class: 'CRITICAL', decided_by: 'SOS', action: 'Saved — will be sent when the network returns', note: 'Call 112 if you can.' }); }
};

/* ---------------------------------------------------------------- incident reports (tourist -> police + other tourists) */
$('#reportBtn').onclick = () => {
  const sel = $('#rCat');
  if (!sel.options.length) Object.entries(S.config?.report_categories || {}).forEach(([k, v]) => sel.add(new Option(v, k)));
  $('#reportCard').classList.remove('hidden'); $('#reportBtn').classList.add('hidden');
  const z = S.lastFix ? Geo.zonesAt(S.config.zones, [S.lastFix.lat, S.lastFix.lon]) : null;
  $('#rLoc').textContent = S.lastFix ? `Location: ${S.lastFix.lat.toFixed(5)}, ${S.lastFix.lon.toFixed(5)} (±${S.lastFix.accuracy || '?'} m)${z && !z.hazard ? ' · the app shows this place as safe — your report will warn others.' : ''}`
                                     : 'Your current location will be attached when you send.';
  $('#reportCard').scrollIntoView({ behavior: 'smooth' });
};
function closeReport() { $('#reportForm').reset(); $('#rPreview').classList.add('hidden'); S.photo = null; $('#reportCard').classList.add('hidden'); $('#reportBtn').classList.remove('hidden'); }
$('#rCancel').onclick = closeReport;
$('#rPhoto').onchange = async e => {
  const f = e.target.files[0]; S.photo = null; if (!f) return $('#rPreview').classList.add('hidden');
  try { S.photo = await shrinkPhoto(f); $('#rPreview').src = S.photo; $('#rPreview').classList.remove('hidden'); }
  catch { toast('Could not read that photo.'); }
};
// phone photos are large: resize to at most 1280 px and re-encode as JPEG (~150–400 KB) before sending
function shrinkPhoto(file, max = 1280) {
  return new Promise((ok, fail) => {
    const img = new Image(), url = URL.createObjectURL(file);
    img.onload = () => { const k = Math.min(1, max / Math.max(img.width, img.height)), c = document.createElement('canvas');
      c.width = Math.round(img.width * k); c.height = Math.round(img.height * k); c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url); ok(c.toDataURL('image/jpeg', 0.75)); };
    img.onerror = () => { URL.revokeObjectURL(url); fail(new Error('bad image')); }; img.src = url;
  });
}
$('#reportForm').onsubmit = async e => {
  e.preventDefault(); $('#rSend').disabled = true;
  let fix = S.lastFix;
  if (!fix) { try { const p = await currentPosition(8000); fix = { lat: p.coords.latitude, lon: p.coords.longitude }; } catch { /* below */ } }
  if (!fix) { $('#rSend').disabled = false; return toast('Location is needed for a report. Turn on GPS / allow location.'); }
  const f = new FormData(e.target);
  const rep = { client_report_id: uuid(), t: new Date().toISOString(), lat: fix.lat, lon: fix.lon,
                category: f.get('category'), description: f.get('description') || '', photo: S.photo || null };
  const mine = { client_report_id: rep.client_report_id, t: rep.t, category: rep.category, description: rep.description, status: 'sending…' };
  S.myReports.unshift(mine); store.set('tsm.reports', S.myReports.slice(0, 30)); renderMyReports();
  closeReport();
  try { await sendReport(rep); }
  catch { S.queue.reports.push({ ...rep, offline: true }); saveQueue(); setNet(false); mine.status = 'saved — will be sent when the network returns';
          store.set('tsm.reports', S.myReports.slice(0, 30)); renderMyReports(); toast('No network: report saved on your phone and will be sent automatically.'); }
  $('#rSend').disabled = false;
};
async function sendReport(rep, queued = false) {
  const r = await api('/api/reports', { method: 'POST', body: rep }); setNet(true);
  const mine = S.myReports.find(x => x.client_report_id === rep.client_report_id);
  if (mine) Object.assign(mine, { status: r.status, severity: r.severity, report_id: r.report_id, chain_tx: r.chain_tx, caution_zone: r.caution_zone });
  store.set('tsm.reports', S.myReports.slice(0, 30)); renderMyReports();
  if (!queued) toast(r.caution_zone ? 'Report sent to the police. Other tourists reported this too — the place is now a caution zone.' : 'Report sent to the police and shared as a warning with nearby tourists. Thank you.', 6000);
  loadNearby(); S.cfgAt = 0;
  return r;
}
function renderMyReports(server) {
  if (server) { server.forEach(s => { const m = S.myReports.find(x => x.client_report_id === s.client_report_id);
    if (m) Object.assign(m, { status: s.status, severity: s.final_class }); else S.myReports.push({ ...s, severity: s.final_class }); });
    S.myReports.sort((a, b) => String(b.t).localeCompare(String(a.t))); store.set('tsm.reports', S.myReports.slice(0, 30)); }
  const names = S.config?.report_categories || {};
  $('#reportList').innerHTML = S.myReports.length ? S.myReports.map(r => `<div class="alert-item">${r.severity ? `<span class="badge ${r.severity}">${r.severity}</span> ` : ''}<b>${esc(names[r.category] || r.category)}</b>
    <br><span class="muted">${new Date(r.t).toLocaleString()} · ${esc(r.status)}${r.description ? ' · ' + esc(r.description) : ''}</span></div>`).join('') : 'You have not reported anything.';
}
async function loadNearby() {
  try { S.nearby = await api('/api/reports/nearby'); S.nearAt = Date.now(); store.set('tsm.nearby', S.nearby); drawNearby(); } catch { /* offline: keep the saved list */ }
}
function drawNearby() {
  if (!S.map) return;
  if (S.layers.reports) S.layers.reports.remove();
  S.layers.reports = L.layerGroup().addTo(S.map);
  S.nearby.forEach(r => L.marker([r.lat, r.lon], { icon: L.divIcon({ className: '', html: `<div class="rep-pin ${r.severity}">!</div>`, iconSize: [26, 26], iconAnchor: [13, 13] }) })
    .bindPopup(`<b>${esc(r.label)}</b><br>Reported ${ago(r.t)}${r.corroborations > 1 ? ` by ${r.corroborations} tourists` : ''}${r.status === 'verified' ? ' · <b>confirmed by police</b>' : ''}${r.description ? '<br>' + esc(r.description) : ''}`)
    .addTo(S.layers.reports));
}
function warnNearby(fix) {
  const near = S.nearby.map(r => ({ ...r, d: Geo.haversine(fix.lat, fix.lon, r.lat, r.lon) })).filter(r => r.d <= 300).sort((a, b) => a.d - b.d);
  const box = $('#nearWarn');
  if (!near.length) return box.classList.add('hidden');
  box.classList.remove('hidden');
  box.innerHTML = `<b>⚠️ Reported by tourists near you</b>` + near.slice(0, 3).map(r => `<div class="muted">${esc(r.label)} · ${Math.round(r.d)} m away · ${ago(r.t)}${r.status === 'verified' ? ' · confirmed by police' : ''}</div>`).join('') + `<div class="muted">Stay alert and keep to busy, well-lit roads.</div>`;
  const fresh = near.filter(r => !S.warned[r.id]);
  if (fresh.length) { fresh.forEach(r => S.warned[r.id] = 1); toast(`⚠️ Careful: ${fresh[0].label} reported ${Math.round(fresh[0].d)} m from you.`, 6000); if (navigator.vibrate) navigator.vibrate([200, 100, 200]); }
}

/* ---------------------------------------------------------------- demonstration: simulated stop in a hazard zone */
$('#simBtn').onclick = async () => {
  const hazards = (S.config.zones || []).filter(z => z.hazard && z.kind === 'circle');
  if (!hazards.length) return toast('No circular hazard zone defined.');
  const start = S.lastFix ? [S.lastFix.lat, S.lastFix.lon] : S.profile.route[0];
  const z = hazards.sort((a, b) => Geo.haversine(start[0], start[1], a.geometry.lat, a.geometry.lon) - Geo.haversine(start[0], start[1], b.geometry.lat, b.geometry.lon))[0];
  const target = [z.geometry.lat, z.geometry.lon], steps = 8, gate = new Geo.GeofenceGate(S.profile.route, S.config.zones, { idleMinutes: S.config.idle_minutes, timeScale: 1 });
  const t0 = Date.now(); let n = 0;
  $('#simBtn').disabled = true; toast(`Simulating a stop in “${z.name}”…`);
  const fixes = [];
  for (let i = 0; i <= steps; i++) fixes.push([start[0] + (target[0] - start[0]) * i / steps, start[1] + (target[1] - start[1]) * i / steps]);
  for (let i = 0; i <= Math.ceil(S.config.idle_minutes) + 1; i++) fixes.push([target[0] + (Math.random() - .5) * 1e-4, target[1] + (Math.random() - .5) * 1e-4]);
  for (const [lat, lon] of fixes) {
    const fix = { t: new Date(t0 + 60000 * n++).toISOString(), lat, lon, accuracy: 8 };
    S.queue.fixes.push(fix); showMe(fix); feed(fix, false, gate);
    await new Promise(r => setTimeout(r, 120));
  }
  saveQueue(); flush(); $('#simBtn').disabled = false;
};

init();
