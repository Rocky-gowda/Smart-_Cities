/* On-device engine for the tourist app (runs in the phone's browser, works offline).
 * A line-by-line port of src/geofence.py (Stage 1) and the deployed logistic-regression
 * model (Stage 2a). tests/test_js_parity.py checks that it gives the same answers as Python.
 */
(function (root) {
  'use strict';
  const R = 6371008.8;
  const CLASSES = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
  const rad = d => d * Math.PI / 180;

  function haversine(lat1, lon1, lat2, lon2) {
    const p1 = rad(lat1), p2 = rad(lat2), dp = p2 - p1, dl = rad(lon2 - lon1);
    const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(a));
  }

  function toLocal(lat, lon, lat0, lon0) {
    return [rad(lon - lon0) * R * Math.cos(rad(lat0)), rad(lat - lat0) * R];
  }

  function pointSegment(p, a, b) {
    const [px, py] = toLocal(p[0], p[1], a[0], a[1]);
    const [bx, by] = toLocal(b[0], b[1], a[0], a[1]);
    const seg2 = bx * bx + by * by;
    const t = seg2 === 0 ? 0 : Math.max(0, Math.min(1, (px * bx + py * by) / seg2));
    return Math.hypot(px - t * bx, py - t * by);
  }

  function distanceToRoute(p, route) {
    if (route.length === 1) return haversine(p[0], p[1], route[0][0], route[0][1]);
    let best = Infinity;
    for (let i = 0; i < route.length - 1; i++) best = Math.min(best, pointSegment(p, route[i], route[i + 1]));
    return best;
  }

  function pointInPolygon(p, poly) {
    const [lat, lon] = p;
    let inside = false;
    for (let i = 0, n = poly.length; i < n; i++) {
      const [y1, x1] = poly[i], [y2, x2] = poly[(i + 1) % n];
      if ((y1 > lat) !== (y2 > lat)) {
        const xc = x1 + (lat - y1) * (x2 - x1) / (y2 - y1);
        if (lon < xc) inside = !inside;
      }
    }
    return inside;
  }

  function zoneContains(z, p) {
    const g = z.geometry;
    return z.kind === 'circle' ? haversine(g.lat, g.lon, p[0], p[1]) <= g.radius_m : pointInPolygon(p, g.vertices);
  }

  function zonesAt(zones, p) {
    const inside = zones.filter(z => zoneContains(z, p));
    return { hazard: inside.some(z => z.hazard) ? 1 : 0, names: inside.map(z => z.name) };
  }

  /* Stage 1: one event per idle episode once the tourist has stayed within `radius` metres for
   * `idleMinutes` (time multiplied by `timeScale`, for accelerated demonstrations). */
  class GeofenceGate {
    constructor(route, zones, opts = {}) {
      this.route = route; this.zones = zones;
      this.idleMinutes = opts.idleMinutes ?? 15; this.radius = opts.radius ?? 30; this.timeScale = opts.timeScale ?? 1;
      this.anchor = null; this.fired = false;
    }
    update(fix) {   // fix = {t: Date|ms|ISO, lat, lon}
      const t = new Date(fix.t).getTime();
      if (!this.anchor || haversine(this.anchor.lat, this.anchor.lon, fix.lat, fix.lon) > this.radius) {
        this.anchor = { t, lat: fix.lat, lon: fix.lon }; this.fired = false; return null;
      }
      const idle = (t - this.anchor.t) / 60000 * this.timeScale;
      if (idle >= this.idleMinutes && !this.fired) {
        this.fired = true;
        const d = new Date(t), z = zonesAt(this.zones, [fix.lat, fix.lon]);
        return { t: d.toISOString(), lat: fix.lat, lon: fix.lon, idle_min: idle,
                 hour: d.getHours() + d.getMinutes() / 60,
                 displacement_m: distanceToRoute([fix.lat, fix.lon], this.route),
                 hazard_flag: z.hazard, zones: z.names };
      }
      return null;
    }
    idleNow(tMs) { return this.anchor ? (tMs - this.anchor.t) / 60000 * this.timeScale : 0; }
  }

  function features(ev) {
    return [Math.log(Math.max(ev.displacement_m, 20)), Math.log(Math.max(ev.idle_min, 1e-3)),
            Math.sin(2 * Math.PI * ev.hour / 24), Math.cos(2 * Math.PI * ev.hour / 24), ev.hazard_flag ? 1 : 0];
  }

  /* Stage 2a: multinomial logistic regression, p = softmax(W (x - mean)/scale + b) */
  function predict(model, x) {
    const z = model.coef.map((w, k) => model.intercept[k] +
      w.reduce((s, wi, i) => s + wi * (x[i] - model.scaler_mean[i]) / model.scaler_scale[i], 0));
    const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);
    const p = e.map(v => v / sum);
    const k = p.indexOf(Math.max(...p));
    return { probabilities: p, index: k, label: CLASSES[k], confidence: p[k] };
  }

  function escalationDecision(pred, theta, policy) {
    const reasons = [];
    if (pred.confidence < theta) reasons.push(`confidence ${pred.confidence.toFixed(2)} < θc ${theta}`);
    if (policy === 'safety' && pred.index >= 2) reasons.push(`${pred.label} needs verification`);
    return { escalate: reasons.length > 0, reasons };
  }

  const ACTIONS = { LOW: 'Log only', MODERATE: 'Safety check-in requested',
                    HIGH: 'Nearest tourist-police unit notified', CRITICAL: 'Responders dispatched, emergency contact alerted' };

  const API = { haversine, distanceToRoute, pointInPolygon, zoneContains, zonesAt, GeofenceGate, features,
                predict, escalationDecision, CLASSES, ACTIONS };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.Geo = API;
})(typeof self !== 'undefined' ? self : this);
