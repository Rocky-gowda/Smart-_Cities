/* Service worker: lets the tourist app open and run without a network connection. */
const CACHE = 'tsm-v2';
const SHELL = ['./', './index.html', './app.js', './manifest.json', './icon-192.png', './icon-512.png',
  '/static/app.css', '/static/geo.js', '/static/vendor/leaflet/leaflet.js', '/static/vendor/leaflet/leaflet.css'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.pathname.startsWith('/api/')) return;      // API calls always go to the network
  if (url.hostname.endsWith('tile.openstreetmap.org')) {                          // keep map tiles you have seen
    e.respondWith(caches.open('tsm-tiles').then(async c => (await c.match(e.request)) ||
      fetch(e.request).then(r => { c.put(e.request, r.clone()); return r; })));
    return;
  }
  e.respondWith(fetch(e.request).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request)));
});
