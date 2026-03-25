const TILE_CACHE_NAME = 'swiftrelief-osm-tiles-v1';
const TILE_HOST_PATTERNS = [
  'tile.openstreetmap.org',
  'a.tile.openstreetmap.org',
  'b.tile.openstreetmap.org',
  'c.tile.openstreetmap.org',
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

function isOsmTileRequest(requestUrl) {
  try {
    const url = new URL(requestUrl);
    if (!TILE_HOST_PATTERNS.includes(url.hostname)) {
      return false;
    }
    return /\/\d+\/\d+\/\d+\.png$/.test(url.pathname);
  } catch (_) {
    return false;
  }
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') {
    return;
  }

  if (!isOsmTileRequest(event.request.url)) {
    return;
  }

  event.respondWith((async () => {
    const cache = await caches.open(TILE_CACHE_NAME);
    const cached = await cache.match(event.request);
    if (cached) {
      return cached;
    }

    try {
      const networkResponse = await fetch(event.request);
      if (networkResponse) {
        await cache.put(event.request, networkResponse.clone());
      }
      return networkResponse;
    } catch (_) {
      return cached || Response.error();
    }
  })());
});
