// Basic client logic (styling handled by teammate)

const TOKEN_KEY = 'hr_token';

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function apiFetch(url, options = {}) {
  const headers = options.headers || {};
  headers['Content-Type'] = 'application/json';
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(url, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.message || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function showNav() {
  const token = getToken();
  const login = document.getElementById('nav-login');
  const logout = document.getElementById('nav-logout');
  if (login && logout) {
    if (token) {
      login.style.display = 'none';
      logout.style.display = 'inline';
    } else {
      login.style.display = 'inline';
      logout.style.display = 'none';
    }
  }
}

function bindLogout() {
  const logout = document.getElementById('nav-logout');
  if (!logout) return;
  logout.addEventListener('click', (e) => {
    e.preventDefault();
    clearToken();
    showNav();
    window.location.href = '/login';
  });
}

async function initDashboard() {
  const who = document.getElementById('whoami');
  if (!who) return;
  const token = getToken();
  if (!token) {
    who.textContent = 'Not logged in. Please login.';
    return;
  }
  try {
    const res = await apiFetch('/api/auth/me', { method: 'GET' });
    who.textContent = `Logged in as ${res.user.name} (${res.user.email})`;
  } catch (err) {
    who.textContent = 'Session expired. Please login again.';
    clearToken();
    showNav();
  }
}

function bindLogin() {
  const form = document.getElementById('login-form');
  if (!form) return;
  const msg = document.getElementById('login-msg');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    const fd = new FormData(form);
    const payload = {
      email: fd.get('email'),
      password: fd.get('password')
    };
    try {
      const res = await apiFetch('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      setToken(res.token);
      showNav();
      window.location.href = '/dashboard';
    } catch (err) {
      msg.textContent = err.message;
    }
  });
}

function bindRegister() {
  const form = document.getElementById('register-form');
  if (!form) return;
  const msg = document.getElementById('register-msg');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    const fd = new FormData(form);
    const payload = {
      name: fd.get('name') || '',
      email: fd.get('email'),
      password: fd.get('password')
    };
    try {
      const res = await apiFetch('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      setToken(res.token);
      showNav();
      window.location.href = '/dashboard';
    } catch (err) {
      msg.textContent = err.message;
    }
  });
}

function bindRecommend() {
  const btnLive = document.getElementById('btn-live');
  const btnGeocode = document.getElementById('btn-geocode');
  const btnRec = document.getElementById('btn-recommend');
  if (!btnLive && !btnGeocode && !btnRec) return;

  const latEl = document.getElementById('lat');
  const lonEl = document.getElementById('lon');
  const placeEl = document.getElementById('place-query');
  const emergencyEl = document.getElementById('emergency-type');
  const msg = document.getElementById('rec-msg');
  const resultsEl = document.getElementById('rec-results');

  function setMsg(text) {
    if (msg) msg.textContent = text;
  }

  function renderResults(list) {
    if (!resultsEl) return;
    resultsEl.innerHTML = '';
    if (!list || list.length === 0) {
      resultsEl.innerHTML = '<p class="muted">No results.</p>';
      return;
    }

    const ul = document.createElement('div');
    ul.className = 'result-list';

    list.forEach((h, idx) => {
      const card = document.createElement('div');
      card.className = 'result-card';

      const ratingTxt = (h.rating !== null && h.rating !== undefined)
        ? `${Number(h.rating).toFixed(1)} ★ (${h.user_rating_count || 0})`
        : 'N/A';
      const phoneTxt = h.phone ? escapeHtml(h.phone) : 'N/A';
      const statusTxt = h.business_status ? escapeHtml(h.business_status) : 'UNVERIFIED';

      card.innerHTML = `
        <div class="row" style="justify-content:space-between; gap:12px;">
          <strong>${idx + 1}. ${escapeHtml(h.name)}</strong>
          <span>${Number(h.distance_km).toFixed(2)} km</span>
        </div>
        <div class="muted">${escapeHtml(h.address || '')}</div>
        <div class="muted">tags: ${(h.tags || []).join(', ')} | geo_good: ${h.geo_good} | status: ${statusTxt}</div>
        <div class="muted">rating: ${ratingTxt} | phone: ${phoneTxt}</div>
        <div class="actions">
          <a class="btn" href="${h.maps}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a>
        </div>
      `;
      ul.appendChild(card);
    });

    resultsEl.appendChild(ul);
  }

  btnLive?.addEventListener('click', (e) => {
    e.preventDefault();
    setMsg('Requesting GPS location…');
    if (!navigator.geolocation) {
      setMsg('Geolocation not supported in this browser. Enter place name or coordinates.');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        latEl.value = pos.coords.latitude;
        lonEl.value = pos.coords.longitude;
        setMsg('Location set from GPS.');
      },
      (err) => {
        setMsg(`GPS error: ${err.message}`);
      },
      { enableHighAccuracy: true, timeout: 15000 }
    );
  });

  btnGeocode?.addEventListener('click', async (e) => {
    e.preventDefault();
    const q = (placeEl?.value || '').trim();
    if (!q) {
      setMsg('Enter a place name first.');
      return;
    }
    setMsg('Geocoding place…');
    try {
      const res = await apiFetch(`/api/geocode?query=${encodeURIComponent(q)}&region=in`, { method: 'GET' });
      latEl.value = res.data.lat;
      lonEl.value = res.data.lon;
      setMsg(`Place found: ${res.data.display_name} (via ${res.data.provider})`);
    } catch (err) {
      setMsg(err.message);
    }
  });

  placeEl?.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    const q = (placeEl?.value || '').trim();
    if (!q) {
      setMsg('Enter a place name first.');
      return;
    }
    setMsg('Geocoding place…');
    try {
      const res = await apiFetch(`/api/geocode?query=${encodeURIComponent(q)}&region=in`, { method: 'GET' });
      latEl.value = res.data.lat;
      lonEl.value = res.data.lon;
      setMsg(`Place found: ${res.data.display_name} (via ${res.data.provider})`);
    } catch (err) {
      setMsg(err.message);
    }
  });

  btnRec?.addEventListener('click', async (e) => {
    e.preventDefault();
    const lat = parseFloat(latEl?.value);
    const lon = parseFloat(lonEl?.value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      setMsg('Please set a valid latitude & longitude (GPS or place).');
      return;
    }
    setMsg('Computing recommendations…');
    try {
      const payload = {
        lat,
        lon,
        emergency_type: emergencyEl?.value || 'general',
        top_k: 5,
        shortlist: 20
      };
      const res = await apiFetch('/api/recommend', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      setMsg('Done.');
      renderResults(res.data);
    } catch (err) {
      setMsg(err.message);
    }
  });
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

// Boot
showNav();
bindLogout();
bindLogin();
bindRegister();
initDashboard();
bindRecommend();
