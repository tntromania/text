// ══════════════════════════════════════════════════════════════
// hub-auth.js — Modul partajat de TOATE sub-aplicațiile
// ══════════════════════════════════════════════════════════════
// Pune acest fișier în root-ul fiecărei aplicații (lângă server.js)
// Adaugă în .env-ul fiecărei aplicații:
//   HUB_URL=https://hub.viralio.ro
//   INTERNAL_API_KEY=aceeași_cheie_ca_pe_hub
//   APP_NAME=captions   ← (v3) nume scurt: downloader|audiocut|captions|voice|pipeline|video
//
// Apoi în server.js:
//   const cookieParser = require('cookie-parser');  // NPM: npm i cookie-parser
//   const { authenticate, hubAPI } = require('./hub-auth');
//   app.use(cookieParser());                         // OBLIGATORIU înainte de authenticate
//
// ── MIGRARE LA COOKIE HttpOnly (v2) ───────────────────────────
// authenticate citește tokenul JWT:
//   1. din cookie HttpOnly `viralio_token` (cross-subdomain)
//   2. fallback pe Authorization: Bearer (compat clienți vechi)
//
// ── TRACKING RICH (v3) ────────────────────────────────────────
// useCredits/useVoiceChars acceptă acum un al 3-lea argument `opts`:
//   { action: 'captions-remove', meta: { outputUrl, model, ... }, durationMs: 3200 }
// Toate sunt OPȚIONALE — apelurile vechi `useCredits(userId, amount)` rămân OK.
// `app` se trimite automat din process.env.APP_NAME (sau HUB îl infereză din Origin).
// HUB-ul loghează un Event cu toate astea → vizibil în CRM /admin per user.
// ══════════════════════════════════════════════════════════════

const HUB_URL = process.env.HUB_URL;
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY;
const APP_NAME = process.env.APP_NAME || null; // ex: 'captions', 'voice'…

// v3: fail-soft — sub-app-urile care folosesc DOAR `hubAPI.trackEvent` (ex: voicepro,
// care are auth propriu și deduce voice_characters direct în Mongo) trebuie să poată
// `require('./hub-auth')` fără ca lipsa HUB_URL să crashuiască startup-ul. Helper-ele
// fac no-op cu un warning dacă nu sunt configurate.
const CONFIGURED = !!(HUB_URL && INTERNAL_API_KEY);
if (!HUB_URL)          console.warn('⚠️ HUB_URL lipsește din .env — funcțiile hubAPI vor fi no-op.');
if (!INTERNAL_API_KEY) console.warn('⚠️ INTERNAL_API_KEY lipsește din .env — funcțiile hubAPI vor fi no-op.');
if (!APP_NAME)         console.warn('⚠️ APP_NAME nu e setat — HUB va încerca să inferze app-ul din Origin.');

// Cache scurt (30s) pentru a nu bombarda HUB-ul la fiecare request
const tokenCache = new Map();
const CACHE_TTL = 30 * 1000; // 30 secunde

// ── sanitizeMeta (v3) ─────────────────────────────────────────
// Cap pe size + scoate field-uri PII evidente, ca să nu umflăm Mongo cu prompturi
// de 100KB sau să logăm parole/tokens.
// Reguli:
//   - max 24 keys
//   - string-uri trunchiate la 2000 chars
//   - rejectăm keys suspecte (password, token, secret, apikey, authorization)
const META_BLACKLIST = /(password|secret|token|apikey|authorization|cookie|session)/i;
function sanitizeMeta(meta) {
    if (!meta || typeof meta !== 'object') return {};
    const out = {};
    let count = 0;
    for (const [k, v] of Object.entries(meta)) {
        if (count >= 24) break;
        if (META_BLACKLIST.test(k)) continue;
        if (v == null) continue;
        if (typeof v === 'string') {
            out[k] = v.length > 2000 ? v.slice(0, 2000) + '…(truncat)' : v;
        } else if (typeof v === 'number' || typeof v === 'boolean') {
            out[k] = v;
        } else if (Array.isArray(v)) {
            out[k] = v.slice(0, 20);
        } else if (typeof v === 'object') {
            try { out[k] = JSON.parse(JSON.stringify(v)); } catch { /* skip */ }
        }
        count++;
    }
    return out;
}

const callHub = async (endpoint, body) => {
    if (!CONFIGURED) {
        throw new Error('hub-auth not configured (set HUB_URL + INTERNAL_API_KEY in .env)');
    }
    const response = await fetch(`${HUB_URL}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-internal-key': INTERNAL_API_KEY,
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(8000),
    });
    const data = await response.json();
    if (!response.ok) {
        const err = new Error(data.error || `Hub error ${response.status}`);
        err.status = response.status;
        err.data = data;
        throw err;
    }
    return data;
};

// ── Middleware authenticate ──────────────────────────────────
// Înlocuiește jwt.verify + mongoose din fiecare app
// Rezultatul: req.userId, req.user (din HUB)
// PREREQUISITE: app.use(cookieParser()) trebuie chemat înainte în server.js
const authenticate = async (req, res, next) => {
    // 1. Citim tokenul din cookie HttpOnly (sursa primară, securitate v2)
    // 2. Fallback pe Authorization Bearer (clienți legacy / API directe)
    const token = req.cookies?.viralio_token
               || req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Trebuie să fii logat!' });

    try {
        // Verificăm cache
        const cached = tokenCache.get(token);
        if (cached && Date.now() - cached.ts < CACHE_TTL) {
            req.userId = cached.userId;
            req.user = cached.user;
            return next();
        }

        // Apel către HUB
        const result = await callHub('/api/internal/verify-token', { token });
        req.userId = result.userId;
        req.user = result.user;

        // Salvăm în cache
        tokenCache.set(token, { userId: result.userId, user: result.user, ts: Date.now() });

        next();
    } catch (e) {
        const status = e.status || 401;
        return res.status(status).json({ error: e.message || 'Sesiune expirată.' });
    }
};

// ── Helper: actualizează creditele/voice în cache pentru userId
// Necesar ca după useCredits/useVoiceChars, /api/auth/me să returneze
// valoarea actualizată FĂRĂ să mai facă roundtrip la HUB.
function patchCachedUser(userId, fields) {
    const id = String(userId);
    for (const entry of tokenCache.values()) {
        if (String(entry.userId) === id && entry.user) {
            Object.assign(entry.user, fields);
        }
    }
}

// ── API-uri helper pentru credite & user ─────────────────────
//
// (v3) useCredits/useVoiceChars acceptă acum și opts pentru CRM tracking:
//   opts = { action, meta, durationMs }
//   - action:     string scurt ex 'captions-remove', 'voice-tts', 'video-generate'
//   - meta:       object cu prompt/outputUrl/model/... — afișat în /admin
//   - durationMs: număr opțional (cât a durat generarea, în ms)
// Toate sunt opționale. `app` se injectează automat din process.env.APP_NAME.
const hubAPI = {
    useCredits: async (userId, amount, opts = {}) => {
        const payload = { userId, amount };
        if (APP_NAME) payload.app = APP_NAME;
        if (opts.action)     payload.action = String(opts.action).slice(0, 64);
        if (opts.meta)       payload.meta = sanitizeMeta(opts.meta);
        if (typeof opts.durationMs === 'number') payload.durationMs = opts.durationMs;
        const result = await callHub('/api/internal/use-credits', payload);
        if (typeof result?.credits === 'number') patchCachedUser(userId, { credits: result.credits });
        return result;
    },

    useVoiceChars: async (userId, amount, opts = {}) => {
        const payload = { userId, amount };
        if (APP_NAME) payload.app = APP_NAME;
        if (opts.action)     payload.action = String(opts.action).slice(0, 64);
        if (opts.meta)       payload.meta = sanitizeMeta(opts.meta);
        if (typeof opts.durationMs === 'number') payload.durationMs = opts.durationMs;
        const result = await callHub('/api/internal/use-voice-chars', payload);
        if (typeof result?.voice_characters === 'number') patchCachedUser(userId, { voice_characters: result.voice_characters });
        return result;
    },

    // Verifică sold (fără a scădea, returnează {credits, voice_characters})
    checkCredits: async (userId) => {
        return callHub('/api/internal/check-credits', { userId });
    },

    // Info user complet — pune și update pe cache (sursa proaspătă de la HUB)
    getUserInfo: async (userId) => {
        const result = await callHub('/api/internal/user-info', { userId });
        if (result?.user) {
            patchCachedUser(userId, {
                credits: result.user.credits,
                voice_characters: result.user.voice_characters,
                subscriptionPlan: result.user.subscriptionPlan,
                subscriptionStatus: result.user.subscriptionStatus,
            });
        }
        return result;
    },

    // Invalidare manuală (dacă vrei să forțezi re-fetch la următorul request)
    invalidateUserCache: (userId) => {
        const id = String(userId);
        for (const [token, entry] of tokenCache) {
            if (String(entry.userId) === id) tokenCache.delete(token);
        }
    },

    // ── trackEvent (v3) ──────────────────────────────────────
    // Pentru sub-app-uri care fac deducerea LOR în Mongo (NU via useCredits/useVoiceChars
    // — ex: voicepro modifică direct voice_characters). Trimite un event către HUB
    // fire-and-forget. Nu modifică credite/voice — doar loghează pentru CRM.
    //
    // Uz:
    //   hubAPI.trackEvent(userId, {
    //       action: 'voice-generate',
    //       voiceCharsCost: 1200,
    //       durationMs: 4500,
    //       meta: { model, voiceId, textPreview: text.slice(0, 200), outputUrl }
    //   });
    trackEvent: (userId, payload = {}) => {
        if (!CONFIGURED) return; // silent no-op dacă nu e configurat — fail-soft pentru sub-app-uri care n-au setat încă HUB_URL
        const body = {
            userId,
            action: String(payload.action || 'unknown').slice(0, 64),
            status: payload.status || 'success',
        };
        if (APP_NAME) body.app = APP_NAME;
        if (payload.meta) body.meta = sanitizeMeta(payload.meta);
        if (typeof payload.durationMs === 'number') body.durationMs = payload.durationMs;
        if (typeof payload.creditsCost === 'number') body.creditsCost = payload.creditsCost;
        if (typeof payload.voiceCharsCost === 'number') body.voiceCharsCost = payload.voiceCharsCost;

        callHub('/api/internal/track-event', body).catch(err => {
            console.warn('⚠️ trackEvent failed:', err.message);
        });
    },
};

// Curățare cache periodică
setInterval(() => {
    const now = Date.now();
    for (const [key, val] of tokenCache) {
        if (now - val.ts > CACHE_TTL) tokenCache.delete(key);
    }
}, 60000);

module.exports = { authenticate, hubAPI };
