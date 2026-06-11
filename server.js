const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const express = require('express');
const multer = require('multer');
const compression = require('compression');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();

const PORT = process.env.PORT || 3000;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const MAX_VIDEO_SIZE_MB = Number(process.env.MAX_VIDEO_SIZE_MB) || 250;
// Base host for the Gemini REST + File API (override only for testing/proxies).
const GEMINI_API_BASE =
  process.env.GEMINI_FILE_API_BASE || 'https://generativelanguage.googleapis.com';

// "High demand" fallback: when the primary key/model returns an overload/quota error
// (429/503/500/403), /api/analyze retries with this (higher-tier) key across the models
// in GEMINI_FALLBACK_MODELS, in order. Unset GEMINI_FALLBACK_API_KEY → fallback disabled
// (behaviour unchanged). The fallback key is env-only and never logged.
const GEMINI_FALLBACK_API_KEY = process.env.GEMINI_FALLBACK_API_KEY || '';
const GEMINI_FALLBACK_MODELS = (
  process.env.GEMINI_FALLBACK_MODELS ||
  'gemini-2.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash'
)
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
if (GEMINI_FALLBACK_API_KEY) {
  console.log(`[analyze] Gemini fallback enabled: ${GEMINI_FALLBACK_MODELS.length} model(s).`);
}

// Passphrase that gates the hidden runtime key-replacement endpoint. The feature is
// DISABLED unless this is set in the environment (.env). Never logged.
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || '';

// Runtime Gemini key. A new key set via POST /api/admin/gemini-key is persisted here
// and loaded at boot, OVERRIDING the .env value — so a swapped key survives a restart.
// Extensionless on purpose: nodemon (watches js/json/…) won't restart when it's written.
const GEMINI_KEY_FILE = path.join(__dirname, '.gemini-key');
const loadPersistedKey = () => {
  try {
    return fs.readFileSync(GEMINI_KEY_FILE, 'utf8').trim();
  } catch (_) {
    return '';
  }
};
// Mutable so the admin endpoint can swap it at runtime; every Gemini request reads the
// current value. Persisted file wins over .env (it's the most recently chosen key).
let geminiApiKey = loadPersistedKey() || process.env.GEMINI_API_KEY || '';

const DEFAULT_PROMPT = `You are an expert in nonverbal communication, emotion analysis and human behavior.

Analyze this video or image. Focus on emotions, facial expressions, posture and gestures. Be concise.

FORMATTING RULES:
- Do NOT start with filler phrases like "Sure!", "Here's...", "Certainly!", etc. Start directly with the analysis.
- Use numbered sections (0. 1. 2. 3. 4.) and subsections (1.1 1.2 etc.) as headings.
- Use * for bullet points, each on its own line.
- Use **bold** for key terms and emotions.
- Separate major sections with --- on its own line.
- Keep blank lines between sections and subsections.
- Each bullet: max 1-2 short sentences.
- Do NOT invent details. If something cannot be assessed, write: "Not enough visual data."

Structure:

0. Overall Picture
- One sentence describing what is happening in the video.

1. Emotional Analysis
1.1 Emotional Tone
- Dominant emotions.
- Valence: positive / neutral / negative.
1.2 Emotion Dynamics
- How emotions shift from start to middle to end.
- Any sharp emotional changes.
1.3 Incongruence
- Mismatch between verbal context and nonverbal signals.

2. Facial Analysis
2.1 People
- How many visible. Label as Person 1, Person 2, etc.
2.2 Expressions
- Main emotions via facial expression per person.
- Micro-expressions if noticeable.
2.3 Gaze
- Eye contact with camera or others.
- Gaze aversion direction and meaning.

3. Body and Gestures
3.1 Posture
- Open vs closed. Body tension level.
3.2 Gestures
- Hand gestures: controlled / natural / excessive.
- Self-soothing gestures (touching neck, face, hands).
3.3 Space
- Distance to others or camera. Leaning direction.

4. Summary
4.1 Profile
- 3-5 bullets on emotional state, confidence, engagement.
4.2 Key Signals
- 3-5 most important nonverbal signals with short explanations.
4.3 Interpretation
- What the nonverbal behavior indicates (trust, stress, confidence, defensiveness, etc.).`;

// Disk storage (not memory) so a large upload never sits fully in RAM — the
// file is streamed to a temp path, then streamed up to the Gemini File API.
const upload = multer({
  storage: multer.diskStorage({
    destination: os.tmpdir(),
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname || '') || '.bin';
      cb(null, `aema-upload-${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`);
    }
  }),
  limits: {
    fileSize: MAX_VIDEO_SIZE_MB * 1024 * 1024
  }
});

if (!geminiApiKey) {
  console.warn('Warning: no Gemini API key set (.gemini-key or GEMINI_API_KEY). /api/analyze requests will fail.');
}

// gzip/deflate text responses (HTML/CSS/JS/JSON/SVG).
app.use(compression());

app.use(express.json({ limit: '2mb' }));
app.use(express.urlencoded({ extended: true, limit: '2mb' }));

// App shell changes between deploys → short TTL. Assets (videos, SVGs, fonts,
// model .task files) → long TTL with etag revalidation. /uploads is transient.
app.use(
  express.static(path.join(__dirname, 'public'), {
    etag: true,
    lastModified: true,
    maxAge: 0
  })
);
app.use(
  '/assets',
  express.static(path.join(__dirname, 'assets'), {
    etag: true,
    lastModified: true,
    maxAge: '7d'
  })
);
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

const framesDir = path.join(__dirname, 'assets', 'export', 'frames');
fs.mkdirSync(framesDir, { recursive: true });

const libraryDir = path.join(__dirname, 'assets', 'archive', 'library');
fs.mkdirSync(libraryDir, { recursive: true });

const LIBRARY_EXTS = new Set(['.mp4', '.mov', '.webm', '.jpg', '.jpeg', '.png']);
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png']);

app.get('/api/library', async (_req, res) => {
  try {
    const files = (await fs.promises.readdir(libraryDir)).filter((f) => {
      const ext = path.extname(f).toLowerCase();
      return LIBRARY_EXTS.has(ext) && !f.startsWith('.');
    });

    // Map of image basename → filename, so each video can find a sibling
    // thumbnail (same basename, image extension) sitting beside it.
    const imageByBase = new Map();
    for (const f of files) {
      const ext = path.extname(f).toLowerCase();
      if (IMAGE_EXTS.has(ext)) {
        const base = f.slice(0, -ext.length);
        if (!imageByBase.has(base)) imageByBase.set(base, f);
      }
    }

    // Images that get attached to a video are not also listed standalone.
    const usedImages = new Set();
    const items = [];
    for (const f of files) {
      const ext = path.extname(f).toLowerCase();
      const isImage = IMAGE_EXTS.has(ext);
      const entry = {
        name: f,
        path: `/assets/archive/library/${encodeURIComponent(f)}`,
        type: isImage ? 'image' : 'video',
      };
      if (!isImage) {
        const base = f.slice(0, -ext.length);
        const thumbFile = imageByBase.get(base);
        if (thumbFile) {
          entry.thumb = `/assets/archive/library/${encodeURIComponent(thumbFile)}`;
          usedImages.add(thumbFile);
        } else {
          // Thumbless video: include its byte size so the client can skip generating
          // a thumbnail for very large imports (decode-memory safety on the kiosk).
          try { entry.size = (await fs.promises.stat(path.join(libraryDir, f))).size; } catch (_) {}
        }
      }
      items.push(entry);
    }

    // Drop image entries that are only acting as a video's thumbnail.
    const filtered = items.filter((it) => !(it.type === 'image' && usedImages.has(it.name)));
    res.json(filtered);
  } catch (err) {
    res.status(500).json({ error: 'Failed to read library' });
  }
});

// Folder-limited pickers: list the files in a WHITELISTED archive subfolder so
// the in-app picker can be constrained to one directory (the OS file dialog
// cannot be). `kind` is a fixed key — never a path — so there's no traversal.
const FOLDER_DIRS = {
  background: { rel: 'background_images', exts: IMAGE_EXTS },
  media:      { rel: 'media',            exts: LIBRARY_EXTS },
};
Object.values(FOLDER_DIRS).forEach(({ rel }) =>
  fs.mkdirSync(path.join(__dirname, 'assets', 'archive', rel), { recursive: true })
);

app.get('/api/folder/:kind', async (req, res) => {
  const cfg = FOLDER_DIRS[req.params.kind];
  if (!cfg) return res.status(404).json({ error: 'Unknown folder.' });
  try {
    const dir = path.join(__dirname, 'assets', 'archive', cfg.rel);
    const files = (await fs.promises.readdir(dir))
      .filter((f) => !f.startsWith('.') && cfg.exts.has(path.extname(f).toLowerCase()))
      .sort();
    res.json({
      items: files.map((f) => ({
        name: f,
        path: `/assets/archive/${cfg.rel}/${encodeURIComponent(f)}`,
        type: IMAGE_EXTS.has(path.extname(f).toLowerCase()) ? 'image' : 'video',
      })),
    });
  } catch (err) {
    res.status(500).json({ error: 'Could not read folder.' });
  }
});

app.post('/api/capture-frame', express.raw({ type: 'image/png', limit: '20mb' }), async (req, res, next) => {
  const filename = req.query.filename;
  if (!filename || !/^frame_.+_\d{2}-\d{2}\.png$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  const safeName = path.basename(filename);
  const ext = path.extname(safeName);
  const stem = safeName.slice(0, -ext.length);
  let finalName = safeName;
  let n = 1;
  // Atomic create with the 'wx' flag — on EEXIST retry with a (copy N)
  // suffix. Closes the TOCTOU race that existsSync + writeFile allowed.
  while (true) {
    let handle;
    try {
      handle = await fs.promises.open(path.join(framesDir, finalName), 'wx');
    } catch (err) {
      if (err.code === 'EEXIST') {
        finalName = `${stem} (copy ${n})${ext}`;
        n++;
        continue;
      }
      return next(err);
    }
    try {
      await handle.writeFile(req.body);
    } finally {
      await handle.close();
    }
    break;
  }
  res.json({ ok: true, name: finalName, path: `/assets/export/frames/${encodeURIComponent(finalName)}` });
});

app.post('/api/capture-frameset-frame-v2', express.raw({ type: ['image/png', 'image/jpeg'], limit: '20mb' }), async (req, res) => {
  const { dir, filename } = req.query;
  if (!dir || !filename) return res.status(400).json({ error: 'Missing dir or filename' });
  const safeFile = path.basename(filename);
  let target;
  if (path.isAbsolute(dir)) {
    // Absolute path = deliberate local-dev intent (e.g. ~/Desktop/out).
    target = dir;
  } else {
    // Relative paths must resolve inside the project root — block ../ escapes.
    target = path.resolve(__dirname, dir);
    if (target !== __dirname && !target.startsWith(__dirname + path.sep)) {
      return res.status(400).json({ error: 'Invalid dir' });
    }
  }
  try {
    await fs.promises.mkdir(target, { recursive: true });
    await fs.promises.writeFile(path.join(target, safeFile), req.body);
    res.json({ ok: true, path: path.join(target, safeFile) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post(
  '/api/archive-clip',
  express.raw({ type: ['video/webm', 'video/mp4'], limit: '50mb' }),
  async (req, res, next) => {
    const now = new Date();
    const ts = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, '0'),
      String(now.getDate()).padStart(2, '0'),
      '_',
      String(now.getHours()).padStart(2, '0'),
      '-',
      String(now.getMinutes()).padStart(2, '0'),
      '-',
      String(now.getSeconds()).padStart(2, '0'),
    ].join('');
    // mp4 on Safari, webm on Chrome/Firefox.
    const ext = (req.headers['content-type'] || '').startsWith('video/mp4') ? 'mp4' : 'webm';
    const filename = `live_${ts}.${ext}`;
    const filePath = path.join(libraryDir, filename);
    try {
      await fs.promises.writeFile(filePath, req.body);
    } catch (err) {
      return next(err);
    }
    res.json({ ok: true, name: filename, path: `/assets/archive/library/${encodeURIComponent(filename)}` });
  }
);

// ── Gemini File API helpers ──────────────────────────────────────────────
// Carries an HTTP status the client can safely receive; the detailed Gemini
// payload is logged server-side only (never echoed in the response body).
class GeminiError extends Error {
  constructor(message, status = 502, upstreamStatus = null) {
    super(message);
    this.name = 'GeminiError';
    this.status = status; // HTTP status to return to the client
    this.upstreamStatus = upstreamStatus; // the real Gemini HTTP status (for fallback classification)
  }
}

// "High demand" / quota statuses that should trigger the Tier-1 fallback chain
// (overload, rate-limit, internal, permission/billing). NOT 400 (malformed) / 401 (bad key).
const FALLBACK_STATUSES = new Set([429, 500, 503, 403]);
const isRetryableUpstream = (err) =>
  FALLBACK_STATUSES.has(err?.upstreamStatus ?? (err instanceof GeminiError ? err.status : null));

// Strip any `?key=…` so the Gemini API key never reaches logs or responses
// (node-fetch FetchError messages embed the full request URL, key included).
const redactKey = (s) => String(s ?? '').replace(/(key=)[^&\s'"]+/gi, '$1REDACTED');

// node-fetch system/socket errors worth retrying — NOT aborts, NOT HTTP errors.
const isTransientNetworkError = (err) =>
  !!err && err.name !== 'AbortError' && err.type !== 'aborted' &&
  (['EPIPE', 'ECONNRESET', 'ETIMEDOUT', 'ECONNREFUSED', 'EAI_AGAIN', 'ENOTFOUND'].includes(err.code) ||
   err.type === 'system' || /socket hang up|EPIPE|ECONNRESET/i.test(err.message || ''));

// Resumable upload of a temp file to the Gemini File API, with retry on transient
// network drops (EPIPE/ECONNRESET, etc.). Returns the file resource.
async function geminiUploadFile(filePath, mimeType, sizeBytes, displayName, signal, apiKey) {
  const attempts = 3;
  let lastErr;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await geminiUploadFileOnce(filePath, mimeType, sizeBytes, displayName, signal, apiKey);
    } catch (err) {
      lastErr = err;
      // Respect the request budget; don't retry real HTTP-level Gemini errors.
      if (signal?.aborted || err?.name === 'AbortError' || err?.type === 'aborted') throw err;
      if (err instanceof GeminiError) throw err;
      if (!isTransientNetworkError(err) || attempt === attempts) break;
      console.warn(`[analyze] upload attempt ${attempt} failed (${err.code || err.message}); retrying…`);
      await new Promise((r) => setTimeout(r, 500 * attempt));
    }
  }
  // Terminal transient failure → clean, key-free message (503 = retryable).
  console.error(`[analyze] upload network error after retries: ${redactKey(lastErr?.message || String(lastErr))}`);
  throw new GeminiError('Could not upload the video to Gemini (connection dropped). Please try again.', 503);
}

// One resumable-upload attempt: start a session, then stream the body.
async function geminiUploadFileOnce(filePath, mimeType, sizeBytes, displayName, signal, apiKey) {
  const startRes = await fetch(`${GEMINI_API_BASE}/upload/v1beta/files?key=${encodeURIComponent(apiKey)}`, {
    method: 'POST',
    headers: {
      'X-Goog-Upload-Protocol': 'resumable',
      'X-Goog-Upload-Command': 'start',
      'X-Goog-Upload-Header-Content-Length': String(sizeBytes),
      'X-Goog-Upload-Header-Content-Type': mimeType,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ file: { display_name: displayName } }),
    signal
  });
  if (!startRes.ok) {
    const raw = await startRes.text();
    console.error(`[analyze] upload init failed HTTP ${startRes.status}: ${raw}`);
    // Surface Gemini's real error message (e.g. "prepayment credits depleted")
    // so the user sees the actual cause instead of a generic message.
    let geminiMsg = '';
    try { geminiMsg = JSON.parse(raw)?.error?.message || ''; } catch (_) {}
    const detail = geminiMsg ? ` ${geminiMsg.trim()}` : '';
    throw new GeminiError(`Gemini upload could not be started (${startRes.status}).${detail}`, 502, startRes.status);
  }
  const uploadUrl = startRes.headers.get('x-goog-upload-url');
  if (!uploadUrl) {
    throw new GeminiError('Gemini upload URL was not returned.', 502);
  }

  const uploadRes = await fetch(uploadUrl, {
    method: 'POST',
    headers: {
      'X-Goog-Upload-Command': 'upload, finalize',
      'X-Goog-Upload-Offset': '0',
      'Content-Length': String(sizeBytes)
    },
    body: fs.createReadStream(filePath),
    signal
  });
  if (!uploadRes.ok) {
    console.error(`[analyze] upload failed HTTP ${uploadRes.status}: ${await uploadRes.text()}`);
    throw new GeminiError(`Gemini upload failed (${uploadRes.status}).`, 502, uploadRes.status);
  }
  const uploaded = await uploadRes.json();
  if (!uploaded?.file?.uri || !uploaded?.file?.name) {
    throw new GeminiError('Gemini upload response was malformed.', 502);
  }
  return uploaded.file;
}

// Poll a file resource until it leaves PROCESSING. 4 min ceiling, backoff.
async function geminiWaitUntilActive(fileName, signal, apiKey) {
  const deadline = Date.now() + 4 * 60 * 1000;
  let delayMs = 1000;
  // fileName is like "files/abc123"
  while (true) {
    const res = await fetch(`${GEMINI_API_BASE}/v1beta/${fileName}?key=${encodeURIComponent(apiKey)}`, { signal });
    if (!res.ok) {
      console.error(`[analyze] file status HTTP ${res.status}: ${await res.text()}`);
      throw new GeminiError(`Gemini file status check failed (${res.status}).`, 502, res.status);
    }
    const info = await res.json();
    if (info.state === 'ACTIVE') return info;
    if (info.state === 'FAILED') {
      throw new GeminiError('Gemini failed to process the uploaded video.', 502);
    }
    if (Date.now() > deadline) {
      throw new GeminiError('Gemini video processing timed out.', 504);
    }
    await new Promise((r) => setTimeout(r, delayMs));
    delayMs = Math.min(Math.round(delayMs * 1.5), 8000);
  }
}

// Best-effort cleanup — never throws. Uses its own short timeout so it still
// runs even when the request's main AbortController has already fired.
async function geminiDeleteFile(fileName, apiKey) {
  try {
    await fetch(`${GEMINI_API_BASE}/v1beta/${fileName}?key=${encodeURIComponent(apiKey)}`, {
      method: 'DELETE',
      signal: AbortSignal.timeout(10000)
    });
  } catch (err) {
    console.warn(`[analyze] could not delete Gemini file ${fileName}: ${err.message}`);
  }
}

// One generateContent call for a given model + key. Throws GeminiError carrying the
// real upstream HTTP status (so the fallback loop can classify 429/503/500/403). The
// raw Gemini payload stays in the server log only — the thrown message is sanitized.
async function geminiGenerateContent(model, payload, signal, apiKey) {
  const res = await fetch(
    `${GEMINI_API_BASE}/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    }
  );
  if (!res.ok) {
    const errorText = await res.text();
    let geminiMessage = errorText;
    try { geminiMessage = JSON.parse(errorText)?.error?.message || errorText; } catch (_) {}
    console.error(`[analyze] generateContent (${model}) HTTP ${res.status}: ${geminiMessage}`);
    throw new GeminiError(`Gemini API error (${res.status}): ${geminiMessage}`, 502, res.status);
  }
  return res.json();
}

app.post('/api/analyze', upload.single('video'), async (req, res, next) => {
  const videoFile = req.file;
  // Every Gemini-side upload, keyed by the API key that owns it, so each is cleaned up
  // under its own key. A fallback to a different key re-uploads the file (File API
  // uploads are key/project-scoped), so there can be more than one.
  const uploads = new Map(); // apiKey -> { name, uri, mimeType }
  // Budget covers upload + processing + generation AND a possible fallback re-upload
  // under the Tier-1 key. Kept under the server's 10 min requestTimeout.
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 9 * 60 * 1000);

  try {
    if (!geminiApiKey) {
      return res.status(500).json({ error: 'Server misconfiguration: missing Gemini API key.' });
    }
    if (!videoFile) {
      return res.status(400).json({ error: 'File is required.' });
    }

    const promptInput = req.body.prompt?.trim();
    const prompt = promptInput || DEFAULT_PROMPT;
    const mimeType = videoFile.mimetype || 'application/octet-stream';
    const isImage = mimeType.startsWith('image/');

    // Images go inline (base64) — no File API upload, so the same payload part works
    // for every attempt. Videos are uploaded per key, on demand, and cached below.
    let imageInline = null;
    if (isImage) {
      const buffer = await fs.promises.readFile(videoFile.path);
      imageInline = { mimeType, data: buffer.toString('base64') };
    }

    // Upload (or reuse) the video under a given key; returns its fileData payload part.
    const fileDataPartFor = async (apiKey) => {
      let rec = uploads.get(apiKey);
      if (!rec) {
        const uploaded = await geminiUploadFile(
          videoFile.path,
          mimeType,
          videoFile.size,
          videoFile.originalname || 'video',
          controller.signal,
          apiKey
        );
        const activeFile = await geminiWaitUntilActive(uploaded.name, controller.signal, apiKey);
        rec = { name: uploaded.name, uri: activeFile.uri, mimeType: activeFile.mimeType || mimeType };
        uploads.set(apiKey, rec);
      }
      return { fileData: { mimeType: rec.mimeType, fileUri: rec.uri } };
    };

    // Attempt 1 = primary key + model; then (if configured) the Tier-1 key across the
    // fallback models, in order. Only overload/quota errors advance the chain.
    const attempts = [{ key: geminiApiKey, model: GEMINI_MODEL }];
    if (GEMINI_FALLBACK_API_KEY) {
      for (const model of GEMINI_FALLBACK_MODELS) {
        attempts.push({ key: GEMINI_FALLBACK_API_KEY, model });
      }
    }

    let lastErr;
    for (let i = 0; i < attempts.length; i++) {
      const { key, model } = attempts[i];
      try {
        const parts = [{ text: prompt }];
        parts.push(isImage ? { inlineData: imageInline } : await fileDataPartFor(key));
        const payload = { contents: [{ role: 'user', parts }] };

        const result = await geminiGenerateContent(model, payload, controller.signal, key);
        const output = [];
        if (Array.isArray(result?.candidates)) {
          result.candidates.forEach((candidate) => {
            candidate?.content?.parts?.forEach((part) => {
              if (part?.text) {
                output.push(part.text);
              }
            });
          });
        }
        return res.json({
          resultText: output.join('\n\n') || 'AI did not return any text.',
          raw: result,
          model
        });
      } catch (err) {
        lastErr = err;
        if (err?.name === 'AbortError' || err?.type === 'aborted') throw err; // budget exceeded
        const hasMore = i < attempts.length - 1;
        if (hasMore && isRetryableUpstream(err)) {
          const status = err?.upstreamStatus ?? err?.status;
          console.warn(`[analyze] attempt ${i + 1} (model ${model}) upstream ${status}; falling back…`);
          continue;
        }
        throw err; // non-retryable, or chain exhausted
      }
    }
    throw lastErr; // unreachable (loop returns or throws) — kept for safety
  } catch (err) {
    if (err?.name === 'AbortError' || err?.type === 'aborted') {
      console.error('[analyze] aborted: 9 min budget exceeded');
      return res.status(504).json({ error: 'Analysis timed out. Try a shorter video.' });
    }
    if (err instanceof GeminiError) {
      console.error(`[analyze] ${err.message}`);
      return res.status(err.status).json({ error: err.message });
    }
    return next(err);
  } finally {
    clearTimeout(abortTimer);
    // Cleanup: remove every Gemini-side file (under its owning key) + the local temp file.
    for (const [apiKey, rec] of uploads) {
      await geminiDeleteFile(rec.name, apiKey);
    }
    if (videoFile?.path) {
      fs.promises.unlink(videoFile.path).catch((err) => {
        if (err.code !== 'ENOENT') {
          console.warn(`[analyze] could not delete temp file: ${err.message}`);
        }
      });
    }
  }
});

// ── Hidden runtime Gemini API key replacement ──────────────────────────────
// Constant-time passphrase compare (sha256 → fixed length so timingSafeEqual never
// throws on length mismatch and the comparison doesn't leak length via timing).
const tokensMatch = (a, b) => {
  if (!a || !b) return false;
  const ha = crypto.createHash('sha256').update(String(a)).digest();
  const hb = crypto.createHash('sha256').update(String(b)).digest();
  return crypto.timingSafeEqual(ha, hb);
};

const persistKey = (key) =>
  fs.promises.writeFile(GEMINI_KEY_FILE, key, { mode: 0o600 });

// Lightweight liveness check on a candidate key: list models. ok → usable; an explicit
// auth/format rejection (400/401/403) → bad key; 429 (rate-limited) is still a VALID key;
// any network/5xx → accept but flag `unverified` (so a key can be swapped in even when
// Gemini is briefly unreachable — the whole point of this feature).
async function validateGeminiKey(key) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(
      `${GEMINI_API_BASE}/v1beta/models?key=${encodeURIComponent(key)}`,
      { signal: controller.signal }
    );
    if (res.ok || res.status === 429) return { ok: true };
    if (res.status === 400 || res.status === 401 || res.status === 403) return { ok: false };
    return { ok: true, unverified: true };
  } catch (_) {
    return { ok: true, unverified: true };
  } finally {
    clearTimeout(timer);
  }
}

// Hidden endpoint — not linked anywhere; reached via the in-app modal (Ctrl+Alt+K or
// the #set-api-key URL). Swaps the live key + persists it so it survives a restart.
app.post('/api/admin/gemini-key', async (req, res) => {
  try {
    if (!ADMIN_TOKEN) {
      return res.status(503).json({ error: 'Key replacement is disabled: set ADMIN_TOKEN in the server .env to enable it.' });
    }
    const { token, key } = req.body || {};
    if (typeof token !== 'string' || !tokensMatch(token, ADMIN_TOKEN)) {
      return res.status(403).json({ error: 'Invalid passphrase.' });
    }
    const newKey = typeof key === 'string' ? key.trim() : '';
    if (!newKey || /\s/.test(newKey) || newKey.length < 20 || newKey.length > 200) {
      return res.status(400).json({ error: 'That does not look like a valid API key.' });
    }
    const check = await validateGeminiKey(newKey);
    if (!check.ok) {
      return res.status(400).json({ error: 'Gemini rejected that key (invalid or unauthorized).' });
    }
    geminiApiKey = newKey;
    await persistKey(newKey);
    console.log(`[admin] Gemini API key replaced at runtime${check.unverified ? ' (unverified — could not reach Gemini to test)' : ''}.`);
    return res.json({ ok: true, unverified: Boolean(check.unverified) });
  } catch (err) {
    console.error('[admin] key update failed:', redactKey(err?.message || String(err)));
    return res.status(500).json({ error: 'Could not save the key.' });
  }
});

app.get('/healthz', (_req, res) => {
  res.json({ ok: true, gemini: Boolean(geminiApiKey) });
});

app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.use((err, _req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res
        .status(413)
        .json({ error: `Video is too large. Max supported size is ${MAX_VIDEO_SIZE_MB} MB.` });
    }
    return res.status(400).json({ error: `Upload failed: ${err.message}` });
  }

  if (err) {
    // Redact any `?key=…` — node-fetch errors embed the full URL incl. the API key.
    console.error('Unhandled error:', redactKey(err?.stack || err?.message || String(err)));
    return res.status(500).json({ error: 'Internal server error', details: redactKey(err?.message) });
  }

  next();
});

const server = app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});

// Large uploads + multi-minute Gemini processing need generous timeouts.
server.keepAliveTimeout = 65000;
server.headersTimeout = 70000;
server.requestTimeout = 600000; // 10 min

// ── Graceful shutdown + process-level error handlers ─────────────────────
let shuttingDown = false;
const shutdown = (code = 0) => {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('Shutting down — draining in-flight requests…');
  // server.close waits for in-flight requests to finish before the callback.
  server.close(() => {
    console.log('All requests drained — exiting.');
    process.exit(code);
  });
  // Drop idle keep-alive sockets so they don't hold server.close open;
  // genuine in-flight requests (e.g. /api/analyze) are left to finish.
  server.closeIdleConnections?.();
  // Hard ceiling matches the /api/analyze budget so a real in-flight
  // request is never cut short, while a wedged socket can't block forever.
  setTimeout(() => {
    console.warn('Shutdown timed out — forcing exit.');
    process.exit(code);
  }, 5 * 60 * 1000).unref();
};

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
process.on('unhandledRejection', (err) => {
  console.error('[unhandledRejection]', err);
});
process.on('uncaughtException', (err) => {
  console.error('[uncaughtException]', err);
  shutdown(1);
});


