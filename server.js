const path = require('path');
const fs = require('fs');
const os = require('os');
const express = require('express');
const multer = require('multer');
const compression = require('compression');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();

const PORT = process.env.PORT || 3000;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const MAX_VIDEO_SIZE_MB = Number(process.env.MAX_VIDEO_SIZE_MB) || 200;
// Base host for the Gemini REST + File API (override only for testing/proxies).
const GEMINI_API_BASE =
  process.env.GEMINI_FILE_API_BASE || 'https://generativelanguage.googleapis.com';

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

if (!GEMINI_API_KEY) {
  console.warn('Warning: GEMINI_API_KEY is not set. /api/analyze requests will fail.');
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

app.post('/api/capture-frameset-frame-v2', express.raw({ type: 'image/png', limit: '20mb' }), async (req, res) => {
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
  constructor(message, status = 502) {
    super(message);
    this.name = 'GeminiError';
    this.status = status;
  }
}

// Resumable upload of a temp file to the Gemini File API. Returns the file
// resource ({ name, uri, mimeType, state }).
async function geminiUploadFile(filePath, mimeType, sizeBytes, displayName, signal) {
  const startRes = await fetch(`${GEMINI_API_BASE}/upload/v1beta/files?key=${GEMINI_API_KEY}`, {
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
    console.error(`[analyze] upload init failed HTTP ${startRes.status}: ${await startRes.text()}`);
    throw new GeminiError(`Gemini upload could not be started (${startRes.status}).`, 502);
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
    throw new GeminiError(`Gemini upload failed (${uploadRes.status}).`, 502);
  }
  const uploaded = await uploadRes.json();
  if (!uploaded?.file?.uri || !uploaded?.file?.name) {
    throw new GeminiError('Gemini upload response was malformed.', 502);
  }
  return uploaded.file;
}

// Poll a file resource until it leaves PROCESSING. 4 min ceiling, backoff.
async function geminiWaitUntilActive(fileName, signal) {
  const deadline = Date.now() + 4 * 60 * 1000;
  let delayMs = 1000;
  // fileName is like "files/abc123"
  while (true) {
    const res = await fetch(`${GEMINI_API_BASE}/v1beta/${fileName}?key=${GEMINI_API_KEY}`, { signal });
    if (!res.ok) {
      console.error(`[analyze] file status HTTP ${res.status}: ${await res.text()}`);
      throw new GeminiError(`Gemini file status check failed (${res.status}).`, 502);
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
async function geminiDeleteFile(fileName) {
  try {
    await fetch(`${GEMINI_API_BASE}/v1beta/${fileName}?key=${GEMINI_API_KEY}`, {
      method: 'DELETE',
      signal: AbortSignal.timeout(10000)
    });
  } catch (err) {
    console.warn(`[analyze] could not delete Gemini file ${fileName}: ${err.message}`);
  }
}

app.post('/api/analyze', upload.single('video'), async (req, res, next) => {
  const videoFile = req.file;
  let geminiFileName = null;
  // 5 min overall budget covering upload + processing + generation.
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 5 * 60 * 1000);

  try {
    if (!GEMINI_API_KEY) {
      return res.status(500).json({ error: 'Server misconfiguration: missing GEMINI_API_KEY.' });
    }
    if (!videoFile) {
      return res.status(400).json({ error: 'File is required.' });
    }

    const promptInput = req.body.prompt?.trim();
    const prompt = promptInput || DEFAULT_PROMPT;
    const mimeType = videoFile.mimetype || 'application/octet-stream';
    const isImage = mimeType.startsWith('image/');

    let payload;
    if (isImage) {
      const buffer = await fs.promises.readFile(videoFile.path);
      payload = {
        contents: [
          {
            role: 'user',
            parts: [
              { text: prompt },
              { inlineData: { mimeType, data: buffer.toString('base64') } }
            ]
          }
        ]
      };
    } else {
      // 1. Upload the temp file, 2. wait until ACTIVE.
      const uploaded = await geminiUploadFile(
        videoFile.path,
        mimeType,
        videoFile.size,
        videoFile.originalname || 'video',
        controller.signal
      );
      geminiFileName = uploaded.name;
      const activeFile = await geminiWaitUntilActive(geminiFileName, controller.signal);

      payload = {
        contents: [
          {
            role: 'user',
            parts: [
              { text: prompt },
              { fileData: { mimeType: activeFile.mimeType || mimeType, fileUri: activeFile.uri } }
            ]
          }
        ]
      };
    }
    const genRes = await fetch(
      `${GEMINI_API_BASE}/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      }
    );

    if (!genRes.ok) {
      const errorText = await genRes.text();
      let geminiMessage = errorText;
      try {
        geminiMessage = JSON.parse(errorText)?.error?.message || errorText;
      } catch (_) {}
      console.error(`[analyze] Gemini generateContent HTTP ${genRes.status}: ${geminiMessage}`);
      // Sanitized — raw Gemini payload stays in the server log only.
      return res
        .status(502)
        .json({ error: `Gemini API error (${genRes.status}): ${geminiMessage}` });
    }

    const result = await genRes.json();
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

    res.json({
      resultText: output.join('\n\n') || 'AI did not return any text.',
      raw: result
    });
  } catch (err) {
    if (err?.name === 'AbortError' || err?.type === 'aborted') {
      console.error('[analyze] aborted: 5 min budget exceeded');
      return res.status(504).json({ error: 'Analysis timed out. Try a shorter video.' });
    }
    if (err instanceof GeminiError) {
      console.error(`[analyze] ${err.message}`);
      return res.status(err.status).json({ error: err.message });
    }
    return next(err);
  } finally {
    clearTimeout(abortTimer);
    // 4. Cleanup: remove the Gemini-side file and the local temp file.
    if (geminiFileName) {
      await geminiDeleteFile(geminiFileName);
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

app.get('/healthz', (_req, res) => {
  res.json({ ok: true, gemini: Boolean(GEMINI_API_KEY) });
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
    console.error('Unhandled error:', err);
    return res.status(500).json({ error: 'Internal server error', details: err.message });
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


