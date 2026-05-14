AEMA Production-Level Performance & Reliability Optimization

 Context

 Three parallel Explore audits + direct file reads surfaced ~50 issues spanning the server, MediaPipe/video pipeline, and asset/loading layer. The user wants production-readiness for
 smooth playback, framerates, MediaPipe overlay performance, AI API communication, with all existing functionality preserved — safe, incremental changes only.

 Three user-confirmed decisions shape this plan:
 1. MediaPipe models: serve locally from /assets/models/mediapipe/ with CDN fallback at runtime.
 2. /api/analyze: switch from inline-base64 multer-memoryStorage to Gemini File API (multer disk storage ? upload ? poll ? generate ? cleanup). Eliminates ~500 MB peak RAM per upload.
 3. Exhibition recorder: add bitrate cap to MediaRecorder (preserves "save since webcam start" UX; reduces RAM growth ~2–3?).

 Current state observations to act on (verified against source):
 - analyzeFaceFrame (script.js:1088) sets imageSmoothingEnabled + imageSmoothingQuality='high' every frame (line 1111-1112) — wasted work on a 3840?2160 canvas at 60 Hz.
 - updateCanvasDimensions + updatePlayerOrientation (script.js:1102-1103) called every frame; orientation toggles classes unconditionally.
 - render4KFrame (script.js:2337) constructs a new DrawingUtils on every export call (line 2353).
 - Exhibition MediaRecorder (script.js:1893) has no videoBitsPerSecond constraint.
 - showBlobInPreview (script.js:1671-1690) adds 4 video listeners per re-upload, clearPreview (l.1717) removes none.
 - URL.createObjectURL (l.1675, 1736) without revoking the previous URL first.
 - server.js: no compression, no Cache-Control on express.static, no AbortController/timeout on Gemini fetch, sync fs.*Sync on hot paths, TOCTOU race in /api/capture-frame (l.127-131),
  path-traversal risk in /api/capture-frameset-frame-v2 (l.139), default keep-alive/header timeouts, no graceful shutdown, no healthcheck.
 - index.html: no preconnect / dns-prefetch to MediaPipe CDN, Google Cloud Storage, or Gemini API.
 - MediaRecorder mime hardcoded to video/webm (script.js:1893) — Safari fails.

 ---
 Critical files to be modified

 - server.js — compression, cache headers, Gemini File API refactor, server timeouts, async fs, atomic writes, path-traversal guard, graceful shutdown, healthcheck, error handlers
 - public/script.js — per-frame hoisting, listener cleanup, MediaRecorder MIME negotiation + bitrate, DrawingUtils cache, URL.revoke ordering, lazy init for opt-in detectors,
 local-first MediaPipe path with CDN fallback
 - public/index.html — preconnect / dns-prefetch hints, aria-labels on icon-only buttons (a11y light pass)
 - public/styles.css — will-change: transform on loader 3D elements
 - package.json — add compression dependency, postinstall model-download script
 - scripts/download-mediapipe-models.sh — new, one-shot bootstrap fetcher for .task files into /assets/models/mediapipe/
 - .gitignore — add assets/models/mediapipe/ so binary weights don't bloat git
 - .env.example — document new optional GEMINI_FILE_API_BASE knob (defaults to standard endpoint)

 ---
 Tiered edits

 Tier A — MediaPipe overlay / per-frame hot path (highest perceived perf win)

 A1. Hoist per-frame canvas property writes — script.js:1111-1112
 Move landmarkCtx.imageSmoothingEnabled = true; landmarkCtx.imageSmoothingQuality = 'high'; out of analyzeFaceFrame. Apply them inside updateCanvasDimensions only when the canvas
 backing-store actually changes (immediately after canvas.width = targetW assignment at script.js:645-647). Save ~120 writes/sec; importantly avoids invalidating the smoothing-quality
 state machine each frame.

 A2. Skip no-op orientation toggles — script.js:651-664
 Cache previous isLandscape in a closure-scoped boolean; early-return when unchanged so classList.toggle doesn't fire 60?/sec.

 A3. Cache DrawingUtils for the capture canvas — script.js:2337-2386
 Hoist a module-scoped capDrawingUtilsByCanvas WeakMap<HTMLCanvasElement, DrawingUtils>. Reuse the existing instance when the offline canvas dimensions match the previous capture;
 create new only on resize. Reduces frameset-export allocation churn.

 A4. Lazy-init opt-in detectors — script.js:1234, 1257, 1279, and the three currently-defaulted-off detectors
 Face, hand, pose are checked by default — keep their eager init. Move objectDetector, gestureRecognizer, faceDetector init into the corresponding toggle change listeners (init on
 first true). Saves three CDN fetches at page load.

 A5. Local-first MediaPipe model loading with CDN fallback — script.js:1212-1334
 Define const MEDIAPIPE_LOCAL_BASE = '/assets/models/mediapipe';. For each createFromOptions, first try modelAssetPath: '${LOCAL}/face_landmarker.task'; on the catch block, retry with
 the existing Google CDN URL and log a one-time warning. FilesetResolver.forVisionTasks keeps its current CDN URL (the WASM is small and cached). Pair with
 scripts/download-mediapipe-models.sh + postinstall.

 Tier B — Network / AI API (the user's "Communication with AI API" focus)

 B1. compression middleware — server.js:8
 Add const compression = require('compression'); and app.use(compression()); before the static middlewares. Compresses HTML/CSS/JS/JSON responses. Add compression: ^1.7.4 to
 package.json.

 B2. Cache-Control on static assets — server.js:84-86
 - app.use(express.static(path.join(__dirname, 'public'), { etag: true, lastModified: true, maxAge: '5m' })) — small TTL on app shell (it changes).
 - app.use('/assets', express.static(path.join(__dirname, 'assets'), { etag: true, lastModified: true, maxAge: '7d', immutable: false })) — videos, SVGs, fonts, model .task files all
 benefit. (Switch to maxAge: '365d', immutable: true once filenames are content-hashed; not in scope.)
 - /uploads untouched (transient).

 B3. Gemini File API refactor — server.js:71-76, 168-247
 Switch multer.memoryStorage() ? multer.diskStorage() writing to os.tmpdir() with cleanup-in-finally. Replace the inline-base64 inlineData payload with a 3-step flow against the Gemini
  File API:
 1. POST /upload/v1beta/files?uploadType=multipart&key=… — stream the temp file as multipart body.
 2. Poll the returned file.name via GET /v1beta/files/{name} until state === 'ACTIVE' (with backoff + total timeout 4 min).
 3. POST /v1beta/models/{model}:generateContent with fileData: { fileUri, mimeType } (replaces inlineData).
 4. finally: DELETE /v1beta/files/{name} and fs.promises.unlink(tempFilePath).

 Wrap all three calls in AbortController with AbortSignal.timeout(300_000) (5 min). Return 504 on AbortError; sanitize Gemini error responses to a single safe error string (don't echo
 raw geminiResponse in body — keep that server-side only).

 B4. Server timeouts — server.js:271-273
 Capture const server = app.listen(...), then server.keepAliveTimeout = 65_000; server.headersTimeout = 70_000; server.requestTimeout = 600_000; (10 min for large uploads + Gemini
 processing).

 B5. JSON body limit — server.js:82
 app.use(express.json({ limit: '2mb' })) so longer custom prompts don't 413.

 B6. MediaRecorder MIME negotiation + bitrate cap — script.js:1893
 const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4;codecs=h264', 'video/mp4'];
 const mimeType = candidates.find((t) => MediaRecorder.isTypeSupported(t)) || '';
 mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 2_500_000 } : { videoBitsPerSecond: 2_500_000 });
 Restores Safari/Firefox compatibility and caps cacheChunks growth (~750 KB/min vs current ~3 MB/min).

 Tier C — Stability / memory / safety

 C1. Atomic capture-frame write — server.js:117-133
 Replace the existsSync + sync write with fs.promises.open(path, 'wx') + await handle.writeFile(req.body) + await handle.close(); on EEXIST, increment the (copy N) counter and retry.
 Eliminates the TOCTOU race when two clicks fire simultaneously.

 C2. Path-traversal guard on frameset endpoint — server.js:135-147
 For relative dir, const resolved = path.resolve(__dirname, dir); then if (!resolved.startsWith(__dirname + path.sep)) return res.status(400).json({ error: 'Invalid dir' });. Absolute
 paths remain explicit user intent (local dev tool).

 C3. Async fs throughout — server.js:97-115, 141-142, 149-166
 Convert all handlers to async (req, res, next) and use fs.promises.readdir, fs.promises.writeFile, fs.promises.mkdir. Boot-time mkdirSync (l.89, 92) is fine — runs once.

 C4. Graceful shutdown + process error handlers — server.js: end of file
 process.on('unhandledRejection', (err) => console.error('[unhandledRejection]', err));
 process.on('uncaughtException', (err) => { console.error('[uncaughtException]', err); shutdown(1); });
 const shutdown = (code = 0) => server.close(() => process.exit(code));
 ['SIGINT', 'SIGTERM'].forEach((sig) => process.on(sig, () => shutdown(0)));

 C5. Healthcheck endpoint — server.js: near the other routes
 app.get('/healthz', (_, res) => res.json({ ok: true, gemini: Boolean(GEMINI_API_KEY) }));

 C6. Preview-element listener cleanup — script.js:1655-1690, 1717-1729
 Hoist onLoaded, onError, tryPlay, tryPlayCanPlay into module-scoped refs (e.g., a previewListeners object). In clearPreview(), explicitly removeEventListener for each. Same treatment
 for the analogous handlers in showImageInPreview (script.js:1738+).

 C7. Revoke previous object URL before creating new — script.js:1675, 1736
 Call revokePreviewUrl() (existing helper) at the top of both showBlobInPreview and showImageInPreview, before the new URL.createObjectURL. Stops the per-upload memory bump on rapid
 re-uploads.

 C8. Capture-menu listener cleanup on mode switch — script.js:switchMode
 Call closeCaptureMenu() from switchMode so dangling document listeners (script.js:2170-2181) are removed when entering Archive or Exhibition.

 Tier D — Loading / boot speed / a11y

 D1. Resource hints — index.html: <head> (after l.7)
 <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
 <link rel="preconnect" href="https://storage.googleapis.com" crossorigin>
 <link rel="dns-prefetch" href="//generativelanguage.googleapis.com">
 <link rel="preload" as="image" href="/assets/AEMA_logo.svg" type="image/svg+xml">

 D2. will-change on loader 3D animation — styles.css: .loader-logo, .loader-logo-img, .loader-logo-shine
 Add will-change: transform; (logo + shine sweep) and will-change: width; (progress bar). Confines compositing to GPU layer.

 D3. Light a11y pass — index.html
 Add aria-label to icon-only buttons: transport-play, transport-pause-icon, fullscreen open/close, exhibition-archive-btn, capture-frame-menu-toggle. Tab buttons get role="tab" +
 aria-selected. No behavior change.

 D4. loader-video preload — index.html:16
 Leave preload="auto" (the loader video is the first thing the user sees; eager loading is correct).

 ---
 New files

 scripts/download-mediapipe-models.sh

 Idempotent curl script downloading the 6 .task files from storage.googleapis.com/mediapipe-models/... into assets/models/mediapipe/. Skips files already present. Runs in postinstall.

 .gitignore addition

 assets/models/mediapipe/*.task

 package.json changes

 - Add "compression": "^1.7.4" to dependencies.
 - Add "postinstall": "bash scripts/download-mediapipe-models.sh || true" (the || true keeps npm install succeeding on networks where the download fails — runtime CDN fallback still
 works).

 ---
 Verification

 Automated (Playwright MCP)

 1. Cold boot timings: browser_navigate http://localhost:3000, capture performance.timing + performance.getEntries(). Compare TTI / FCP before vs after Tier B+D changes. Expect FCP
 improvement from preconnect + compression.
 2. Compression: browser_evaluate fetch('/').then(r => r.headers.get('content-encoding')) ? expect gzip/br.
 3. Cache headers: second-load network audit — assets/circumplex_diagram.svg, assets/AEMA_logo.svg, assets/loading/loading.mp4, assets/models/mediapipe/*.task should all return 304 Not
  Modified.
 4. MediaPipe overlay: open a sample 1080p video, enable face+pose+hand+inverted. Use Chrome DevTools Performance recording (30 s). Compare: avg frame time, % long tasks, GPU
 rasterizer time. Before/after target: ~25 % main-thread frame time reduction from A1+A2.
 5. renderScale invariant: confirm renderScale === 1.5 after the changes (regression guard).
 6. Capture frame: rapid 5? click — browser_evaluate lists /assets/export/frames/ directory, confirm (copy N) semantics intact with atomic write.
 7. Frame set with .. path: try dir=../../etc via the input ? expect 400 Invalid dir (path-traversal guard).
 8. Gemini analyze: submit a small video ? confirm response intact (Gemini File API path). Mock the upload endpoint to 503 ? confirm 504 clean response and tempfile cleanup. Submit a
 500 KB custom prompt ? confirm no 413 (body limit).
 9. MediaRecorder MIME: stub MediaRecorder.isTypeSupported to return false for webm in browser_evaluate ? confirm mp4 path is selected without errors.
 10. Listener leak guard: re-upload 5 videos; previewEl.getEventListeners ?? new MutationObserver… (or read via DevTools) ? confirm no growth.
 11. Healthcheck: curl -i /healthz ? 200 JSON.
 12. Graceful shutdown: kill the server with SIGTERM during an in-flight /api/analyze ? confirm the request completes before exit.

Reporting (deliverable to the user)

Before/after table with: 
 - Cold-cache page load (DOMContentLoaded, FCP, TTI)
- Warm-cache page load
- MediaPipe frame time (median, p95) @ 1080p with all three detectors on
- Server peak RSS during a 200 MB upload analysis
- cacheChunks size after 1 min of exhibition
 Gemini API timeout behavior (cold cancel path)
- Bundle delivered bytes (gzip vs raw)


Explicitly out of scope (with reasoning)

- OffscreenCanvas + Web Worker for MediaPipe — architectural change; risk to overlay correctness; user asked for "safe incremental."
- HTTP/2 / TLS — deploy-environment concern, not application code.
- Service Worker / PWA / offline — large surface; would alter loader UX.
- JS/CSS minification or bundler — would change build/deploy workflow; current 87 KB script is acceptable when gzipped.
- Mobile-specific breakpoints under 600 px — needs design review; deferred.
- Helmet / CSP / CORS — beneficial in prod but requires policy decisions (which origins/inline scripts to permit); deferred.
- Replace <object> circumplex SVG with inline <svg> — script depends on bootstrapCircumplexSvg() via the <object> load event; non-trivial refactor.
- Lower exhibition canvas resolution to 720p — would visually change overlay perception; deferred until UX review.
