# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Automation Rules for Claude

Whenever you propose or apply changes to the source code in this repository:

1. **Branches**
   - Never commit directly to `main` or `develop`.
   - If the change is not already on a feature branch, create or use a feature branch:
     - Name format: `feature/<short-description>`.
   - Ensure the base branch (e.g. `main`, `develop`) is recorded in `README.md` under "Active Branches".
   - Ensure any git command will break changes history and lose source code state.

2. **Docs: CLAUDE.md and PLAN.md**
   - If the change affects architecture, behavior, or scope:
     - Update `CLAUDE.md` to reflect the new current state and constraints.
   - If the change affects the implementation sequence:
     - Update `PLAN.md` to reflect completed tasks and next steps.
   - Always show updated versions of these files alongside code changes.

3. **Git history**
   - For each logical change:
     - Provide a `git commit` command with a clear message.
     - Mention in the commit message when `CLAUDE.md` and `PLAN.md` are updated.
   - Example:
     - `git commit -am "Add rate limiting to user API; update CLAUDE.md and PLAN.md"`

4. **Output format**
   - When proposing changes, always include:
     - Modified source files.
     - Modified `CLAUDE.md`.
     - Modified `PLAN.md`.
     - Suggested branch name and base branch.
     - Suggested commit message(s).

## Plan Mode

- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.


## Commands

```bash
npm run dev      # Start development server with auto-reload (nodemon)
npm start        # Start production server
```

No build, lint, or test scripts exist. The server runs on port 3000 by default.

## Environment Setup

Copy `.env.example` to `.env` and set:
- `GEMINI_API_KEY` — required for AI analysis
- `GEMINI_MODEL` — defaults to `gemini-2.5-flash`
- `MAX_VIDEO_SIZE_MB` — upload size limit (default: 200)
- `PORT` — server port (default: 3000)

## Architecture

**Stack:** Node.js + Express backend, vanilla JS + MediaPipe frontend, Google Gemini AI.

### Key Files

| File | Purpose |
|------|---------|
| `server.js` | Express server, `POST /api/analyze` (Gemini), `POST /api/archive-clip` (save webcam clip), `GET /api/library` |
| `public/script.js` | All app logic: MediaPipe CV, video upload, AI analysis, circumplex diagram, state machine, button handlers |
| `public/styles.css` | All styles: dark theme, workspace layout, loader overlay |
| `public/index.html` | HTML structure: `#loader-overlay` + `#workspace-root` |
| `public/analytics.html` | Separate analytics dashboard page (not part of main flow) |

### Backend (`server.js`)

- Middleware: `compression()` (gzip) runs before the static handlers. `express.static` serves `public/` with `maxAge: '5m'` and `/assets` with `maxAge: '7d'`, both with etag/lastModified revalidation. `express.json`/`urlencoded` bodies are capped at 2 MB.
- `POST /api/analyze` — uploads the video via the **Gemini File API** (not inline base64). Flow: Multer **disk** storage (temp file in `os.tmpdir()`, so a large upload never sits in RAM) → resumable upload to `/upload/v1beta/files` → poll the file until `state === 'ACTIVE'` (4 min cap, backoff) → `generateContent` with `fileData: { fileUri, mimeType }` → returns `{ resultText, raw }`. `finally` deletes the Gemini-side file and unlinks the temp file. The whole request is bounded by a 5 min `AbortController` (504 on abort). Gemini error responses are sanitized to a single `error` string — the raw payload stays in the server log only.
- If client sends a `prompt` field, it fully replaces `DEFAULT_PROMPT`. Empty prompt = server default.
- `GEMINI_FILE_API_BASE` env var overrides the Gemini API host (defaults to `https://generativelanguage.googleapis.com`).
- `POST /api/capture-frame` — Accepts raw `image/png` (limit 20 MB) with `?filename=frame_<base>_<mm>-<ss>.png`. Saves to `assets/export/frames/`. Written via an atomic `'wx'` open; if the target name exists the server appends ` (copy N)` (1-indexed, walking until free) so previous captures are never overwritten and concurrent clicks cannot collide. Returns `{ ok, name, path }` where `name` is the actually-saved filename.
- `POST /api/capture-frameset-frame-v2` — Accepts raw `image/png` (limit 20 MB) with `?dir=<path>&filename=<name>`. `dir` may be absolute (e.g. `/Users/me/Desktop/out`) or relative; **relative paths must resolve inside the project root** (a `../` escape returns `400 Invalid dir`). The folder is created via `mkdir -p` if missing. The filename is written as-is (no dedup — caller controls naming). Returns `{ ok, path }`. Used by the frame-set export when the user types a destination path; the FSAA picker path bypasses the server and writes via the browser's File System Access API.
- `POST /api/archive-clip` — Accepts a raw `video/webm` **or** `video/mp4` blob (limit 50 MB). Saves to `assets/archive/library/` with timestamped filename `live_YYYY-MM-DD_HH-mm-ss.<ext>` (extension follows the request Content-Type — webm on Chrome/Firefox, mp4 on Safari). Returns `{ ok, name, path }`. (Historical filenames may still use the `exhibition_` prefix.)
- `GET /api/library` — Lists files in `assets/archive/library/` (videos + images).
- `GET /healthz` — Liveness probe; returns `{ ok: true, gemini: <boolean> }`.
- Server timeouts are widened for large uploads (`keepAliveTimeout` 65 s, `headersTimeout` 70 s, `requestTimeout` 10 min). `SIGINT`/`SIGTERM` trigger a graceful shutdown that drains in-flight requests (`server.close` + `closeIdleConnections`).

### MediaPipe models

- Models load **local-first** from `/assets/models/mediapipe/` with a transparent CDN fallback (`storage.googleapis.com`). Run `npm run fetch-models` (or `scripts/download-mediapipe-models.sh`, also wired to `postinstall`) to populate the directory; the `.task`/`.tflite` files are gitignored.
- Face/hand/pose detectors initialise eagerly at boot (default on). Object, face-detection and gesture detectors are **lazy** — initialised on first toggle-enable.

---

## App State Machine

Variable `appState` in `script.js` drives the entire UI. Two states:

```
loading ──(2s timeout / click)──► workspace ──(switchMode('live'))──► Live
```

### State: `loading`

- **What's visible**: Full-screen black `#loader-overlay` with the spinning AEMA logo (`#loader-logo` — `.loader-logo-img` masked by `.loader-logo-shine`) centered. No video, no progress bar.
- **`#workspace-root`**: hidden behind overlay.
- **Transition**: 2-second `setTimeout(endLoader, 2000)` OR click on overlay → `endLoader()` → hide overlay, call `showWorkspace()`.

### State: `workspace`

- **Trigger**: Loader auto-dismisses after 2s or user clicks overlay.
- **Transition** (`showWorkspace()`):
  1. `workspaceRoot.hidden = false` — shows standalone workspace.
  2. `form.classList.remove('hidden')` — shows controls sidebar.
  3. Shows `showAnalyticsBtn`, sets text to "View Analytics".
  4. **Resets analytics panel to closed state**: `outputsPanel.hidden = true`, `submitBtn.style.display = 'none'`, removes `analytics-visible` class.
  5. `appState = 'workspace'`.
  6. Forces entry into Live mode: `workspaceMode = null; switchMode('live')` (the null reset is required because `switchMode` early-returns when `mode === workspaceMode`).

---

## HTML Structure

Two top-level containers:

```
<body>
  <div id="loader-overlay">           ← visible during loading state
    <div id="loader-logo">             ← spinning AEMA logo (no video)
  </div>

  <div id="workspace-root" hidden>     ← visible in workspace state
    <main class="container">
      <section class="workspace">
        <form id="analyze-form">       ← controls sidebar (left)
        <div class="center-column">    ← center column wrapper
          <div class="players-panel">  ← video players (hidden when no video)
          <div id="analytics-bottom">  ← analytics panel (independent of players-panel)
        </div>
      </section>
      <div id="fullscreen-overlay">    ← fullscreen result view
    </main>
  </div>
</body>
```

---

## Workspace Layout (3-column)

### Left: Controls Sidebar (`#analyze-form`, 240px fixed)

Three collapsible sections using `<details>`/`<summary>`:

1. **Video** — file upload input. Label changes "Select video" → "Change video" after selection. File name hint shown below.
2. **Computer vision** (collapsed by default) — toggles for: video background, inverted mode, face/hand/pose landmarks, object detection, hand gestures, face detection, face style (mesh/dots), pose joints, pose trails.
3. **Nonverbal analysis** (collapsed by default) — contains "View Analytics" and "Behavior Analysis" buttons.

Sidebar starts with `.hidden` class, shown when entering workspace.

### Center: `.center-column` wrapper

Contains two siblings in a 50/50 flex split:
1. **`.players-panel`** (`flex: 1 1 50%`) — Hidden on page load (`hidden` attribute). Shown only after first video selected. Contains canvas + transport bar.
2. **`#analytics-bottom`** (`.analytics-bottom-panel`, `flex: 1 1 50%`) — Hidden by default. Toggled by "View Analytics" button. Independent of players-panel visibility (stays visible even without video). Two tabs: "Emotions AI" (`#tab-data`) and "Behavior Analysis" (`#tab-ai`). When players-panel is hidden, analytics takes full height.

---

## Workspace Modes

Three modes controlled by `workspaceMode` variable and mode bar buttons:

### Processing (`data-mode="edit"`)
- Shows `#analyze-form` sidebar. **Not** the default — Live is.
- User uploads video/image, MediaPipe processes it, can send for Gemini analysis.
- **Sharp / hi-DPI overlay**: `#landmark-canvas` backing store auto-sizes per frame inside `updateCanvasDimensions()` to `max(cssWidth × devicePixelRatio, videoNative)`, capped at `MAX_CANVAS_WIDTH = 3840` (4K width). Aspect ratio is locked to the source. `renderScale = canvasWidth / 1280` is recomputed on every resize and feeds every overlay's `lineWidth` / dot `radius` (face mesh, face dots, hand connectors+joints, pose connectors+joints, pose trails, torso fill, object/face-detection boxes+labels), so stroke thickness stays perceptually constant across resolutions. Per-frame video paint uses `imageSmoothingQuality = 'high'`. Same logic runs in Archive and Live.
- **Inverted mode** (`#toggle-inverted-mode`, off by default): when enabled, the canvas gets the CSS class `.inverted-mode` which applies `filter: grayscale(100%) invert(100%)` on the GPU compositor — this gives the negative grayscale of the source at native FPS without per-frame Skia software filtering. The overlay draw functions (`drawFaceLandmarks`, `drawHandLandmarks`, `drawPoseLandmarks`, `drawObjectDetections`, `drawFaceDetections`) already paint in `#FFFFFF`, so the same CSS invert flips them to black for free — no `landmarkCtx.filter` per overlay draw is needed. The class is kept in sync inside `analyzeFaceFrame()` and the toggle's `change` listener. Effect is gated by `workspaceMode === 'edit'` so it never activates in Archive or Live. Tradeoff: pixels read via `getImageData` are pre-CSS-filter (raw color); the CSS filter is applied only at composition for display. **Capture frame / frameset**: `render4KFrame()` writes to an offline canvas that the CSS rule cannot reach, so after all draws complete it mirrors the same gate (`invertedModeEnabled && workspaceMode === 'edit'`) by drawing the finished composite once into a fresh canvas with `outCtx.filter = 'grayscale(100%) invert(100%)'` and returning that. Single post-process pass — identical to the CSS rule which inverts the final composite once. Per-op `ctx.filter` was tried first and produced wrong output because MediaPipe `DrawingUtils` save/restores the context, dropping the filter for landmark passes; the post-process pass sidesteps that entirely.
- **Capture Frame Set popup** (`#frameset-popup`): two extra fields above From/To. **Destination folder** (`#frameset-dest`, text input + `#frameset-pick-btn` Pick button). Prefilled with `assets/export/frames/<base>_00-00_<dur>_frameset`. User can type any path (absolute or relative to project root — server creates it via `mkdir -p` and writes through `/api/capture-frameset-frame-v2`), or click Pick to open `window.showDirectoryPicker()` (FSAA). When a directory is picked, the handle is stored in `pickedDirHandle`, the input becomes read-only and shows the folder name, and frames write directly to disk via `FileSystemDirectoryHandle.getFileHandle().createWritable()` — no server roundtrip. Pick button toggles to "Clear ✕" to drop the handle and revert to typed mode. FSAA is Chromium-only; Safari/Firefox alert and fall through to typed mode. **Filename prefix** (`#frameset-prefix`): defaults to `frame_<base>`. Files are saved as `<prefix>_mm-ss.png`. Prefix is sanitized client-side to `[A-Za-z0-9_\-]` (other chars → `_`); empty falls back to `frame`. **No duration gate**: the previous `dur > 300` (5 min) confirm was removed — every export now shows a single count-based confirm `This will export N frames. Continue?` regardless of video length.

### Archive (`data-mode="archive"`)
- Shows `#library-panel` sidebar with thumbnails from `assets/archive/library/`.
- Click item → loads into shared player with MediaPipe overlay.
- Has "Nonverbal analysis" button for emotion circumplex.

### Live (`data-mode="live"`) — **default mode on boot**
- No sidebar — the live stream takes the full workspace width. Analytics panel closed on entry (`closeAnalyticsPanel`) so player fills full height.
- Starts webcam via `getUserMedia` → streams to `previewEl.srcObject`.
- MediaPipe overlay runs on live feed via existing `analyzeFaceFrame()` loop. Face + pose + hand landmarks auto-enabled on entry.
- **Rolling 10-second buffer.** `MediaRecorder` records raw webcam (no overlay) in 1s chunks. Every `LIVE_WINDOW_MS = 10000` a rotation timer calls `rotateCacheRecording()`: stop the recorder (await `onstop` so the container is fully finalized → playable WebM/MP4), snapshot `cacheChunks` as `previousWindowBlob`, then start a fresh recorder. The rotation guarantees that the previously-stored 10s blob has a valid EBML/MP4 header — a naive `cacheChunks.slice(-10)` would not.
- **Floating "Save & analise 10s" overlay** (`#live-overlay`, bottom-left of `.player-card`): pill-shaped `#live-save-btn` + inline `#live-status` pill. Click flow:
  1. Disable button; status "Saving…".
  2. Clear rotation timer, stop current recorder, await `onstop` → `currentBlob`.
  3. Pick `currentBlob` if `cacheChunks.length >= 3` (≥3s of fresh material), else `previousWindowBlob`. Restart cache recording.
  4. POST blob to `/api/archive-clip`.
  5. Status "Analyzing…". POST same blob as multipart `FormData(video, prompt=default preset)` to `/api/analyze`.
  6. On success: `resultText.innerHTML = formatAnalysisResponse(payload.resultText)`, `openAnalyticsAiTab()` (forces analytics panel open + Behavior Analysis tab visible). Status shows "Analysis ready. Open in Archive" link → `switchMode('archive')`.
- Transport bar shows `LIVE • MM:SS` indicator (`liveMode` flag + `.transport-bar--live` class hides play/pause and timeline). Reverts to normal timeline when leaving Live.
- Webcam stops on mode exit via `stopWebcam()` (also clears `liveMode`, `rotationTimer`, `previousWindowBlob`).

## Keyboard Shortcuts

- **`f`** — toggle player fullscreen (when over a video).
- **`Shift+F`** — toggle document fullscreen.
- **`Cmd/Ctrl + +` / `=`** — increase root font size (clamp 32px). Hidden, no UI.
- **`Cmd/Ctrl + -`** — decrease root font size (clamp 10px).
- **`Cmd/Ctrl + 0`** — reset font size to 16px baseline.
- Font-size choice persists across reloads via `localStorage['fontSizePx']`. Shortcut is suppressed while typing in inputs/textareas/contenteditable.

## Checkbox styling

All `.controls-panel .toggle input[type="checkbox"]` use a custom PNG-backed visual instead of the native UA checkbox:
- Unchecked: `/assets/checkbox.png`
- Checked: `/assets/checkbox_crossed.png`
Both are 512×512 white-on-transparent and rendered at `1rem × 1rem`. Preloaded from `index.html` so the first toggle flip doesn't flash.

---

## Button Logic (CRITICAL — must maintain these invariants)

### "View Analytics" / "Hide Analytics" (`#show-analytics-btn`)

- **Location**: Inside "Nonverbal analysis" dropdown in sidebar.
- **Initial state**: `style="display: none"`, text "View Analytics". Shown (`display: inline-flex`) when entering workspace via `showWorkspace()`.
- **Click behavior** (toggle):
  - **If panel hidden** (`outputsPanel.hidden === true`):
    - `outputsPanel.hidden = false` — show right panel.
    - `submitBtn.style.display = 'inline-flex'` — show "Behavior Analysis" button.
    - `enableFaceLandmarks()` — auto-enable face landmarks if unchecked.
    - Reset to "Emotions AI" tab: `tabData` active, `tabAi` hidden, `viewData` visible, `viewAi` hidden.
    - Button text → "Hide Analytics".
    - `.workspace` gets class `analytics-visible`.
    - Smooth scroll to panel.
  - **If panel visible** (`outputsPanel.hidden === false`):
    - `outputsPanel.hidden = true` — hide right panel.
    - `submitBtn.style.display = 'none'` — hide "Behavior Analysis" button.
    - `tabAi.hidden = true`.
    - Button text → "View Analytics".
    - `.workspace` loses class `analytics-visible`.
- **Invariant**: Button text always reflects current panel state. Button itself is always visible while in workspace.

### "Behavior Analysis" (`#submit-btn`)

- **Location**: Inside "Nonverbal analysis" dropdown, below "View Analytics".
- **Initial state**: `style="display: none"`. Only shown when analytics panel is open.
- **Click behavior**:
  - `tabAi.hidden = false`, add `active` class.
  - `tabData` loses `active` class.
  - `viewAi.hidden = false`, `viewData.hidden = true` — switch to Behavior Analysis tab.
  - `aiControls.hidden = false` — ensure AI controls visible.
- **Visibility rule**: Visible (`display: inline-flex`) ONLY when `outputsPanel` is not hidden. Hidden when panel closes.

### Tab: "Emotions AI" (`#tab-data`)

- Click → `tabData` active, `tabAi` inactive, `viewData` shown, `viewAi` hidden.
- Default active tab when analytics panel opens.

### Tab: "Behavior Analysis" (`#tab-ai`)

- **Starts hidden** (`hidden` attribute). Only un-hidden when "Behavior Analysis" is clicked.
- Click → `tabAi` active, `tabData` inactive, `viewAi` shown, `viewData` hidden.
- Re-hidden when analytics panel closes.

### Prompt presets (`#prompt-preset`)

- Native `<select>` rendered above "Customize prompt" inside `#ai-controls`. Options come from the `PROMPT_PRESETS` array in `script.js` (each entry: `{ id, name, prompt }`). Default selection = `DEFAULT_PRESET_ID` (`'full-nonverbal'`).
- On init: textarea `#prompt` is seeded once with the default preset's text (no longer lazy-seeded on first textarea open).
- On preset change: textarea value is replaced unconditionally with the chosen preset's prompt. Textarea visibility is **not** toggled — that stays under `#toggle-prompt` control.
- `runAnalysis()` continues to send `promptField.value`; presets are purely a client-side authoring convenience.
- To add a new preset: append an object to `PROMPT_PRESETS` in `script.js` (right after `DEFAULT_PROMPT`). No HTML or CSS changes needed.

### "Customize prompt" (`#toggle-prompt`)

- Toggles `#prompt` textarea visibility (`hidden` attribute).
- Variable `promptVisible` tracks state.
- Textarea is pre-populated with the active preset (see "Prompt presets" above) so the user immediately sees the prompt that will be sent.

### "Send for Analysis" (`#send-analysis-btn`)

- Click → `runAnalysis()`:
  1. Validate video selected and file size ≤ 100MB.
  2. Hide `aiControls`, show status "Uploading...".
  3. Build FormData with video + custom prompt.
  4. `POST /api/analyze`.
  5. Parse response → `formatAnalysisResponse()` → render in `#result-text`.
  6. Show status "AI response ready." or error.
  7. Re-enable button in `finally`.

### Fullscreen result (`#fullscreen-result-btn` / `#fullscreen-close-btn`)

- Open: copies `resultText.innerHTML` into `#fullscreen-result-content`, shows `#fullscreen-overlay`.
- Close: hides overlay. Also closes on Escape key.

### Video file input (`#video`)

- On change → `handleVideoSelection()`:
  - If file: show `.players-panel`, load into preview, update label to "Change video", show filename hint.
  - If no file: hide `.players-panel`, clear preview.

### "Background" (`#toggle-video-bg`)

- **Checked by default** → `showVideoBackground = true` (video drawn as canvas background).
- Unchecked → `showVideoBackground = false`, reveals "Choose background" button for custom still background.
- `showVideoBackground = event.target.checked` (direct mapping, not inverted).

---

## Circumplex Diagram

### Workspace Circumplex (`#circumplex-svg` + `#emotion-trail-canvas`, inside `.circumplex-stage`)

- **Visual base**: Custom Illustrator SVG at `assets/circumplex_diagram.svg` (viewBox `0 0 635.77 552.25`), embedded via `<object>` so its DOM is scriptable. Provides quadrants, axes, labels, and the dot pointer.
- **Live pointer**: SVG element `#pointer` is moved each frame via `transform="translate(...) scale(pulse) ..."` — no canvas re-paint, only attribute mutation.
- **Trail**: A transparent `<canvas id="emotion-trail-canvas">` overlay above the SVG renders the fading breadcrumb dots (cheap clear+redraw each frame). Sized via `ResizeObserver` to its layout pixels.
- **Required SVG IDs** (re-export from Illustrator with named layers):
  - `pointer` — the dot moved to (v, a)
  - `axes-circle` — the outer reference ring; bbox gives center `(cx, cy)` + radius `R` for the v/a → SVG coordinate mapping
  - `emotion-<label>` (e.g. `emotion-happy`, `emotion-excited`) — optional groups; their bbox centroids extend the `EMOTIONS` array used by `getDominantEmotion`. If absent, the default 6 Ekman emotions are kept.
- **Bootstrap**: `bootstrapCircumplexSvg()` runs on `<object>` `load`, reads `getBBox()` of `#axes-circle` and `#pointer`, builds the extended `EMOTIONS` list from `[id^="emotion-"]` groups, and sets `svgState.ready = true`. If required IDs are missing it logs a warning and the pointer simply doesn't move.
- **Data source**: Video playback FaceLandmarker blendshapes → `updateEmotionWheel()` → `renderEmotionWheel()` sets targets.
- **Animation**: `animateEmotionWheel()` runs via `requestAnimationFrame`. Lerps `wsValence`/`wsArousal` towards `wsTargetValence`/`wsTargetArousal` (factor `WS_LERP = 0.08`).
- **Trail buffer**: Points pushed every 5 frames, max 30 points.
- **Visible in**: `workspace` state, inside "Emotions AI" tab of analytics panel.
- **Removed in this revision**: legend (`#ekman-legend-workspace`), V/A numeric readout (`#emotion-result-card`), label (`#emotion-wheel-name`), and all canvas-drawn art (background ring, quadrant tints, dashed crosshair/ring, canvas-drawn Ekman markers, canvas-drawn pulsing pointer). Functions `buildWorkspaceLegend` / `updateWorkspaceLegend` removed.

---

## AI Response Pipeline

1. Client sends video + prompt to `POST /api/analyze`.
2. Server: `promptInput || DEFAULT_PROMPT` — if client sends empty string, server default used.
3. Gemini returns markdown text.
4. Client: `formatAnalysisResponse(text)` parses:
   - `---` → `<hr>`
   - `## heading` → `<h4>`, `### heading` → `<h5>`
   - `0. Title` / `1. Title` → `<h4>`, `1.1 Subtitle` → `<h5>`
   - `* bullet` / `- bullet` → `<li>` inside `<ul>`
   - `**bold**` → `<strong>`, `*italic*` → `<em>`, `` `code` `` → `<code>`
   - Everything else → `<p>`
5. Result displayed in `#result-text`, also copyable to fullscreen overlay.

---

## CSS Conventions

- **`hidden` attribute conflict**: Several elements use `display: flex` in CSS which overrides `[hidden]`. Each needs an explicit `[hidden] { display: none }` rule. Already done for: `.outputs-panel`, `.workspace-root`, `.fullscreen-overlay`, `.panel-view`, `.loader-overlay`.
- **Loader overlay**: Fixed position at `z-index: 200`, covers viewport with loading video. Hidden via `[hidden]` after video ends.
- **Workspace visibility**: Toggled via `hidden` attribute on `#workspace-root` (a fixed-position overlay at `z-index: 100`).
