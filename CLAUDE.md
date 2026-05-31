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
- `MAX_VIDEO_SIZE_MB` — upload size limit (default: 250)
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

- Middleware: `compression()` (gzip) runs before the static handlers. `express.static` serves `public/` (the app shell) with `maxAge: 0` (revalidate every load via etag — so CSS/JS edits show without a hard refresh) and `/assets` with `maxAge: '7d'`, both with etag/lastModified revalidation. `express.json`/`urlencoded` bodies are capped at 2 MB.
- `POST /api/analyze` — uploads the video via the **Gemini File API** (not inline base64). Flow: Multer **disk** storage (temp file in `os.tmpdir()`, so a large upload never sits in RAM) → resumable upload to `/upload/v1beta/files` → poll the file until `state === 'ACTIVE'` (4 min cap, backoff) → `generateContent` with `fileData: { fileUri, mimeType }` → returns `{ resultText, raw }`. `finally` deletes the Gemini-side file and unlinks the temp file. The whole request is bounded by a 5 min `AbortController` (504 on abort). Gemini error responses are sanitized to a single `error` string — the raw payload stays in the server log only.
  - **Resilient upload**: the resumable upload (`geminiUploadFile` → `geminiUploadFileOnce`) **retries transient network drops** (`EPIPE`/`ECONNRESET`/`ETIMEDOUT`/… via `isTransientNetworkError`, up to 3 attempts with 500 ms × attempt backoff, fresh session + stream each time). Aborts and real HTTP-level `GeminiError`s are not retried. A persistent drop returns a clean **503** "Could not upload the video to Gemini (connection dropped). Please try again." — fixing the unhandled `write EPIPE` seen when analysing live-captured clips.
  - **API-key redaction**: node-fetch error messages embed the full request URL (which includes `?key=<GEMINI_API_KEY>`). `redactKey()` strips `key=…` → `key=REDACTED` in the upload-failure log **and** in the generic error handler's `console.error` + `details` response, so the key never reaches logs or the client.
- If client sends a `prompt` field, it fully replaces `DEFAULT_PROMPT`. Empty prompt = server default.
- `GEMINI_FILE_API_BASE` env var overrides the Gemini API host (defaults to `https://generativelanguage.googleapis.com`).
- `POST /api/capture-frame` — Accepts raw `image/png` (limit 20 MB) with `?filename=frame_<base>_<mm>-<ss>.png`. Saves to `assets/export/frames/`. Written via an atomic `'wx'` open; if the target name exists the server appends ` (copy N)` (1-indexed, walking until free) so previous captures are never overwritten and concurrent clicks cannot collide. Returns `{ ok, name, path }` where `name` is the actually-saved filename.
- `POST /api/capture-frameset-frame-v2` — Accepts raw `image/png` (limit 20 MB) with `?dir=<path>&filename=<name>`. `dir` may be absolute (e.g. `/Users/me/Desktop/out`) or relative; **relative paths must resolve inside the project root** (a `../` escape returns `400 Invalid dir`). The folder is created via `mkdir -p` if missing. The filename is written as-is (no dedup — caller controls naming). Returns `{ ok, path }`. Used by the frame-set export when the user types a destination path; the FSAA picker path bypasses the server and writes via the browser's File System Access API.
- `POST /api/archive-clip` — Accepts a raw `video/webm` **or** `video/mp4` blob (limit 50 MB). Saves to `assets/archive/library/` with timestamped filename `live_YYYY-MM-DD_HH-mm-ss.<ext>` (extension follows the request Content-Type — webm on Chrome/Firefox, mp4 on Safari). Returns `{ ok, name, path }`. (Historical filenames may still use the `exhibition_` prefix.)
- `GET /api/library` — Lists files in `assets/archive/library/` (videos + images).
- `GET /api/folder/:kind` — Lists a **whitelisted** archive subfolder for the in-app folder-limited pickers: `background` → `assets/archive/background_images/` (images only), `media` → `assets/archive/media/` (videos + images). `kind` is a fixed key (never a path → no traversal); unknown → 404. Returns `{ items: [{ name, path, type }] }` with `path` under `/assets/...` (served by the static mount). Both dirs are `mkdir -p`'d at boot.
- `GET /healthz` — Liveness probe; returns `{ ok: true, gemini: <boolean> }`.
- Server timeouts are widened for large uploads (`keepAliveTimeout` 65 s, `headersTimeout` 70 s, `requestTimeout` 10 min). `SIGINT`/`SIGTERM` trigger a graceful shutdown that drains in-flight requests (`server.close` + `closeIdleConnections`).

### MediaPipe models

- Models load **local-first** from `/assets/models/mediapipe/` with a transparent CDN fallback (`storage.googleapis.com`). Run `npm run fetch-models` (or `scripts/download-mediapipe-models.sh`, also wired to `postinstall`) to populate the directory; the `.task`/`.tflite` files are gitignored.
- Face/hand/pose detectors initialise eagerly at boot (default on). Object, face-detection and gesture detectors are **lazy** — initialised on first toggle-enable.

---

## `/tool` — Independent Mode Pages

A second entry point at **`/tool`** (served by the existing `express.static(public)` → `public/tool/index.html`; no server route needed). It exposes the same three modes as `/`, but **each mode is a fully independent copy** — its own prefixed ids/classes, its own CSS file, and its own self-contained JS module. **Nothing (ids, classes, labels, styles, code) is shared between modes**, so restyling/editing one mode can never affect another. `/tool` is a **frozen old-style snapshot**: it keeps the original 3-tab mode bar and the pre-refactor button styles. The main `/` app has since diverged (mode bar removed, buttons unified) — see "Workspace Modes" — so the generator output no longer matches `/tool`; do **not** rerun it without intent.

### Generated, not hand-maintained

All nine files under `public/tool/` were produced by **`scripts/gen-tool-pages.js`** from the `public/` originals. **They are now a deliberate frozen snapshot** — the `/` originals have since been refactored (mode bar removed, buttons unified), so rerunning the generator would overwrite `/tool` with the new `/` design. Only rerun it if you intend to re-sync `/tool` to current `/`. The generator (for reference):

- Builds an id set (every `id=` in `index.html` + the dynamic `open-archive-link`) and a class set (every `.class` in `styles.css` + JS `classList`/`className` + HTML `class=`). SVG-internal ids (`pointer`, `axes-circle`, `emotion-*`, `Circumplex_diagram`) are never in `index.html`, so they are **never** prefixed (they're read from the circumplex SVG's `contentDocument`).
- Prefixes per mode: `proc-` (Processing/`edit`), `arch-` (Archive), `live-` (Live). Applied context-aware to CSS selectors (`.x`/`#x` only — value keywords like `overflow: hidden` and `[hidden]` attribute selectors are left intact), HTML attributes (`id`/`for`/`aria-controls`/`class`), and JS (`getElementById`, `classList.*`, `className`, `closest`, `querySelector(All)`, and `class=`/`id=` inside template literals).
- Per-module patches: strips the page-global keyboard/fullscreen block (cut at `const fsPlayerCard = …` to EOF) so the shell owns it once; neutralises the loader auto-boot (`setTimeout(endLoader…)`, loader click listener); in `live.js` rewrites `switchMode('archive')` → `window.__toolGoMode('archive')`; appends a bootstrap IIFE.

### Files

| File | Purpose |
|------|---------|
| `public/tool/index.html` | Loader + mode bar (`#tool-bar`) + 3 sibling `<section id="{proc,arch,live}-section" class="tool-section">`. Loads `base.css` + 3 mode CSS files + `shell.js`. |
| `public/tool/shell.js` | Owns the loader (dismiss → boot Live), the mode bar, and page-global shortcuts (font-size, Shift+F / `f` fullscreen). On switch it toggles section `hidden` and **dynamic-imports** the mode's module on first activation. Exposes `window.__toolGoMode(mode)`. Uses distinct `tool-*` class names — no overlap with any mode. |
| `public/tool/base.css` | Shared chrome only: font-face/reset/`:root` globals, the real (unprefixed) loader styles, `.tool-stage`/`.tool-section`/`.tool-bar`. |
| `public/tool/{processing,archive,live}.css` | Full `styles.css` with every selector prefixed for that mode. (Carries redundant but harmless global/dead rules.) |
| `public/tool/{processing,archive,live}.js` | Self-contained copy of `script.js` for that mode, prefixed + patched. |

### Mode activation contract (no inter-module coupling)

The shell only toggles `[hidden]` on each `<section>`. Each module watches **its own** section via a `MutationObserver` on the `hidden` attribute: visible → `activate()` (`workspaceMode = null; switchMode(<its mode>)`, which lazily inits MediaPipe + starts the webcam in Live), hidden → `deactivate()` (`stopWebcam()`). Modules never reference each other; the single cross-mode action ("Open in Archive" in Live) goes through `window.__toolGoMode`. Each module is an ES module, so its top-level `const`s are isolated even though all three coexist in one document.

---

## App State Machine

Variable `appState` in `script.js` drives the entire UI. Two states:

```
loading ──(intro video `ended` / click)──► workspace ──(switchMode('live'))──► Live
```

### State: `loading`

- **What's visible**: Full-screen `#loader-overlay` playing the intro video (`#loader-video` = `/assets/loading/Intro_logo.mp4`, ~5.6 s, `autoplay muted playsinline`). No progress bar.
- **`#workspace-root`**: hidden behind overlay.
- **Transition**: the loader waits for the video to **play to the end** — `loaderVideo`'s `ended` event → `endLoader()` → hide overlay, call `showWorkspace()`. **Safety nets** so it can never hang: an `error` listener (missing/undecodable src) and a `loadedmetadata` fallback `setTimeout(endLoader, duration*1000 + 1000)` (caps the wait at the real video length in case `ended` is dropped); if `#loader-video` is absent entirely it falls back to a 2 s timeout. A **click** on the overlay skips the intro and also requests `document.documentElement.requestFullscreen()` (hides the browser headbar); the `ended`/timeout paths can't (no user gesture).

### State: `workspace`

- **Trigger**: Intro video finishes (or user clicks overlay to skip).
- **Transition** (`showWorkspace()`):
  1. `workspaceRoot.hidden = false` — shows standalone workspace.
  2. `form.classList.remove('hidden')` — shows controls sidebar.
  3. Shows `showAnalyticsBtn`, sets text to "View Analytics".
  4. **Resets analytics panel to closed state**: `outputsPanel.hidden = true`, `submitBtn.style.display = 'none'`, removes `analytics-visible` class.
  5. `appState = 'workspace'`.
  6. Forces entry into Live mode: `workspaceMode = null; switchMode('live')` (the null reset is required because `switchMode` early-returns when `mode === workspaceMode`).
  7. **Arms idle/attract mode** (`resetIdleTimer()`) — done here (post-boot) so the reload boot flow is never pre-empted.

### Idle / attract mode

- After **`IDLE_TIMEOUT_MS`** (60 s default; override via `?idleMs=<n>` query param, clamped ≥1000) with **no keyboard/mouse activity**, `enterIdle()` runs an **attract loop** that repeats `[#loader-video intro → Live appear animation]`: shows `#loader-overlay` + replays the intro video, then on its `ended` hides the overlay and calls **`playLiveAppear()`** (re-arms `initialLiveFooterFadePending` and replays the footer+video appear sequence), holds `IDLE_LIVE_HOLD_MS` (~6 s), then loops. A `cycle` token (`idleCycle`) cancels stale async steps.
- The loop runs **continuously** until input — it does NOT wait the timeout between cycles. `mousemove` is filtered by real cursor movement (`isRealMouseMove`, >3px) so the loop's own layout-shift mousemoves (spurious, same coords, fired under a stationary cursor) don't make it exit itself.
- **Any** keyboard/mouse input (`mousemove`[real]/`keydown`/`pointerdown`/`wheel`/`touchstart`, capture-phase) → `exitIdle()` → hides the overlay, stops the loop, and **lands on Live home with the appear animation** (`playLiveAppear()`). The waking `keydown`/`pointerdown` is **consumed** (`stopImmediatePropagation` + `preventDefault` + a one-shot capture `click` swallow) so the wake input can't trigger a control. Non-idle activity just throttle-resets the timer.
- **Fullscreen-safe**: idle never calls `requestFullscreen`/`exitFullscreen`, so it stays in whatever fullscreen state it was in. Cursor hidden via `body.idle-cursor-hidden`.
- **No reload impact**: armed only inside `showWorkspace()`; reuses the existing loader overlay + appear funcs without disturbing their one-time boot use.

---

## HTML Structure

Two top-level containers:

```
<body>
  <div id="loader-overlay">           ← visible during loading state
    <video id="loader-video">          ← intro video, plays to end then dismisses
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
2. **Computer vision** (collapsed by default) — toggles for: video background, inverted mode, face/hand/pose landmarks, object detection, hand gestures, face detection, face style (mesh/dots), pose joints, pose trails. Switching **Face landmarks off** fires a transient awareness toast (`showToast()` → `#toast`, top-center, auto-dismiss ~3.5 s, `pointer-events:none`) — "Face landmarks off — mood analysis is unavailable" — since mood/emotion analysis depends on face blendshapes. A **`CLEAR ALL`** button (`#clear-all-btn`, styled like the other CV rows — white uppercase, right-aligned text — at the bottom of the row list) turns **off every CV toggle except the background controls** — it dispatches a real `change` on each currently-checked box (inverted/face/hand/pose/object/gesture/face-detect) so their existing off-side effects run; Video background + the AEMA/Choose sub-list are left untouched.
3. **Nonverbal analysis** (collapsed by default) — contains "View Analytics" and "Behavior Analysis" buttons.

Sidebar starts with `.hidden` class, shown when entering workspace.

### Center: `.center-column` wrapper

Contains two siblings in a 50/50 flex split:
1. **`.players-panel`** (`flex: 1 1 50%`) — Hidden on page load (`hidden` attribute). Shown only after first video selected. Contains canvas + transport bar.
2. **`#analytics-bottom`** (`.analytics-bottom-panel`, `flex: 1 1 50%`) — Hidden by default. Toggled by "View Analytics" button. Independent of players-panel visibility (stays visible even without video). Two tabs: "Emotions AI" (`#tab-data`) and "Behavior Analysis" (`#tab-ai`). When players-panel is hidden, analytics takes full height.

---

## Workspace Modes

The `/` app is a **connected flow** across three `switchMode()` modes (`live`/`edit`/`archive`); there is **no mode bar**. Boot → **Live**. Navigation is a shared **silver bottom footer** (`#app-footer`) on every page, rebuilt per mode by `renderFooter(mode)` (`FOOTER_SPEC`). Footer buttons **fill equal slots** (1/3 each, full height; 1/2 in Archive); the **current page's own button is disabled** (lighter bg, **solid black** text). Buttons have **no border** and the footer uses `gap: 1px`, so the silver footer bg (`#b9b9b9`) shows through as a single divider line between each pair (no doubled-border overlap).
> - **Live**: `GO Live` (disabled) · `ANALISE` · `ARCHIVE`
> - **Analyse-detail (#5)**: `GO Live` · `ANALISE` (active → main) · `ARCHIVE`
> - **Analyse-main (#4)**: `GO Live` · `ANALISE` (disabled) · `ARCHIVE`
> - **Archive**: `GO Live` · `ARCHIVE` (disabled) — **no ANALISE**
>
> Footer actions: `GO Live`→`switchMode('live')`; `ARCHIVE`→`switchMode('archive')`; `ANALISE`→ on Live runs `saveLiveClipAndAnalyse()` (save 15s → open #4 main), on Analyse-detail runs `openAnalyse('main')`. `switchMode` sets `.workspace[data-mode="…"]`; in `edit` it also sets `data-analyse="detail"|"main"` (via `applyAnalyseView()`) — the hook all per-screen layout CSS is scoped under. `renderFooter` reads `analyseView` so ANALISE is active on detail, disabled on main.

> **Buttons**: every button/label-button/tab shares one unified look — black bg, white text, white 1px square outline (no radius), centered, weight 100, hover → `rgba(255,255,255,0.08)`. The active analytics tab inverts (white bg / black text). Defined on the generic `button` + `.upload-btn` rule with per-context layout-only overrides (`styles.css`).
>
> **`/tool` is a frozen old-style snapshot** — it still has the 3-tab mode bar and the old button styles, served from its own `public/tool/*` files. It does **not** track these `/` changes. Do **not** rerun `scripts/gen-tool-pages.js` (it would overwrite `/tool` with the new `/` design).

### Analyse (`data-mode="edit"`, ex-"Processing") — two sub-views via `data-analyse`
The Analyse experience is `edit` mode with a sub-view set by `analyseView` (`'detail'`|`'main'`) and reflected on `.workspace[data-mode="edit"][data-analyse="…"]`. `openAnalyse(view, clip)` loads a clip + switches; `loadClipIntoAnalyse()` stores `analyseClipBlob` (the file input is gone). Entry: **Archive tile → detail**; **Live ANALISE → main**; **detail footer ANALISE → main**.

**Analyse-detail (#5)** — `[data-analyse="detail"]`: left sidebar = the Computer-vision `<details>` (opened/expanded by `applyAnalyseView`; face/hand/pose/object/face-detect/inverted toggles); center = the clip on the canvas player **fit to full width** with CV overlay; bottom = the **transport bar** (play/pause + timecode, re-enabled here; clip is scrubbable **and looped** — `applyAnalyseView` sets `previewEl.loop = true` for both detail and main) followed by the **`Select media` picker** (`.analyse-media-picker` → `#analyse-media-btn`). It opens the **folder-limited picker modal** (`openFolderPicker('media', …)` → `#folder-picker`) — a custom in-app grid of `assets/archive/media/` (`GET /api/folder/media`), since the OS dialog can't be locked to a folder. Picking a tile calls `loadClipIntoAnalyse(item.path)` (loads by URL into the detail player; kind auto-detected via `isImageBlobOrPath`/`loadMediaIntoAnalysePreview`). The picker button is a standalone `.center-column` child (no longer inside `#analyse-controls`), `display:none` by default and shown only on this sub-view. The sidebar `#show-analytics-btn` is hidden (CSS `!important`, beating the inline display set by `showWorkspace`). Presets/circumplex/response are hidden.

**Analyse-main (#4)** — `[data-analyse="main"]`: `.center-column` is a 2-col grid (areas `"video types" / "circ resp"`) — **left** = small looped video (top) + the **circumplex** (`#circumplex-svg`, with labels) below it (**clicking the video player opens Analyse-detail (#5) with the loaded clip** — `#landmark-canvas` click → `openAnalyse('detail')` when `workspaceMode==='edit' && analyseView==='main'`); **right-top (`types`)** = a row of **`TYPE_01`…`TYPE_0N`** buttons, one per `PROMPT_PRESETS` entry — currently 5 (`#analyse-presets .analyse-num`, `flex:1 1 0` so they always fill the row regardless of count; black default, selected→white via `.is-selected`). **TYPE_01** = main-person-only analysis: strict 0-100% naturalness score + a 0-100% score for each Ekman emotion with the dominant highlighted + a concise general nonverbal summary (prompt `TYPE_01_PROMPT`). **TYPE_02** (`format: 'pose-rows'`) = main-person-only body-pose "poem": a single left-aligned column of a random 15-30 rows (5px vertical gap between rows), each row a **single word** naming a body pose or its clichéd emotion, prefixed with "#  " (hash + two spaces) and rendered UPPERCASE at the default result text size. **TYPE_03** (`format: 'facs-spec'`) = main-person-only concise FACS-style face read rendered as a **spec sheet**: one section per face region, header `## <FACE PART> | <CODE>` (CODE = an *illustrative* numeric range like `88-125`, not a real Action Unit), 1-2 short observation lines below — rendered single-column with the face part on the left, code on the right edge, and a thin rule between sections. **TYPE_04** (`format: 'aema-dossier'`) = main-person-only **AEMA personal reading**: a concise, poetic + slightly-tragic, technically-framed read emitted as 2-4 poem lines + three `LABEL | VALUE` rows (`NATURALNESS | 0-100`, `REGISTER | 0-100`, `VERDICT | phrase`); rendered as a dossier — static "AEMA · PERSONAL READING" title, italic stanza, a Naturalness `N / 100` row, a **REGISTER row drawn as a marker on the INSTAGRAMISH↔NICHE axis** (`REGISTER` = 0-100 left→right position), and a Verdict row. **TYPE_05** (`ask: true`, no `format` → markdown) = **Ask AEMA**: selecting it does NOT auto-run — it reveals an `#analyse-ask` question box (a **4-line `<textarea>`** + `ASK`, red focus outline) inside the type-description card, between the description and the pinned status bar (`updateAskVisibility()` toggles it on `ask` presets only). `ASK` (or ⌘/Ctrl+Enter; plain Enter inserts a newline) wraps the user's typed question in the trusted `TYPE_05_PREPROMPT` framing — the question is fenced as untrusted DATA (injection filter), gated for relevance to AEMA's domain, and given response-format constraints — then runs the normal `runAnalysis()` → markdown. Off-topic / injection / exploit input gets a short **poetic-scare** brush-off. The input + `ASK` are disabled during cooldown. The right-top region then continues, followed by the **type-description box** (`.analyse-typedesc-card` → `#analyse-typedesc` = the selected preset's `description`, "what this TYPE analyses" — the `description` strings are intentionally **general/non-revealing**, they hint at the output without exposing the prompt mechanics) plus the **status bar** (`#status`, moved here from `#view-ai`). That box's bottom aligns with the left video player and scrolls internally (inner `.analyse-typedesc-inner` is `position:absolute` so its content never expands the auto row-1 track); `.analyse-controls` is `align-self:stretch` to fill row-1 height. **Right-bottom (`resp`)** = the **AI response** card (`#view-ai` → `#result-text`, scrollable), beside the circumplex. `#analytics-bottom` uses `display:contents` so `#view-data`(circumplex) and `#view-ai`(result) join the grid. The CV sidebar is hidden in main. **No START, no `?` help, no open-in-fullscreen button** (all removed). Clicking a TYPE sets `selectedPresetIndex`, updates the description via `updateAnalyseTypedesc()`, **and immediately** runs `runSelectedAnalysis()` → `runAnalysis()` (re-POSTs `analyseClipBlob`) → renders `#result-text` — **except TYPE_05**, which reveals the ask box and waits for `ASK` (`askAema()` composes the prompt then calls `runAnalysis()`). **TYPE cooldown** (request-lifecycle driven): initial state has **no TYPE selected** (`selectedPresetIndex = null`, empty `#analyse-typedesc`). The cooldown **begins when a request is actually sent** (`beginTypeCooldown()` inside `runAnalysis`, gated to `analyseView === 'main'`) and releases only when **BOTH** the response has settled **AND** a 10 s minimum has elapsed — lock duration = `max(responseTime, 10 s)`; it never unlocks while a request is in flight (`notifyTypeCooldownResponse` in `runAnalysis`'s `finally` → `maybeReleaseTypeCooldown`, token-guarded against a stale request settling after a newer cooldown began). While locked, `#analyse-presets` gets `.is-cooldown` — buttons `pointer-events:none`, **unselected text 50%-transparent, selected (active) text solid black** — and the click handler early-returns while `typeCooldownActive()`. The `#status` bar shows "Analysing…" in flight → "Cooldown — Ns" until the 10 s mark → "Ready — select a type" on release; the AI result renders into the separate result card. Overlay thickness fixed at 1.0, face-dot density 1.
- **Sharp / hi-DPI overlay**: `#landmark-canvas` backing store auto-sizes per frame inside `updateCanvasDimensions()` to `max(cssWidth × devicePixelRatio, videoNative)`, capped at `MAX_CANVAS_WIDTH = 3840` (4K width). Aspect ratio is locked to the source. `renderScale = canvasWidth / 1280` is recomputed on every resize and feeds every overlay's `lineWidth` / dot `radius` (face mesh, face dots, hand connectors+joints, pose connectors+joints, pose trails, torso fill, object/face-detection boxes+labels), so stroke thickness stays perceptually constant across resolutions. Per-frame video paint uses `imageSmoothingQuality = 'high'`. Same logic runs in Archive and Live.
- **Inverted mode** (`#toggle-inverted-mode`, off by default): when enabled, the canvas gets the CSS class `.inverted-mode` which applies `filter: grayscale(100%) invert(100%)` on the GPU compositor — this gives the negative grayscale of the source at native FPS without per-frame Skia software filtering. The overlay draw functions (`drawFaceLandmarks`, `drawHandLandmarks`, `drawPoseLandmarks`, `drawObjectDetections`, `drawFaceDetections`) already paint in `#FFFFFF`, so the same CSS invert flips them to black for free — no `landmarkCtx.filter` per overlay draw is needed. The class is kept in sync inside `analyzeFaceFrame()` and the toggle's `change` listener. Effect is gated by `workspaceMode === 'edit'` so it never activates in Archive or Live. Tradeoff: pixels read via `getImageData` are pre-CSS-filter (raw color); the CSS filter is applied only at composition for display. **Capture frame / frameset**: `render4KFrame()` writes to an offline canvas that the CSS rule cannot reach, so after all draws complete it mirrors the same gate (`invertedModeEnabled && workspaceMode === 'edit'`) by drawing the finished composite once into a fresh canvas with `outCtx.filter = 'grayscale(100%) invert(100%)'` and returning that. Single post-process pass — identical to the CSS rule which inverts the final composite once. Per-op `ctx.filter` was tried first and produced wrong output because MediaPipe `DrawingUtils` save/restores the context, dropping the filter for landmark passes; the post-process pass sidesteps that entirely.
- **CAPTURE FRAME button** (`#analyse-capture-btn`, sidebar, directly **above** SELECT MEDIA, same `.analyse-capture-select` styling; both pinned to the sidebar bottom as a group). Clicking it opens the **unified capture popup** (`openCapturePopup()` → `#frameset-popup`): a top **"Capture current frame"** button (`#frameset-capture-frame-btn` → `captureCurrentFrame()` → `render4KFrame()` → `POST /api/capture-frame`, default dir `assets/export/frames/`, then closes + toast) plus the **frame-set** range export in `#frameset-set-section` (`startFramesetExport()` → `/api/capture-frameset-frame-v2`). For a still image the set-section is hidden (single capture only). The `#export-overlay` progress UI is unchanged; the FSAA folder-picker from the old capture-frame menu is not re-added (default-dir save only).

### Archive (`data-mode="archive"`)
- **No player.** Full-screen responsive grid (`#library-grid`, CSS `repeat(auto-fill, minmax(260px, 1fr))`) of `.archive-tile`s built by `renderLibrary()` from `GET /api/library` — one tile per **video**. Each tile is a 16:9 thumbnail (`item.thumb` `<img>`, with a first-frame `<video>` fallback) plus a dark translucent bottom bar with white centered text = the clip's creation timestamp (`clipTimestampLabel()` parses `YYYYMMDD_HH-mm-ss`, falls back to the raw name). Empty library → "No clips yet".
- Click a tile → `openAnalyse('detail', item.path)` → the **Analyse-detail (#5)** screen. Navigation is the shared footer (`GO Live` + disabled `ARCHIVE`). (`.controls-panel` + `.center-column` are hidden in archive.)

### Live (`data-mode="live"`) — **boot mode**
- Full-width webcam (`getUserMedia` → `previewEl.srcObject`); MediaPipe overlay via `analyzeFaceFrame()` (face + pose + hand auto-enabled).
- **Defaults reset on Live entry**: the CV/overlay/background state is **global/shared** across modes, so `switchMode`'s live branch calls `resetOverlaySettingsToLiveDefaults()` (before `startWebcam()`) to force defaults — face/hand/pose ON; object/gesture/face-detection/inverted OFF; Video background ON with `backgroundImage` cleared + AEMA/custom unchecked; overlay thickness 1 / dot density 1. This stops Analyse-page adjustments from leaking into Live (notably *Video background off → AEMA bg* drawing over the webcam). Because the state is shared, returning to Analyse afterwards also starts from defaults (`setToggleChecked` fires each toggle's existing change handler only when the value actually changes).
- **Always-warm camera stream**: `ensureWebcamStream()` acquires the camera **once at boot** (during the loader) and keeps the tracks alive for the whole session (idempotent; cached in `webcamStream`, in-flight `getUserMedia` deduped via `webcamWarmupPromise`). `startWebcam()` (Live entry) just attaches that already-live stream — no per-switch `getUserMedia` lag — after `clearPreview()` so no leftover clip plays; the webcam is on screen instantly. `stopWebcam()` (Live exit) **detaches** the stream from the player + stops the rolling-buffer recorder but **does not stop the tracks** (camera stays warm in Archive/Analyse). Tracks are released only on `pagehide`. The rolling-buffer rotation/restart guards key off `liveMode` (not `webcamStream`, which is now always set).
- **Full-bleed video**: in `.workspace[data-mode="live"]` the `#landmark-canvas` is `object-fit: cover` (fills the area between the top indicator and the footer; no letterbox). The old in-player `#transport-bar` is hidden in Live.
- **Top-right indicator** `#live-indicator` (Live only): a red `.live-dot` ● + `LIVE` + the `#live-clock` wall-clock `HH:MM:SS` (`updateLiveClock` on a 1s `liveClockTimer`, started in `switchMode`'s live branch / cleared in `stopWebcam`).
- **Footer**: the shared silver `#app-footer` (see Workspace Modes) — `GO Live` (disabled here) · `ANALISE` · `ARCHIVE`. On **first boot only** (`initialLiveFooterFadePending`), `footer-initial-fade` + `footer-buttons-reveal` run once: buttons stagger in over 3s (1s each, delays 0 / 1 / 2s); when the last button finishes (and webcam is ready), `footer-bg-reveal` + `live-visual-visible` fire together — silver bg fades in over 1s while the canvas opacity rises over 3s; intro classes clear after the 1s bg fade. Later Live re-entries skip the intro and reveal video immediately.
- **Rolling 15-second buffer** (`LIVE_WINDOW_MS = 15000`). `MediaRecorder` records raw webcam (no overlay) in 1s chunks; `rotateCacheRecording()` rotates each window so a finalized previous blob is always available.
- **Analyse click** (`saveLiveClipAndAnalyse(btn)`): the ANALISE button immediately shows **"Saving…"**; grab the last ~15s blob (finalize the recorder), then **switch to Analyse-main right away** (`loadClipIntoAnalyse(finalBlob)` — the analysis runs on this **in-memory** blob, so it isn't gated on the archive). The **archive + thumbnail run in the background** (`archiveLiveClipInBackground`, idle-scheduled via `requestIdleCallback`): `POST /api/archive-clip` (saves `live_YYYYMMDD_HH-mm-ss.<ext>`) → `captureRandomThumbnail(blob)` (the CPU-heavy decode, now off the switch's critical path) + `saveThumbnail(name, png)` (random-frame PNG beside the clip via `/api/capture-frameset-frame-v2?dir=assets/archive/library`) + invalidate `libraryCache`. This removed the click lag/stutter. **No auto-Gemini** — analysis is run later on the Analyse page via a TYPE / footer ANALISE. (Detail→main ANALISE is the plain `openAnalyse('main')` — no save, no "Saving…".)
- **Archive click**: `switchMode('archive')`. `stopWebcam()` detaches the stream + stops recording on mode exit, but the camera tracks stay warm (see "Always-warm camera stream").
- `GET /api/library` pairs each video with its sibling `<base>.png` (the `thumb` field) and omits those images as standalone entries.

## Keyboard Shortcuts

- **`f`** — toggle player fullscreen (when over a video).
- **`Shift+F`** — toggle document fullscreen (hides the browser headbar/chrome). Also entered automatically when the loader is dismissed **by click** (that click is the required user gesture; the intro-video `ended`/timeout auto-dismiss can't request fullscreen). State persists via `localStorage['pageFullscreen']` and re-enters on the next gesture after reload.
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

- Native `<select>` rendered above "Customize prompt" inside `#ai-controls`. Options come from the `PROMPT_PRESETS` array in `script.js` (each entry: `{ id, name, description, prompt, format? }` — `format` is optional and selects a custom response renderer, see AI Response Pipeline; omit for default markdown). Default selection = `DEFAULT_PRESET_ID` (`'full-nonverbal'`).
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
  1. Validate video selected and file size ≤ 250MB.
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

### "Video background" (`#toggle-video-bg`) + background sub-list

- **Checked by default** → `showVideoBackground = true` (source video drawn as canvas background). `showVideoBackground = event.target.checked` (direct mapping, not inverted).
- Unchecked → `showVideoBackground = false`, reveals the background sub-list (`#video-bg-custom-btns`). The single `backgroundImage` slot (null = blank, AEMA logo, or a custom image) is drawn instead of video; null + video-off = black canvas with overlays only ([public/script.js](public/script.js) render branch in `analyzeFaceFrame`). Switching video bg off does **not** auto-enable AEMA — it lands on a blank canvas; the user opts into AEMA or a custom image from the sub-list.
- Sub-list items (order):
  1. **AEMA background** — a **checkbox** (`#toggle-aema-bg`). Checked → `enableCustomBackgroundMode()` + `loadCustomBackgroundFromUrl(AEMA_BACKGROUND_URL, 0.13, true)` (faint contain-fit logo). Unchecked → `clearBackgroundImage()` → blank (no video, no AEMA).
  2. **Choose background** (`#background-image-btn`, a `.toggle`-styled row with a decorative `#toggle-custom-bg` checkbox) — opens the **folder-limited picker modal** (`openFolderPicker('background', …)` → `#folder-picker`, a custom in-app grid of `assets/archive/background_images/` via `GET /api/folder/background`; the OS dialog can't be folder-restricted). Picking a tile loads it by URL (`loadCustomBackgroundFromUrl(item.path)`, stretched) and unchecks AEMA (custom replaces it). Shared modal: ✕ / backdrop / Escape close it.
  3. **Clear background** (`#no-background-btn`) — reset to the blank default: `clearBackgroundImage()` + uncheck AEMA. Video stays off, so the canvas goes black (overlays only).
- The three are alternatives for the one `backgroundImage` slot (mutually exclusive). `clearBackgroundImage()` sets `backgroundImage = null` + `markPreviewDirty()`.
- **Layout** (Analyse-detail CV rows, [public/styles.css](public/styles.css)): the VIDEO BACKGROUND row carries a **↓ (U+2193) on its right edge** (`> .video-bg-control:first-child > .toggle:first-child::after`) marking it as a dropdown, mirroring the checkbox on the left. Sub-list labels are right-aligned (`.video-bg-custom-btns .toggle/.upload-btn` → `justify-content:flex-end`); the AEMA/Choose **checkboxes are shifted right** (`left: 2.5rem` vs the parent's `1rem`) so they read as nested under VIDEO BACKGROUND. Row/button width is unchanged.

---

## Circumplex Diagram

### Workspace Circumplex (`#circumplex-svg` + `#emotion-trail-canvas`, inside `.circumplex-stage`)

- **Visual base**: Custom Illustrator SVG at `assets/circumplex_diagram.svg` (viewBox `0 0 635.77 552.25`), embedded via `<object>` so its DOM is scriptable. Provides quadrants, axes, labels, and the dot pointer.
- **Live pointer**: SVG element `#pointer` is moved each frame via `transform="translate(dx dy)"` — no canvas re-paint, only attribute mutation. **Constant size** (no breath/pulse — removed). It is **clamped to stay fully inside the outer ring**: the (v,a) offset vector is scaled so its length never exceeds `radius − pointerRadius` (circular clamp; fixes the old square mapping where combined v≈1,a≈1 pushed the dot ~1.41× out past the rim).
- **Calibration**: center `(cx,cy)` + `radius` come from the **outer ring** = the masked circle clipped by **`#clippath-3`** (its `<rect>` → `cx=x+w/2`, `cy=y+h/2`, `radius=min(w,h)/2`). Fallback chain: `#clippath-3` rect → axis-label centroids (`emotion-positive/negative/exciting/calming`) → `axes-circle`/`Circumplex_diagram` bbox → viewBox. (Axis labels sit *outside* the ring, so calibrating from them over-sized the radius — hence the ring rect is primary.) `pointerRadius` = half the `#pointer` bbox (≈29).
- **Required SVG IDs**: `pointer` (the dot), `clippath-3` (defines the ring box), and optional `emotion-<label>` groups (`emotion-happy`, …). Mapping: v→+x (right), a→−y (up), so v=1/a=1 map to the ring edge.
- **Bootstrap**: `bootstrapCircumplexSvg()` runs on `<object>` `load` — calibrates from the ring (above), captures the pointer centroid + radius, **enlarges every `[id^="emotion-"]` label ~10%** via a position-preserving `translate(c)·scale(1.1)·translate(-c)` transform (around each label's own bbox centre), re-parents `#pointer` to the SVG root, and sets `svgState.ready = true`. (`getDominantEmotion`/`EMOTIONS` are built here but are **dead code** — defined, never called.)
- **Data source**: FaceLandmarker blendshapes → `computeEmotionCoordinates()` → `updateEmotionWheel()` → `renderEmotionWheel()` sets targets. **(valence, arousal)** = `peak(POSITIVE) − peak(NEGATIVE)` per axis × `EMOTION_GAIN` (1.4), clamped to ±1. Tuned for sad/fear: `cheekSquint` is **excluded** from positive valence (it fires when squeezing eyes shut and was pinning the pointer at centre); `browInnerUp` + `mouthStretch` drive **negative valence** (AU1 sad/fear marker + fear grimace). **`browInnerUp` is valence-only — NOT an arousal cue** (it's ambiguous: high in fear/surprise, low in sadness), so arousal is set by **eye/jaw/mouth state**: high = `eyeWide`/`jawOpen`/`browOuterUp`/`mouthStretch` + **`mouthSmile`** (a smile is activated-positive → upper-right HAPPY); low = full eye-closing (`eyeBlink`/`mouthClose`) + downturned mouth (`mouthFrown`, low-energy sadness). **`eyeSquint` is NOT a low-arousal cue** — it doubles as the Duchenne-smile (cheek-raise) marker, and counting it as low was dragging happy faces down into "dissapointed". So inner-brows-up + corners-down (no wide eyes) → lower-left (sad); wide eyes + open jaw → upper-left (fear); a smile → upper-right (happy). **Calm-sink**: a quiet/relaxed face (overall activation `max(|rawValence|, hiArousal, loArousal)` below `CALM_QUIET` = 0.3) sinks arousal toward CALMING (bottom) by up to `CALM_SINK` (0.7), ramping to 0 at the threshold — so a calm face reads as low-arousal (bottom-centre) instead of dead-centre neutral, while any recognized expression (activation ≥ 0.3) is left untouched. Known limit: `browInnerUp` also fires in surprise, so pure surprise reads slightly toward fear.
- **Animation**: `animateEmotionWheel()` runs via `requestAnimationFrame`. Lerps `wsValence`/`wsArousal` towards `wsTargetValence`/`wsTargetArousal` (factor `WS_LERP = 0.08`), then applies the circular clamp above.
- **Visible in**: `workspace` state, inside "Emotions AI" tab of analytics panel.
- **Removed in this revision**: legend (`#ekman-legend-workspace`), V/A numeric readout (`#emotion-result-card`), label (`#emotion-wheel-name`), and all canvas-drawn art (background ring, quadrant tints, dashed crosshair/ring, canvas-drawn Ekman markers, canvas-drawn pulsing pointer). Functions `buildWorkspaceLegend` / `updateWorkspaceLegend` removed.

---

## AI Response Pipeline

1. Client sends video + prompt to `POST /api/analyze`. Every TYPE preset's `prompt` has `RESPONSE_STYLE_RULES` appended (no semicolons; capitalize the first letter of each line).
2. Server: `promptInput || DEFAULT_PROMPT` — if client sends empty string, server default used.
3. Gemini returns markdown text.
4. Client: `renderAnalysisResult(text, activePreset)` dispatches on the active preset's `format` field (captured at request-send time so a mid-flight selection change can't repaint with the wrong format):
   - **default** → `formatAnalysisResponse(text)` parses markdown:
     - `---` → `<hr>`
     - `## heading` → `<h4>`, `### heading` → `<h5>`
     - `0. Title` / `1. Title` → `<h4>`, `1.1 Subtitle` → `<h5>`
     - `* bullet` / `- bullet` → `<li>` inside `<ul>`
     - `**bold**` → `<strong>`, `*italic*` → `<em>`, `` `code` `` → `<code>`
     - Everything else → `<p>`
   - **`format: 'pose-rows'`** (TYPE_02) → `formatPoseRows(text)`: each non-empty line → `<div class="pose-row">` (verbatim, HTML-escaped), and `#result-text` gets the `.pose-output` class. CSS renders the rows as one left-aligned UPPERCASE flex column at the default result text size with a 5px row gap (`white-space: pre` keeps the literal "#  " prefix).
   - **`format: 'facs-spec'`** (TYPE_03) → `formatFacsSpec(text)`: groups lines into sections (a `## …` line opens one; header split on ` | ` into face part + code, with a tolerant fallback that peels a trailing numeric range when no pipe is present), each emitted as `<div class="facs-section">` (head row `.facs-part`/`.facs-code` + `.facs-body`), and `#result-text` gets the `.facs-output` class. CSS renders the spec-sheet column (part left / code right, thin rule between sections).
   - **`format: 'aema-dossier'`** (TYPE_04) → `formatAemaDossier(text)`: lines before the first `LABEL | VALUE` row → poem stanza; the three rows (`NATURALNESS`/`REGISTER`/`VERDICT`) → readout rows, with `REGISTER` (clamped 0-100) rendered as a `.aema-marker` positioned `left:N%` on the INSTAGRAMISH↔NICHE `.aema-track`. The renderer adds the static title; `#result-text` gets the `.aema-output` class.
   - All custom-format classes (`.pose-output`, `.facs-output`, `.aema-output`) are cleared on every render — including the default markdown branch and the error/`geminiResponse` branch — so styling from one format never leaks into the next response.
5. Result displayed in `#result-text` (font-size `calc(0.82rem + 2pt)` — 2pt larger than the shared base; the custom formats inherit it, the fullscreen overlay stays at base), also copyable to fullscreen overlay.

---

## CSS Conventions

- **`hidden` attribute conflict**: Several elements use `display: flex` in CSS which overrides `[hidden]`. Each needs an explicit `[hidden] { display: none }` rule. Already done for: `.outputs-panel`, `.workspace-root`, `.fullscreen-overlay`, `.panel-view`, `.loader-overlay`.
- **Loader overlay**: Fixed position at `z-index: 200`, covers viewport with loading video. Hidden via `[hidden]` after video ends.
- **Workspace visibility**: Toggled via `hidden` attribute on `#workspace-root` (a fixed-position overlay at `z-index: 100`).
