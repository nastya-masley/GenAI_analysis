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

- `POST /api/analyze` — Multer upload → base64 encode → Gemini API → returns `{ resultText, raw }`.
- If client sends a `prompt` field, it fully replaces `DEFAULT_PROMPT`. Empty prompt = server default.
- `POST /api/capture-frame` — Accepts raw `image/png` (limit 20 MB) with `?filename=frame_<base>_<mm>-<ss>.png`. Saves to `assets/export/frames/`. If the target name already exists, the server appends ` (copy N)` (1-indexed, walking until free) so previous captures are never overwritten. Returns `{ ok, name, path }` where `name` is the actually-saved filename.
- `POST /api/archive-clip` — Accepts raw `video/webm` blob (limit 50 MB). Saves to `assets/archive/library/` with timestamped filename `exhibition_YYYY-MM-DD_HH-mm-ss.webm`. Returns `{ ok, name, path }`.
- `GET /api/library` — Lists files in `assets/archive/library/` (videos + images).

---

## App State Machine

Variable `appState` in `script.js` drives the entire UI. Two states:

```
loading ──(video ends / click)──► workspace
```

### State: `loading`

- **What's visible**: Full-screen black `#loader-overlay` with `<video>` playing `assets/loading/loading.mp4`.
- **`#workspace-root`**: hidden behind overlay.
- **Transition**: video `ended` event OR click on overlay → `endLoader()` → hide overlay, call `showWorkspace()`.
- **Fallback**: if video fails to load/play, skip straight to workspace.

### State: `workspace`

- **Trigger**: Loading video ends or user clicks overlay.
- **Transition** (`showWorkspace()`):
  1. `workspaceRoot.hidden = false` — shows standalone workspace.
  2. `form.classList.remove('hidden')` — shows controls sidebar.
  3. Shows `showAnalyticsBtn`, sets text to "View Analytics".
  4. **Resets analytics panel to closed state**: `outputsPanel.hidden = true`, `submitBtn.style.display = 'none'`, removes `analytics-visible` class.
  5. `appState = 'workspace'`.

---

## HTML Structure

Two top-level containers:

```
<body>
  <div id="loader-overlay">           ← visible during loading state
    <video id="loader-video">          ← plays loading.mp4
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
- Default mode. Shows `#analyze-form` sidebar.
- User uploads video/image, MediaPipe processes it, can send for Gemini analysis.
- **Inverted mode** (`#toggle-inverted-mode`, off by default): when enabled, the canvas gets the CSS class `.inverted-mode` which applies `filter: grayscale(100%) invert(100%)` on the GPU compositor — this gives the negative grayscale of the source at native FPS without per-frame Skia software filtering. The overlay draw functions (`drawFaceLandmarks`, `drawHandLandmarks`, `drawPoseLandmarks`, `drawObjectDetections`, `drawFaceDetections`) already paint in `#FFFFFF`, so the same CSS invert flips them to black for free — no `landmarkCtx.filter` per overlay draw is needed. The class is kept in sync inside `analyzeFaceFrame()` and the toggle's `change` listener. Effect is gated by `workspaceMode === 'edit'` so it never activates in Archive or Exhibition. Tradeoff: pixels read via `getImageData` are pre-CSS-filter (raw color); the CSS filter is applied only at composition for display.

### Archive (`data-mode="archive"`)
- Shows `#library-panel` sidebar with thumbnails from `assets/archive/library/`.
- Click item → loads into shared player with MediaPipe overlay.
- Has "Nonverbal analysis" button for emotion circumplex.

### Exhibition (`data-mode="exhibition"`)
- No sidebar — the live stream takes the full workspace width. Analytics panel closed on entry (`closeAnalyticsPanel`) so player fills full height.
- Starts webcam via `getUserMedia` → streams to `previewEl.srcObject`.
- MediaPipe overlay runs on live feed via existing `analyzeFaceFrame()` loop. Face + pose + hand landmarks auto-enabled on entry.
- `MediaRecorder` records raw webcam (no overlay) continuously in 1s chunks. Chunks accumulate in a single `cacheChunks` array for the current session (from webcam start or since last save).
- **Floating archive overlay** (`#exhibition-overlay`, bottom-left of `.player-card`): pill-shaped **"Archive"** button + inline status pill. Click triggers **stop → flush → save → restart**: stops the recorder (waits for `onstop` so the final cluster is flushed), POSTs the full `cacheChunks` blob to `/api/archive-clip`, then immediately starts a fresh `MediaRecorder` so live recording resumes. Saved clip duration = time since webcam start (or since last save). Sliding-window header+tail approach was removed — it produced files with discontinuous cluster timecodes. Status then shows "Saved! Open in Archive" link that calls `switchMode('archive')`.
- Transport bar shows `LIVE • MM:SS` indicator (`liveMode` flag + `.transport-bar--live` class hides play/pause and timeline). Reverts to normal timeline when leaving exhibition.
- Webcam stops on mode exit via `stopWebcam()` (also clears `liveMode`).

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
