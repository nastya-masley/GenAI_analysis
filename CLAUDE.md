# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Automation Rules for Claude

Whenever you propose or apply changes to the source code in this repository:

1. **Branches**
   - Never commit directly to `main` or `develop`.
   - If the change is not already on a feature branch, create or use a feature branch:
     - Name format: `feature/<short-description>`.
   - Ensure the base branch (e.g. `main`, `develop`) is recorded in `README.md` under "Active Branches".

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
| `server.js` | Express server, single `POST /api/analyze` endpoint, FFmpeg compression pipeline |
| `public/script.js` | All app logic: MediaPipe CV, video upload, AI analysis, webcam pipeline, circumplex diagrams, state machine, button handlers |
| `public/bust3d.js` | Three.js 3D wireframe bust — loads OBJ model, infinite spin, `window.bust3d.activate()`/`deactivate()` for camera z lerp |
| `public/styles.css` | All styles: split-screen layout, flex transitions, dark theme, workspace layout |
| `public/index.html` | HTML structure: `#split-screen` + `#workspace-root` (sibling divs), inline click bridge script |
| `public/analytics.html` | Separate analytics dashboard page (not part of main flow) |

### Backend (`server.js`)

- `POST /api/analyze` — Multer upload → optional FFmpeg compression (3 fallback attempts at degrading quality) → base64 → Gemini API → returns `{ resultText, raw }`.
- If client sends a `prompt` field, it fully replaces `DEFAULT_PROMPT`. Empty prompt = server default.

---

## App State Machine

Variable `appState` in `script.js` drives the entire UI. Three states:

```
initial  ──(bust click)──►  webcam  ──(bust click)──►  workspace
                              ▲                            │
                              └────────(bust click)────────┘
```

### State: `initial`

- **What's visible**: Full-screen black, `#split-screen` centered, 3D bust in `#logo-panel` at full size (`flex: 1`), `#right-panel` at `flex: 0 0 0%` (invisible).
- **`#workspace-root`**: hidden.
- **Bust canvas**: centered, clickable.

### State: `webcam`

- **Trigger**: Bust click from `initial`.
- **Transition sequence** (in `bust-click` handler):
  1. `splitScreen.classList.add('activated')` → CSS flex transition: `#logo-panel` shrinks to `flex: 0 0 30%`, `#right-panel` grows to `flex: 1` (0.8s ease).
  2. `window.bust3d.activate()` → camera z lerps from 5.5 to 7.5 (zoom out).
  3. Wait 900ms for CSS transition to finish.
  4. `typeText()` plays: "Hi..." → "Let me reveal how I see your emotions right now..." → "Look at the camera..." (60ms/char, 1200ms pause between sentences).
  5. `webcamSection.hidden = false` + `classList.add('visible')` → webcam feed + face dots appear with opacity fade-in.
  6. `startWebcam()` → `getUserMedia` → init MediaPipe FaceLandmarker → `analyzeWebcamFrame` loop starts.
  7. After 1s delay: `webcam-diagram.classList.add('diagram-visible')` → circumplex diagram fades in.
  8. After 15s: hint text "To reveal even more click on me..." types out (only if still in `webcam` state).
- **What's visible**: Split-screen 30/70. Left: bust (smaller) + typing text below. Right: webcam video + face landmark dots + circumplex diagram + Ekman legend + emotion name.
- **`#workspace-root`**: hidden.

### State: `workspace`

- **Trigger**: Bust click from `webcam`.
- **Transition** (`showWorkspace()`):
  1. `stopWebcam()` — stops all tracks, sets `webcamRunning = false`.
  2. Hides `webcamSection`, clears typing text.
  3. `splitScreen.hidden = true` — hides entire split-screen.
  4. `workspaceRoot.hidden = false` — shows standalone workspace.
  5. `form.classList.remove('hidden')` — shows controls sidebar.
  6. Shows `showAnalyticsBtn`, sets text to "View Analytics".
  7. **Resets analytics panel to closed state**: `outputsPanel.hidden = true`, `submitBtn.style.display = 'none'`, removes `analytics-visible` class.
  8. `appState = 'workspace'`.
- **Back to webcam** (bust click from `workspace` → `showWebcam()`):
  1. `workspaceRoot.hidden = true`.
  2. `splitScreen.hidden = false`.
  3. `webcamSection` shown with fade-in.
  4. `startWebcam()` re-initializes camera (has race condition guard: if state changed during `getUserMedia` await, stops tracks immediately).

---

## HTML Structure

Two top-level sibling containers (not nested):

```
<body>
  <div id="split-screen">          ← visible in initial + webcam states
    <div id="logo-panel">          ← 3D bust + typing text
    <div id="right-panel">         ← webcam feed + circumplex
  </div>

  <div id="workspace-root" hidden> ← visible in workspace state
    <main class="container">
      <section class="workspace">
        <form id="analyze-form">   ← controls sidebar (left)
        <div class="players-panel"> ← video players (center)
        <div class="outputs-panel"> ← analytics tabs (right)
      </section>
      <div id="fullscreen-overlay"> ← fullscreen result view
    </main>
  </div>
</body>
```

An inline `<script>` outside ES modules dispatches `bust-click` CustomEvent when `#bust-canvas` is clicked. This bridges the non-module click to the module-scoped handler in `script.js`.

---

## Workspace Layout (3-column)

### Left: Controls Sidebar (`#analyze-form`, 240px fixed)

Three collapsible sections using `<details>`/`<summary>`:

1. **Video** — file upload input. Label changes "Select video" → "Change video" after selection. File name hint shown below.
2. **Computer vision** (collapsed by default) — toggles for: video background, face/hand/pose landmarks, object detection, hand gestures, face detection, face style (mesh/dots), pose joints, pose trails.
3. **Nonverbal analysis** (collapsed by default) — contains "View Analytics" and "Behavior Analysis" buttons.

Sidebar starts with `.hidden` class, shown when entering workspace.

### Center: Video Players (`.players-panel`)

- **Hidden on page load** (`hidden` attribute). Shown only after first video selected.
- Two cards: "Input" (`<video id="preview">` with native `controls`) and "Output" (`<canvas id="landmark-canvas">`).
- Placeholders shown when no video loaded.

### Right: Analytics Panel (`.outputs-panel`, 420px)

- **Hidden by default** (`hidden` attribute). Toggled by "View Analytics" button.
- Two tabs: "Emotions AI" (`#tab-data`) and "Nonverbal Analysis" (`#tab-ai`).
- Tab switching via click handlers on `#tab-data` / `#tab-ai`.

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
  - `viewAi.hidden = false`, `viewData.hidden = true` — switch to Nonverbal Analysis tab.
  - `aiControls.hidden = false` — ensure AI controls visible.
- **Visibility rule**: Visible (`display: inline-flex`) ONLY when `outputsPanel` is not hidden. Hidden when panel closes.

### Tab: "Emotions AI" (`#tab-data`)

- Click → `tabData` active, `tabAi` inactive, `viewData` shown, `viewAi` hidden.
- Default active tab when analytics panel opens.

### Tab: "Nonverbal Analysis" (`#tab-ai`)

- **Starts hidden** (`hidden` attribute). Only un-hidden when "Behavior Analysis" is clicked.
- Click → `tabAi` active, `tabData` inactive, `viewAi` shown, `viewData` hidden.
- Re-hidden when analytics panel closes.

### "Customize prompt" (`#toggle-prompt`)

- Toggles `#prompt` textarea visibility (`hidden` attribute).
- Variable `promptVisible` tracks state.

### "Truth/Lie mode" (`#toggle-true-false`)

- Checkbox. Sets `trueFalseEnabled` boolean.
- When enabled and analysis sent: `TRUE_FALSE_PROMPT` replaces any custom prompt (takes priority).
- After response: `renderVerdictCard()` parses verdict/confidence/signals, shows `#verdict-card`.
- When disabled: `#verdict-card` hidden on next analysis.

### "Send for Analysis" (`#send-analysis-btn`)

- Click → `runAnalysis()`:
  1. Validate video selected and file size ≤ 100MB.
  2. Hide `aiControls`, show status "Uploading...".
  3. Build FormData with video + prompt (Truth/Lie overrides custom prompt).
  4. `POST /api/analyze`.
  5. Parse response → `formatAnalysisResponse()` → render in `#result-text`.
  6. If Truth/Lie enabled → `renderVerdictCard()`.
  7. Show status "AI response ready." or error.
  8. Re-enable button in `finally`.

### Fullscreen result (`#fullscreen-result-btn` / `#fullscreen-close-btn`)

- Open: copies `resultText.innerHTML` into `#fullscreen-result-content`, shows `#fullscreen-overlay`.
- Close: hides overlay. Also closes on Escape key.

### Video file input (`#video`)

- On change → `handleVideoSelection()`:
  - If file: show `.players-panel`, load into preview, update label to "Change video", show filename hint.
  - If no file: hide `.players-panel`, clear preview.

### "Don't show video background" (`#toggle-video-bg`)

- **Checked by default** → `showVideoBackground = false` (background hidden).
- Unchecked → `showVideoBackground = true`, shows "Choose background" button.
- Logic is inverted: `showVideoBackground = !event.target.checked`.

---

## Circumplex Diagrams

Two independent circumplex diagrams with identical visual style but separate data sources:

### Webcam Circumplex (`#webcam-emotion-canvas`, 420×420)

- **Data source**: Live webcam FaceLandmarker blendshapes → `computeEmotionCoordinates()` → `pushEmotionFrame()` sets targets.
- **Animation**: `animateWebcamCircumplex()` runs via `requestAnimationFrame`. Lerps `liveValence`/`liveArousal` towards `liveTargetValence`/`liveTargetArousal` (factor `LIVE_LERP = 0.08`).
- **Trail**: Points pushed every 5 frames, max 30 points.
- **Legend**: `#ekman-legend` built by `buildEkmanLegend()`, updated by `updateEkmanLegend()`.
- **Visible in**: `webcam` state only.

### Workspace Circumplex (`#emotion-wheel-canvas`, 420×420)

- **Data source**: Video playback FaceLandmarker blendshapes → `updateEmotionWheel()` → `renderEmotionWheel()` sets targets.
- **Animation**: `animateEmotionWheel()` runs via `requestAnimationFrame`. Lerps `wsValence`/`wsArousal` towards `wsTargetValence`/`wsTargetArousal` (factor `WS_LERP = 0.08`).
- **Trail**: Points pushed every 5 frames, max 30 points.
- **Legend**: `#ekman-legend-workspace` built by `buildWorkspaceLegend()`, updated by `updateWorkspaceLegend()`.
- **Visible in**: `workspace` state, inside "Emotions AI" tab of analytics panel.

### Shared visual style (both diagrams):

- Circular semi-transparent background (`rgba(0,0,0,0.55)`).
- 4 quadrant tints (subtle colored arcs).
- Dashed crosshair lines + dashed outer ring (`rgba(255,255,255,0.12-0.2)`).
- Ekman emotion markers: 7 dots at fixed v/a positions, dominant one highlighted (larger, brighter, glow).
- Fading trail dots (opacity proportional to recency).
- Pulsing pointer: outer ring oscillates size (sin wave at `timestamp/400`), solid white center dot.

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

### Verdict Card (Truth/Lie mode only)

`renderVerdictCard(text)` scans response lines for:
- Verdict: first line containing "LIKELY TRUTHFUL", "LIKELY DECEPTIVE", or "INCONCLUSIVE".
- Confidence: `CONFIDENCE: XX%` pattern.
- Signals: bullets under sections matching "Deception Indicators" or "Truthful Indicators".

Verdict label colored: green (`.verdict-truth`), red (`.verdict-lie`), yellow (`.verdict-inconclusive`).

---

## CSS Conventions

- **`hidden` attribute conflict**: Several elements use `display: flex` in CSS which overrides `[hidden]`. Each needs an explicit `[hidden] { display: none }` rule. Already done for: `.outputs-panel`, `.split-panel`, `.workspace-root`, `.typing-text`, `.webcam-section`, `.fullscreen-overlay`, `.panel-view`.
- **Split-screen animation**: Uses `flex` transitions (0.8s ease), NOT `hidden`/`display` toggling. `#right-panel` goes from `flex: 0 0 0%` to `flex: 1`.
- **Workspace visibility**: Toggled via `hidden` attribute on `#workspace-root` (a fixed-position overlay at `z-index: 100`).
- **Dropdown icons**: `<details>` elements use `::after` pseudo-element on `<summary>`: `+` when closed, `×` when `[open]`.
