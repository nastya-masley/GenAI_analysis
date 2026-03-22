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

### Backend (`server.js`)

Single API endpoint:

- **Video Analysis** (`POST /api/analyze`) — Accepts video upload via Multer, optionally compresses with FFmpeg (multi-attempt H.264 pipeline at degrading quality/resolution), then sends base64-encoded video to Gemini API with a nonverbal communication analysis prompt. Returns AI text response.

FFmpeg compression triggers at `COMPRESSION_THRESHOLD_MB` with 3 fallback attempts (640px→480px, trimming, increasing CRF).

### Frontend (`public/`)

- **`script.js`** — MediaPipe computer vision running entirely in-browser: Face Landmarker (468 points), Hand Landmarker, Pose Landmarker, Gesture Recognizer, Object Detector, Face Detector. Results drawn on canvas overlay. Also handles video upload, AI analysis request, results display, split-screen webcam pipeline, and emotion circumplex visualization.

- **`bust3d.js`** — Three.js 3D wireframe bust (OBJ model), infinite spin, camera zoom on activation.

- **`analytics.html`** — Separate analytics dashboard page.

### UI Flow

1. Page loads → Full-screen black with 3D wireframe bust centered
2. Click bust → Screen splits 30%/70%, bust animates to left panel, typing text effect plays
3. Webcam face landmarks + circumplex diagram appear in right panel
4. Click bust again → Webcam stops, standalone workspace appears (video upload, CV controls, analytics)
5. Click bust again → Returns to webcam mode

App states cycle: `initial` → `webcam` → `workspace` ↔ `webcam`

### Workspace Layout

- **Left sidebar** (240px fixed): Video upload, Computer Vision toggles (collapsible), Nonverbal analysis controls (View/Hide Analytics, Behavior Analysis buttons)
- **Center**: Input/Output video players (hidden until video selected, native controls)
- **Right panel** (420px, toggled by View/Hide Analytics):
  - **Emotions AI tab**: Circumplex diagram (lerp-smoothed, matching split-screen style with quadrant tints, Ekman markers, trail dots, pulsing pointer), emotion name, Ekman legend, valence/arousal readout, blend shapes list
  - **Nonverbal Analysis tab** (opened by Behavior Analysis button): AI analysis controls, customizable prompt, Truth/Lie mode, verdict card, formatted AI response with fullscreen option

### Truth/Lie Mode

When enabled, replaces the default prompt with a deception analysis prompt. Gemini returns a verdict (LIKELY TRUTHFUL / LIKELY DECEPTIVE / INCONCLUSIVE) with confidence %, deception/truth indicators, and behavioral signals. Results render in a color-coded verdict card (green/red/yellow) above the full analysis.

### AI Response Formatting

Gemini responses use markdown (bold, bullets, `---` separators). The `formatAnalysisResponse` parser converts to HTML: numbered sections → `<h4>`/`<h5>`, `---` → `<hr>`, `**bold**` → `<strong>`, bullets → `<ul>/<li>`.

### Data Flow

Video upload → Multer (memory buffer) → FFmpeg compression if needed → Gemini API (base64) → analysis text returned to frontend → parsed as markdown → displayed alongside MediaPipe local CV overlay.

### Key Files

| File | Purpose |
|------|---------|
| `server.js` | Express server, API routes, FFmpeg compression pipeline |
| `public/script.js` | MediaPipe CV, video upload, AI analysis UI, webcam emotion pipeline, circumplex diagrams |
| `public/bust3d.js` | Three.js 3D wireframe bust with camera animation |
| `public/styles.css` | All styles including split-screen layout, animations, dark theme |
| `public/index.html` | Main page with split-screen + standalone workspace layout |
| `public/analytics.html` | Analytics dashboard page |
