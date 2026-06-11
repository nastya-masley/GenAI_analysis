# AEMA

Simple Express + vanilla JS app that lets you upload a short video, send it to AI with a customizable prompt, and view the generated response.

## Active Branches
this section is to track all feature branches, it's base branches and feature description.

- `feature/analytics-page-refactor` — base: `feature/split-screen-webcam-flow` — Strip bust/webcam/Truth-Lie/FFmpeg, add loading video screen, workspace with CV + AI analysis
- `feature/ver3` — base: `main` — Production-level performance & reliability pass (MediaPipe hot-path, Gemini File API, server hardening, loader/a11y)
- `feature/ver4` — base: `feature/3-1-perf` — Rename Exhibition → Live, boot directly into Live, logo-only loader, Save & analise 10s (rolling buffer + auto-analyze), custom PNG checkboxes, hidden Cmd/Ctrl +/-/0 font-size shortcut; add standalone `/tool` page (3 fully independent mode copies — own ids/classes/CSS/JS per mode)

## Requirements
- Node.js 18+
- Google Gemini API access + key (used under the hood by AI)

## Setup
1. Install dependencies
   ```bash
   npm install
   ```
2. Copy the env template and add your real key
   ```bash
   cp .env.example .env
   # edit .env and set GEMINI_API_KEY
   ```
3. (Optional) tweak settings via env vars:
   - `GEMINI_MODEL` – override the default `gemini-2.5-flash`
   - `PORT` – change the server port (default 3000)
   - `MAX_VIDEO_SIZE_MB` – cap uploads (default 200 MB)

## Run the app
```bash
npm start
# visit http://localhost:3000
```

The app opens with a loading video screen. Once the video finishes (or is clicked to skip), the workspace appears with video upload, CV controls, and AI analysis.

A second, standalone entry point lives at `http://localhost:3000/tool` — same three modes (Processing / Archive / Live), but each mode is a fully independent copy with its own prefixed ids/classes, its own CSS file, and its own self-contained JS module (no styles/ids/labels shared between modes). The original `/` app is unaffected. The `/tool` files are generated — run `node scripts/gen-tool-pages.js` to regenerate them after editing the originals in `public/`.

## Exhibition / kiosk run (long-running, unattended)

**Easiest:** double-click **`start.command`** in Finder — it does everything below automatically (installs pm2 the first time, (re)starts the supervised server, waits for it, and opens Chrome in kiosk mode). The server keeps running under pm2 even after you close Chrome or the window. The manual equivalent:

**Do NOT use `npm run dev` for the exhibition** — `nodemon` restarts the server on any file touch and gives no crash recovery. Run the server supervised by **pm2**, and the browser tab in kiosk mode.

1. Install pm2 once (global):
   ```bash
   npm i -g pm2
   ```
2. Start the supervised server (auto-restarts on crash; restarts if it crosses an ~800 MB memory cap — see `ecosystem.config.js`):
   ```bash
   npm run start:kiosk      # pm2 start ecosystem.config.js
   npm run logs:kiosk       # tail logs
   npm run stop:kiosk       # stop
   ```
   (Optional) survive a machine reboot: `pm2 startup` then `pm2 save`.
3. Launch Chrome in kiosk/fullscreen pointed at the app:
   ```bash
   google-chrome --kiosk --autoplay-policy=no-user-gesture-required --app=http://localhost:3000
   # macOS: "Google Chrome" via: open -a "Google Chrome" --args --kiosk --app=http://localhost:3000
   ```

**Built-in stability for long runs** (no action needed):
- While idle (no visitor for 35 s) the attract loop runs and the heavy live **recording/mirror encoder is suspended** — it resumes instantly on the next interaction.
- The page **silently reloads itself during idle every ~3 h** (override with `?reloadHours=<n>`) to clear any accumulated browser memory — invisible to visitors.
- Uncaught runtime errors **self-heal** (the tab reloads after repeated fatals) and the boot loader can never hang on a blank screen.

### Workspace features
- **View/Hide Analytics** — toggles the right-side analytics panel with Emotions AI (circumplex diagram, emotion data, blend shapes)
- **Behavior Analysis** — opens the Nonverbal Analysis tab with AI controls, customizable prompt, and results
- **Computer vision** — toggle overlays independently: video background, face/hand/pose landmarks, object detection, hand gestures, face detection

### MediaPipe Integration
- All MediaPipe processing happens locally in the browser and does not influence the AI upload.
- Blend-shape scores and gesture labels stay in sync with the paused/playing video.
- Models load local-first from `assets/models/mediapipe/` with a CDN fallback. Run `npm run fetch-models` to pre-populate them (also runs on `postinstall`).
- Uploaded videos are streamed to the Gemini File API (no inline base64), so server memory stays flat regardless of file size.

## Troubleshooting
- Requests fail immediately → ensure `GEMINI_API_KEY` is present and valid.
- AI errors / timeouts → check server logs for the response payload in the terminal.
- Large files rejected → keep under `MAX_VIDEO_SIZE_MB` or raise the limit (consider API quotas and browser upload time).
