# AEMA

Simple Express + vanilla JS app that lets you upload a short video, send it to AI with a customizable prompt, and view the generated response.

## Active Branches
this section is to track all feature branches, it's base branches and feature description.

- `feature/analytics-page-refactor` — base: `feature/split-screen-webcam-flow` — Strip bust/webcam/Truth-Lie/FFmpeg, add loading video screen, workspace with CV + AI analysis
- `feature/ver3` — base: `main` — Production-level performance & reliability pass (MediaPipe hot-path, Gemini File API, server hardening, loader/a11y)
- `feature/ver4` — base: `feature/3-1-perf` — Rename Exhibition → Live, boot directly into Live, logo-only loader, Save & analise 10s (rolling buffer + auto-analyze), custom PNG checkboxes, hidden Cmd/Ctrl +/-/0 font-size shortcut

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
