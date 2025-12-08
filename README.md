# AI Analysis

Simple Express + vanilla JS app that lets you upload a short video, send it to AI with a customizable prompt, and view the generated response.

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
   - `COMPRESSION_THRESHOLD_MB` – files above this size are recompressed for Gemini (defaults to `MAX_VIDEO_SIZE_MB`)

## Run the app
```bash
npm start
# visit http://localhost:3000
```

Use the form to upload a file (<=`MAX_VIDEO_SIZE_MB`, 200 MB by default). The app automatically sends a built-in AI prompt; if you need to tweak it, click **Customize prompt** to reveal and edit the full template before submitting. Results appear once AI finishes analysis.

### Optional MediaPipe computer vision
- Open the **Computer vision** panel to toggle each overlay independently: show/hide the video background, face landmarks, hand landmarks, pose landmarks, object detection, hand gestures, and face-detection bounding boxes.
- All MediaPipe processing happens locally in the browser and does not influence the AI upload.
- Blend-shape scores and gesture labels stay in sync with the paused/playing video, and both players auto-adjust to the video's aspect ratio (vertical vs. horizontal layout).
- When an upload exceeds `COMPRESSION_THRESHOLD_MB`, the server transcodes a lightweight 640-wide (or 640-tall for portrait) copy just for Gemini while keeping the original resolution for local playback and MediaPipe analysis. If the recompressed clip is still too large, the server further trims/downsamples it (or rejects it if it still can't hit the limit) before sending it to Gemini.

## Troubleshooting
- Requests fail immediately → ensure `GEMINI_API_KEY` is present and valid.
- AI errors / timeouts → check server logs for the response payload in the terminal.
- Large files rejected → keep under `MAX_VIDEO_SIZE_MB` or raise the limit (consider API quotas and browser upload time).
