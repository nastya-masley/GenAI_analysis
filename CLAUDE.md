# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # Start development server with auto-reload (nodemon)
npm start        # Start production server
```

No build, lint, or test scripts exist. The server runs on port 3000 by default.

## Environment Setup

Copy `.env.example` to `.env` and set:
- `GEMINI_API_KEY` — required for AI analysis
- `GEMINI_MODEL` — defaults to `gemini-2.5-flash-preview-09-2025`
- `MAX_VIDEO_SIZE_MB` — upload size limit (default: 200)
- `PORT` — server port (default: 3000)

## Architecture

**Stack:** Node.js + Express backend, vanilla JS + Three.js + MediaPipe frontend, SQLite database, Google Gemini AI.

### Backend (`server.js` + `db/init.js`)

Two API domains:

1. **Bust Gallery** (`/api/busts`) — CRUD for 3D gallery items stored in SQLite. Images saved to `uploads/busts/`. Database seeded with 120 entries on startup.

2. **Video Analysis** (`POST /api/analyze`) — Accepts video upload via Multer, optionally compresses with FFmpeg (multi-attempt H.264 pipeline at degrading quality/resolution), then sends base64-encoded video to Gemini API with a nonverbal communication analysis prompt. Returns AI text response.

FFmpeg compression triggers at `COMPRESSION_THRESHOLD_MB` with 3 fallback attempts (640px→480px, trimming, increasing CRF).

### Frontend (`public/`)

- **`gallery.js`** — Three.js 3D scene with state machine (GLOBE_OUTSIDE → GLOBE_ENTER → SPHERE_INSIDE → MORPHING → TUNNEL). Bust images rendered as curved plane meshes; scroll-driven navigation; raycaster click detection.

- **`script.js`** — MediaPipe computer vision running entirely in-browser: Face Landmarker (468 points), Hand Landmarker, Pose Landmarker, Gesture Recognizer, Object Detector, Face Detector. Results drawn on canvas overlay. Also handles video upload, AI analysis request, and results display.

- **`analytics.html`** — Separate analytics dashboard page.

### Data Flow

Video upload → Multer (memory buffer) → FFmpeg compression if needed → Gemini API (base64) → analysis text returned to frontend → displayed alongside MediaPipe local CV overlay.

### Key Files

| File | Purpose |
|------|---------|
| `server.js` | Express server, API routes, FFmpeg compression pipeline |
| `db/init.js` | SQLite schema, seeding, bust CRUD functions |
| `public/script.js` | MediaPipe CV, video upload, AI analysis UI |
| `public/gallery.js` | Three.js 3D gallery with state machine |
| `assets/bust-model/` | GLTF 3D model with PBR textures |
