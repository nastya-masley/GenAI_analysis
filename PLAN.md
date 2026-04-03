# PLAN.md

## Completed

- [x] Remove FFmpeg compression from server.js — direct upload → base64 → Gemini
- [x] Remove 3D bust (bust3d.js, Three.js, OBJ loader, split-screen layout)
- [x] Remove webcam state, webcam circumplex diagram, webcam MediaPipe pipeline
- [x] Remove Truth/Lie mode (toggle, TRUE_FALSE_PROMPT, renderVerdictCard, verdict card UI/CSS)
- [x] Add loading video screen (`assets/loading/loading.mp4`) — plays to end or click-to-skip → workspace
- [x] Simplify state machine: `loading → workspace` (2 states)
- [x] Remove ffmpeg-static + fluent-ffmpeg deps, COMPRESSION_THRESHOLD_MB env var
- [x] Update CLAUDE.md, README.md, PLAN.md

## Current State

- App loads with full-screen loading video overlay
- Video end or click → workspace (video upload + CV overlays + AI analysis)
- Single circumplex diagram (workspace only, Emotions AI tab)
- Custom prompt support, no Truth/Lie mode
- No server-side video compression — raw upload to Gemini

## Next Steps

- [ ] Place `loading.mp4` video file in `assets/loading/`
- [ ] Test full flow end-to-end
