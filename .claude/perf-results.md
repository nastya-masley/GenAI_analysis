# AEMA — Production-Level Performance Pass: Before / After

Branch `feature/ver3`. Baseline `4e0b028`; after `f25e04b` (Tiers A–D).
Measured via Playwright MCP + curl against `localhost:3000`, 2026-05-17/18.

## Before / After table

| Metric | BEFORE | AFTER | Notes |
|--------|--------|-------|-------|
| Cold load — DOMContentLoaded | 216 ms | 94 ms | compression + cache + resource hints |
| Cold load — load event | 625 ms | 195 ms | |
| Cold load — First Contentful Paint | 444 ms | 76 ms | varies with OS disk cache; directionally large |
| Root `content-encoding` | none | `gzip` | B1 |
| index.html transfer (encoded/decoded) | — | 3.5 KB / 16.2 KB | gzipped |
| `script.js` transfer | 87.2 KB raw | 21.8 KB gzip (91.6 KB raw) | 76% smaller on the wire |
| `styles.css` transfer | 32.6 KB raw | 5.7 KB gzip | 82% smaller on the wire |
| Asset `Cache-Control` | `max-age=0` | `max-age=300` (shell) / `604800` (/assets) | B2 |
| Warm-cache revalidation | always 200 | 304 for js/css/svg/models | B2 |
| Server peak RSS — 40 MB analyze | n/a (base64 in RAM) | 85 MB (idle 59 MB → +26 MB) | B3; old path held the whole file + base64 string in RAM |
| MediaPipe models at boot | 6 fetched eagerly | 3 fetched (face/hand/pose), local-first | A4 + A5 |
| Temp files after analyze | n/a | 0 (cleaned in `finally`) | B3 |
| In-flight request on SIGTERM | killed | drains to completion (full 502) | C4 |
| `compression` dependency footprint | — | 84 KB in node_modules | |

## MediaPipe frame rate (rAF proxy, 1080p video, 4K canvas, 3 detectors)

The rAF-based FPS proxy is **too noisy to attribute a clean number** to the
Tier A per-frame savings — three consecutive 7 s samples after the changes
read 21.0 / 21.8 / 33.2 fps; baseline runs read 19–26 fps. The variance
(MediaPipe GPU scheduling + video content complexity at 3840×2160) swamps the
effect.

Tier A is still sound: A1 removes ~120 canvas-property writes/sec, A2 removes
~60 no-op `classList.toggle` calls/sec, A3 stops per-export `DrawingUtils`/canvas
allocation. At 4K with three models the frame cost is dominated by MediaPipe
inference + rasterization, so these savings are real but below measurement
noise on wall-clock FPS. No regression observed.

> The Tier A commit message (`7b0df88`) cites "avg FPS 19.1 → 25.9" from a
> single sample pair — later multi-sampling showed that was measurement noise,
> not a reproducible gain. Treat this table as the corrected record.

## Verified behaviours

- **B3** Gemini File API: upload → poll-to-ACTIVE → `generateContent` runs end
  to end; sanitized `502` on Gemini error (no raw payload leaked). A true `200`
  response was **not** verified — the Gemini key's prepaid credits are depleted
  (HTTP 429). The flow plumbing and memory profile are confirmed.
- **B6** MediaRecorder MIME negotiation + 2.5 Mbps bitrate cap; archive endpoint
  accepts webm/mp4.
- **C1** 5 concurrent identical-name capture-frame writes → 5 distinct files,
  no overwrite (atomic `wx`).
- **C2** `dir=../../../tmp/evil` → `400 Invalid dir`; valid relative dir → `200`.
- **C4** SIGTERM during a 40 MB analyze → request completed with full `502`,
  new connections refused during drain, clean exit.
- **C5** `GET /healthz` → `200 {"ok":true,"gemini":true}`.
- **D1** preconnect/dns-prefetch/preload present in `<head>`.

## Action item for the user

`GEMINI_API_KEY` billing is depleted (HTTP 429 on every `generateContent`).
Top up at https://ai.studio to verify a successful `/api/analyze` 200 response.
