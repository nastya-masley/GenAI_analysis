---
name: holistic-ui-overhaul
overview: Refine the visualization UI (without switching to MediaPipe Holistic) by unifying styles, adding toggles, and layering new analytics like an emotion wheel.
todos: []
---

# Holistic Visualization Upgrade Plan

1. **OCR-X Typography & Controls Layout**

- Import an OCR-X equivalent webfont (e.g., OCR A Extended) near the top of [`public/styles.css`](public/styles.css) and set it as the base font for `:root`, headings, buttons, and overlays.
- Update `public/index.html` control sections to add switches for every visualization feature (face mesh vs. dots, pose extras, detector labels, hand+gesture overlay, emotion wheel). Keep markup minimal but responsive.

2. **Simulated Holistic Data Flow**

- In [`public/script.js`](public/script.js), keep the separate Face/Hand/Pose runners but normalize their results into shared `state.lastFace/Hand/Pose` objects with timestamps so they can be drawn in one pass.
- Centralize overlay clearing/resizing logic so all drawers respect the same canvas updates and toggle checks, giving a unified “Holistic” feel without changing models.

3. **Face Landmarker Modes**

- Refactor `drawFaceLandmarks` to branch on a new `faceDotsOnly` flag: when true, skip connectors and draw white dots (landmarks + irises) only; when false, use the existing mesh style.
- Wire the new UI switch to flip this flag live and display the current mode label; ensure blendshape extraction still runs regardless of mode.

4. **Detector Overlay Restyle**

- Modify `drawFaceDetections` and `drawObjectDetections` so bounding boxes use thicker white strokes and their captions render as larger white rectangles with black text (using the new `.landmark-label` style for consistency).
- Ensure label positions clamp within the canvas and scale with DPI so they stay readable in both portrait/landscape previews.

5. **Hand + Gesture Fusion**

- Rework the hand overlay routine so each detected hand shows joints/segments plus its recognized gesture (category + confidence) rendered adjacent to the wrist using the shared label style.
- Remove the separate gesture text panel from the DOM, replacing it with per-hand annotations and a condensed status line in the outputs card if needed.

6. **Pose Visualization Enhancements**

- Expand `drawPoseLandmarks` to include optional extras: highlighted torso triangle, emphasized joints, and optional motion trails (simple fading polylines) controlled via toggles.
- Use line width/alpha variations to make the skeleton pop against the video, reusing white as the primary color to match the holistic aesthetic.

7. **Emotion Wheel (F.A.C.S.)**

- Implement a new component (canvas or SVG added to `public/index.html`) that renders a black/white Russell circumplex diagram.
- In `public/script.js`, derive approximate valence/arousal scores from available blendshape data (e.g., smile vs. brow tension) and animate a pointer on the wheel; include a switch to show/hide the visualization.

8. **Switch Wiring & Final QA**

- Ensure each toggle updates local state, clears overlays when disabled, and resumes drawing instantly when re-enabled.
- Smoke-test upload + webcam flows to confirm the new UI doesn’t break the existing Gemini analysis pipeline, and verify overlays remain synchronized across pause/play/seek events.