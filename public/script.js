import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const {
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
  ObjectDetector,
  GestureRecognizer,
  FaceDetector,
  FilesetResolver,
  DrawingUtils
} = vision;

const form = document.getElementById('analyze-form');
const statusEl = document.getElementById('status');
const resultSection = document.getElementById('result');
const resultText = document.getElementById('result-text');
const submitBtn = document.getElementById('submit-btn');
const showAnalyticsBtn = document.getElementById('show-analytics-btn');
const outputsPanel = document.getElementById('analytics-bottom');
const previewEl = document.getElementById('preview');
const videoPlaceholder = document.getElementById('video-placeholder');
const canvasPlaceholder = document.getElementById('canvas-placeholder');
const videoInput = document.getElementById('video');
const uploadButtonLabel = document.getElementById('upload-btn-label');
const selectedFileHint = document.getElementById('selected-file-hint');
const togglePromptBtn = document.getElementById('toggle-prompt');
const promptField = document.getElementById('prompt');
const promptPresetSelect = document.getElementById('prompt-preset');
const toggleVideoBg = document.getElementById('toggle-video-bg');
const toggleInvertedMode = document.getElementById('toggle-inverted-mode');
const backgroundImageBtn = document.getElementById('background-image-btn');
const backgroundImageInput = document.getElementById('background-image');
const toggleFace = document.getElementById('toggle-face');
const toggleHand = document.getElementById('toggle-hand');
const togglePose = document.getElementById('toggle-pose');
const toggleObject = document.getElementById('toggle-object');
const toggleGesture = document.getElementById('toggle-gesture');
const toggleFaceDetect = document.getElementById('toggle-face-detect');
const faceStyleSelect = document.getElementById('face-style');
const togglePoseJoints = document.getElementById('toggle-pose-joints');
const togglePoseTrails = document.getElementById('toggle-pose-trails');
const toggleEmotionWheel = document.getElementById('toggle-emotion-wheel');
const playersPanel = document.querySelector('.players-panel');
let landmarkCanvas = document.getElementById('landmark-canvas');
const blendShapeList = document.getElementById('blend-shape-list');
const emotionWheelContainer = document.getElementById('emotion-wheel-container');
const circumplexSvgObject = document.getElementById('circumplex-svg');
const tabData = document.getElementById('tab-data');
const tabAi = document.getElementById('tab-ai');
const viewData = document.getElementById('view-data');
const viewAi = document.getElementById('view-ai');
const captureFrameGroup = document.getElementById('capture-frame-group');
const captureFrameBtn = document.getElementById('capture-frame-btn');
const captureFrameMenuToggle = document.getElementById('capture-frame-menu-toggle');
const captureFrameMenu = document.getElementById('capture-frame-menu');
const captureFrameMenuCurrent = document.getElementById('capture-frame-menu-current');
const captureFrameDestBtn = document.getElementById('capture-frame-dest-btn');
const captureFrameDestClearBtn = document.getElementById('capture-frame-dest-clear-btn');
const captureFramesetBtn = document.getElementById('capture-frameset-btn');
let captureFramePickedDirHandle = null;
let landmarkCtx = landmarkCanvas?.getContext('2d');
if (landmarkCtx) {
  landmarkCtx.imageSmoothingEnabled = true;
  landmarkCtx.imageSmoothingQuality = 'high';
}
const OVERLAY_RENDER_SCALE = 1.5;
let renderScale = OVERLAY_RENDER_SCALE;

const modeBar = document.getElementById('mode-bar');
const libraryPanel = document.getElementById('library-panel');
const libraryGrid = document.getElementById('library-grid');
const archiveAnalyticsBtn = document.getElementById('archive-analytics-btn');
const exhibitionOverlay = document.getElementById('exhibition-overlay');
const exhibitionArchiveBtn = document.getElementById('exhibition-archive-btn');
const exhibitionStatus = document.getElementById('exhibition-status');

let workspaceMode = 'edit';
let libraryCache = null;
let webcamStream = null;
let mediaRecorder = null;
let cacheChunks = [];
let cacheRecorderMime = 'video/webm';
let liveMode = false;

let promptVisible = false;
let previewObjectUrl = null;
let isStaticImage = false;
const imagePreviewEl = document.getElementById('image-preview');

let faceLandmarker;
let handLandmarker;
let poseLandmarker;
let objectDetector;
let gestureRecognizer;
let faceDetector;
let drawingUtils = null;
let runningMode = 'IMAGE';
let lastVideoTime = -1;
let showVideoBackground = toggleVideoBg ? toggleVideoBg.checked : true;
let invertedModeEnabled = toggleInvertedMode ? toggleInvertedMode.checked : false;
let backgroundImage = null;
let faceEnabled = toggleFace ? toggleFace.checked : true;
let handEnabled = toggleHand ? toggleHand.checked : true;
let poseEnabled = togglePose ? togglePose.checked : true;
let objectEnabled = toggleObject ? toggleObject.checked : false;
let gestureEnabled = toggleGesture ? toggleGesture.checked : false;
let faceDetectionEnabled = toggleFaceDetect ? toggleFaceDetect.checked : false;
let faceLoopStarted = false;
let faceRenderMode = faceStyleSelect ? faceStyleSelect.value : 'dots';
let poseJointsEnabled = togglePoseJoints ? togglePoseJoints.checked : true;
let emotionWheelEnabled = toggleEmotionWheel ? toggleEmotionWheel.checked : true;

// Workspace emotion tracking
// Workspace emotion — smoothed via lerp
let wsTargetValence = 0;
let wsTargetArousal = 0;
let wsValence = 0;
let wsArousal = 0;
const WS_LERP = 0.08;

const isPoseTrailsEnabled = () => Boolean(togglePoseTrails?.checked);

const pipelineState = {
  face: null,
  hands: null,
  pose: null,
  objects: null,
  gestures: null,
  faceDetections: null
};
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17]
];
const POSE_CONNECTIONS = PoseLandmarker.POSE_CONNECTIONS || [];
const POSE_VISIBILITY_THRESHOLD = 0.4;
const POSE_TRAIL_INDICES = [15, 16, 27, 28];
const POSE_TRAIL_LENGTH = 12;
const poseTrails = new Map();
const VALENCE_POSITIVE = ['mouthSmileLeft','mouthSmileRight','mouthDimpleLeft','mouthDimpleRight','cheekSquintLeft','cheekSquintRight','cheekPuff'];
const VALENCE_NEGATIVE = ['mouthFrownLeft','mouthFrownRight','browDownLeft','browDownRight','noseSneerLeft','noseSneerRight','mouthPucker'];
const AROUSAL_POSITIVE = ['eyeWideLeft','eyeWideRight','browInnerUp','browOuterUpLeft','browOuterUpRight','jawOpen','eyeSquintLeft','eyeSquintRight'];
const AROUSAL_NEGATIVE = ['eyeBlinkLeft','eyeBlinkRight','mouthClose'];

const clamp = (value, min = -1, max = 1) => Math.min(Math.max(value, min), max);

const getBlendshapeScore = (categories = [], targetName) => {
  const match = categories.find(
    (shape) => shape.categoryName === targetName || shape.displayName === targetName
  );
  return match ? Number(match.score) : 0;
};

const computeEmotionCoordinates = (categories = []) => {
  if (!categories.length) return null;
  const peak = (arr) => arr.length ? Math.max(...arr.map(name => getBlendshapeScore(categories, name))) : 0;

  const valence = clamp(peak(VALENCE_POSITIVE) - peak(VALENCE_NEGATIVE));
  const arousal = clamp(peak(AROUSAL_POSITIVE) - peak(AROUSAL_NEGATIVE));
  return { valence, arousal };
};

// Default Ekman set — extended at SVG bootstrap from named #emotion-* groups.
let EMOTIONS = [
  { label: 'HAPPINESS', v:  0.82, a:  0.20 },
  { label: 'SURPRISE',  v:  0.05, a:  0.85 },
  { label: 'FEAR',      v: -0.55, a:  0.72 },
  { label: 'ANGER',     v: -0.68, a:  0.44 },
  { label: 'DISGUST',   v: -0.72, a:  0.02 },
  { label: 'SADNESS',   v: -0.50, a: -0.60 },
];

const getDominantEmotion = (v, a) => {
  let closest = EMOTIONS[0];
  let minDist = Infinity;
  EMOTIONS.forEach(e => {
    const d = Math.hypot(e.v - v, e.a - a);
    if (d < minDist) { minDist = d; closest = e; }
  });
  return closest;
};

const formatAnalysisResponse = (text) => {
  // Inline markdown: **bold**, *italic*, `code`
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>');

  const lines = text.split('\n');
  let html = '';
  let inList = false;

  for (const raw of lines) {
    const line = raw.trim();

    // Empty line — close list
    if (!line) {
      if (inList) { html += '</ul>'; inList = false; }
      continue;
    }

    // Horizontal rule (---)
    if (/^-{3,}$/.test(line)) {
      if (inList) { html += '</ul>'; inList = false; }
      html += '<hr>';
      continue;
    }

    // Markdown headings: ## or ###
    const headingMatch = line.match(/^(#{1,4})\s+(.+)/);
    if (headingMatch) {
      if (inList) { html += '</ul>'; inList = false; }
      const level = Math.min(headingMatch[1].length + 2, 6); // ## → h4, ### → h5
      html += `<h${level}>${inline(headingMatch[2])}</h${level}>`;
      continue;
    }

    // Numbered section: "0.", "1.", "2." etc (top-level heading)
    if (/^\d+\.\s+/.test(line)) {
      if (inList) { html += '</ul>'; inList = false; }
      const title = line.replace(/^\d+\.\s+/, '');
      html += `<h4>${inline(title)}</h4>`;
      continue;
    }

    // Sub-section: "1.1", "2.3" etc
    if (/^\d+\.\d+\s+/.test(line)) {
      if (inList) { html += '</ul>'; inList = false; }
      const title = line.replace(/^\d+\.\d+\s+/, '');
      html += `<h5>${inline(title)}</h5>`;
      continue;
    }

    // Bullet list: "- " or "* "
    if (/^[-*]\s+/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      const content = line.replace(/^[-*]\s+/, '');
      html += `<li>${inline(content)}</li>`;
      continue;
    }

    // Plain paragraph
    if (inList) { html += '</ul>'; inList = false; }
    html += `<p>${inline(line)}</p>`;
  }

  if (inList) html += '</ul>';
  return html;
};

// No-data state: re-center pointer by lerping the targets back to (0, 0).
const clearEmotionWheel = () => {
  wsTargetValence = 0;
  wsTargetArousal = 0;
};

// Set target values — the animation loop will lerp towards them
const renderEmotionWheel = ({ valence, arousal }) => {
  wsTargetValence = valence;
  wsTargetArousal = arousal;
};

// SVG bootstrap state — populated when circumplex_diagram.svg loads.
const svgState = {
  ready: false,
  doc: null,
  pointer: null,
  pointerOriginX: 0,
  pointerOriginY: 0,
  cx: 0,
  cy: 0,
  radius: 0,
};

function bootstrapCircumplexSvg() {
  if (!circumplexSvgObject) return;
  const doc = circumplexSvgObject.contentDocument;
  if (!doc) return;
  const root = doc.documentElement;
  if (!root) return;

  const pointer = doc.getElementById('pointer');
  if (!pointer) {
    console.warn('[circumplex] SVG missing #pointer. Re-export with the dot layer named "pointer".');
    return;
  }

  const centroid = (el) => {
    const b = el.getBBox();
    return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
  };

  // Calibrate frame from axis-emotion centroids if present (most accurate).
  // Fallback chain: axes-circle → Circumplex_diagram bbox → SVG viewBox.
  const positive = doc.getElementById('emotion-positive');
  const negative = doc.getElementById('emotion-negative');
  const exciting = doc.getElementById('emotion-exciting');
  const calming = doc.getElementById('emotion-calming');
  if (positive && negative && exciting && calming) {
    const p = centroid(positive);
    const n = centroid(negative);
    const e = centroid(exciting);
    const c = centroid(calming);
    svgState.cx = (p.x + n.x) / 2;
    svgState.cy = (e.y + c.y) / 2;
    svgState.radius = ((p.x - n.x) / 2 + (c.y - e.y) / 2) / 2;
  } else {
    const axes = doc.getElementById('axes-circle')
              || doc.getElementById('Circumplex_diagram');
    if (axes) {
      const box = axes.getBBox();
      svgState.cx = box.x + box.width / 2;
      svgState.cy = box.y + box.height / 2;
      svgState.radius = Math.min(box.width, box.height) / 2;
    } else {
      const vb = root.viewBox?.baseVal;
      const w = vb?.width || root.getBBox().width;
      const h = vb?.height || root.getBBox().height;
      svgState.cx = (vb?.x || 0) + w / 2;
      svgState.cy = (vb?.y || 0) + h / 2;
      svgState.radius = Math.min(w, h) / 2;
    }
  }

  // Capture pointer's design-time centroid BEFORE we re-parent it.
  const pCentroid = centroid(pointer);
  svgState.pointerOriginX = pCentroid.x;
  svgState.pointerOriginY = pCentroid.y;

  // Detach pointer from its masked parent (cls-6 has mask-1) and re-parent
  // to the SVG root so it can move freely without being clipped.
  root.appendChild(pointer);

  const extended = [];
  doc.querySelectorAll('[id^="emotion-"]').forEach(node => {
    const label = node.id.replace(/^emotion-/, '').toUpperCase();
    if (!label) return;
    const c = centroid(node);
    const v = (c.x - svgState.cx) / svgState.radius;
    const a = (svgState.cy - c.y) / svgState.radius;
    extended.push({ label, v, a });
  });
  if (extended.length) EMOTIONS = extended;

  svgState.doc = doc;
  svgState.pointer = pointer;
  svgState.ready = true;
}

if (circumplexSvgObject) {
  if (circumplexSvgObject.contentDocument?.readyState === 'complete') {
    bootstrapCircumplexSvg();
  } else {
    circumplexSvgObject.addEventListener('load', bootstrapCircumplexSvg);
  }
}

function drawEmotionWheel(timestamp) {
  // Lerp towards target
  wsValence += (wsTargetValence - wsValence) * WS_LERP;
  wsArousal += (wsTargetArousal - wsArousal) * WS_LERP;

  // Move SVG pointer to match the lerped (v, a).
  if (svgState.ready) {
    const px = svgState.cx + wsValence * svgState.radius;
    const py = svgState.cy - wsArousal * svgState.radius;
    const dx = px - svgState.pointerOriginX;
    const dy = py - svgState.pointerOriginY;
    const pulse = 1 + 0.08 * Math.sin(timestamp / 400);
    svgState.pointer.setAttribute(
      'transform',
      `translate(${dx} ${dy}) translate(${svgState.pointerOriginX} ${svgState.pointerOriginY}) scale(${pulse}) translate(${-svgState.pointerOriginX} ${-svgState.pointerOriginY})`
    );
  }
}

// Continuous animation loop for smooth workspace circumplex
function animateEmotionWheel(timestamp) {
  drawEmotionWheel(timestamp);
  requestAnimationFrame(animateEmotionWheel);
}
requestAnimationFrame(animateEmotionWheel);

const updateEmotionWheel = (blendShapes = []) => {
  if (!emotionWheelEnabled) {
    clearEmotionWheel('Emotion wheel disabled.');
    return;
  }
  const categories = blendShapes[0]?.categories || [];
  if (!categories.length) {
    clearEmotionWheel('Waiting for blendshapes…');
    return;
  }
  const coords = computeEmotionCoordinates(categories);
  if (!coords) {
    clearEmotionWheel('Waiting for blendshapes…');
    return;
  }
  renderEmotionWheel(coords);
};

updateHandGesturePanel(null);
clearEmotionWheel(
  emotionWheelEnabled ? 'Emotion wheel enabled. Waiting for blendshapes…' : 'Emotion wheel disabled.'
);

const DEFAULT_PROMPT = `You are an expert in nonverbal communication, emotion analysis and human behavior.

Analyze this video with focus on EmotionsAI, Face Detection, Posture Detection and give a final summary from a nonverbal communication perspective.

Keep the structure below EXACTLY the same every time. Be detailed in observation but concise in wording.

---

0. Overall picture

* In one sentence describe what is going on video or try to guess *add most likely

1. Emotions Analysis

1.1 Overall Emotional Tone

* Dominant emotions.
* Valence: mainly positive / neutral / negative.

1.2 Emotion Dynamics Over Time

* How emotions change from start -> middle -> end.
* Note any sharp emotional shifts (if present).

1.3 Incongruence

* Any mismatch between likely verbal content/context and nonverbal emotions.
* Brief examples of such mismatch.

---

2. Face Analysis

2.1 Number and Roles of People

* How many visible people.
* Label them as Person 1, Person 2, etc.

2.2 Facial Expressions
For each key person (if few):

* Main emotions via facial expression (smile, jaw tension, frown, raised/lowered brows, eye.).
* Presence of micro-expressions (quick emotional changes, if noticeable).

2.3 Gaze and Focus

* Direct eye contact with camera or other people.
* Frequency and direction of gaze aversion (down, sideways) and possible meaning.

---

3. Posture Analysis
3.1 Posture detailed overview

* Open vs closed posture (arms, torso angle, shoulders).
* Level of body tension (relaxed vs rigid).

3.2 Gestures and Movements

* Use of hand gestures (controlled / natural / excessive).
* Describe hand gesture over the video
* Self-soothing gestures (touching neck, face, hands, etc.).

3.3 Space and Distance

* Distance to others or to camera.
* Leaning forward/backward as signal of engagement or avoidance.

---

4. Summary from a Nonverbal Communication Perspective

4.1 Brief Profile

* 3-5 short bullets on emotional state, confidence level, engagement.

4.2 Key Nonverbal Signals

* 3-7 most important signals (emotion, gaze, gestures, posture) with short explanations.

4.3 Interpretation and Recommendations

* What this nonverbal behavior may indicate (trust, defensiveness, stress, confidence, etc.).

---

Formatting Rules:

* Always use this 0-4 numbered structure and subpoints exactly as shown.
* Use markdown formatting: **bold** for key terms, * for bullet lists.
* Separate each section and subsection with a blank line.
* Use --- between major sections (before 1, 2, 3, 4).
* Each bullet point must be on its own line, starting with "* ".
* Be concise: each bullet max 1-2 short sentences.
* Do NOT invent details. If something cannot be seen or judged, write: "Not enough visual data to assess."`;

const PROMPT_PRESETS = [
  {
    id: 'full-nonverbal',
    name: 'Default',
    prompt: DEFAULT_PROMPT,
  },
  {
    id: 'ekman-naturalness',
    name: 'Ekman + naturalness score',
    prompt: `You are an expert in emotion analysis.

Watch the entire media (video or image) and return:

1. 1 Dominant emotion based on Ekman's 6 basic emotions (happiness, sadness, anger, fear, surprise, disgust)
2. Naturalness — a single 0.00000001%-10% score with step 0.00000000 * 10 each time up to 10 reflecting how natural / authentic the captured behavior looks (0.00000000% = staged, scripted, or AI-generated; 10% = fully natural and spontaneous).

Keep the output structure below EXACTLY the same every time.

---

1. Emotion:
* <Dominant emotion>

2. Naturalness
* Score: N.NNNNNNNN%
* One-sentence concise justification.
---`,
  },
];
const DEFAULT_PRESET_ID = 'full-nonverbal';

const getPresetById = (id) => PROMPT_PRESETS.find((p) => p.id === id) || PROMPT_PRESETS[0];

const populatePresetSelect = () => {
  if (!promptPresetSelect) return;
  promptPresetSelect.innerHTML = '';
  for (const preset of PROMPT_PRESETS) {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.name;
    promptPresetSelect.appendChild(option);
  }
  promptPresetSelect.value = DEFAULT_PRESET_ID;
};

const setStatus = (message, variant = 'info') => {
  statusEl.textContent = message;
  statusEl.dataset.variant = variant;
  statusEl.hidden = false;
};

const setBlendShapesMessage = (message) => {
  if (!blendShapeList) return;
  blendShapeList.innerHTML = '';
};

function updateHandGesturePanel(entries) {
  // Panel removed
}

const resetFaceOutputs = () => {
  if (landmarkCtx && landmarkCanvas) {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    // Fill with black background
    landmarkCtx.fillStyle = '#000000';
    landmarkCtx.fillRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    // Draw background image if available and video background is off
    if (!showVideoBackground && backgroundImage) {
      landmarkCtx.drawImage(backgroundImage, 0, 0, landmarkCanvas.width, landmarkCanvas.height);
    }
  }
  setBlendShapesMessage('Waiting for MediaPipe data…');
  updateHandGesturePanel(null);
  clearEmotionWheel(
    emotionWheelEnabled ? 'Emotion wheel enabled. Waiting for blendshapes…' : 'Emotion wheel disabled.'
  );
};

const previewHasVideo = () => {
  if (isStaticImage) return !!imagePreviewEl?.naturalWidth;
  return previewEl && (previewEl.readyState >= 2 || !!previewEl.srcObject);
};

const markPreviewDirty = () => {
  lastVideoTime = -1;
};

const handlePreviewChange = () => {
  markPreviewDirty();
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
  if (!gestureEnabled || !handEnabled) {
    updateHandGesturePanel(null);
  }
};

const MAX_CANVAS_WIDTH = 3840;

const updateCanvasDimensions = () => {
  if (!landmarkCanvas) return;
  if (isStaticImage && !imagePreviewEl) return;
  if (!isStaticImage && !previewEl) return;

  const srcW = isStaticImage
    ? (imagePreviewEl.naturalWidth || 640)
    : (previewEl.videoWidth || previewEl.clientWidth || 640);
  const srcH = isStaticImage
    ? (imagePreviewEl.naturalHeight || 360)
    : (previewEl.videoHeight || previewEl.clientHeight || 360);

  const aspect = srcW / Math.max(srcH, 1);
  const fitTargetW = aspect >= (3840 / 2160)
    ? 3840
    : Math.round(2160 * aspect);
  const targetW = Math.max(1, fitTargetW);
  const targetH = Math.max(1, Math.round(targetW / aspect));

  if (landmarkCanvas.width !== targetW || landmarkCanvas.height !== targetH) {
    landmarkCanvas.width = targetW;
    landmarkCanvas.height = targetH;
    // Resizing a canvas resets its 2D context state — re-apply smoothing here
    // (once per resize) instead of every frame inside analyzeFaceFrame.
    if (landmarkCtx) {
      landmarkCtx.imageSmoothingEnabled = true;
      landmarkCtx.imageSmoothingQuality = 'high';
    }
  }
};

let prevIsLandscape;
const updatePlayerOrientation = () => {
  if (!playersPanel) return;
  if (!isStaticImage && !previewEl) return;
  const videoWidth = isStaticImage
    ? (imagePreviewEl?.naturalWidth || 640)
    : (previewEl.videoWidth || previewEl.clientWidth);
  const videoHeight = isStaticImage
    ? (imagePreviewEl?.naturalHeight || 360)
    : (previewEl.videoHeight || previewEl.clientHeight);
  if (!videoWidth || !videoHeight) return;
  const isLandscape = videoWidth / Math.max(videoHeight, 1) >= 1;
  // Called every frame — skip the classList writes when orientation is unchanged.
  if (isLandscape === prevIsLandscape) return;
  prevIsLandscape = isLandscape;
  playersPanel.classList.toggle('vertical', isLandscape);
  playersPanel.classList.toggle('horizontal', !isLandscape);
};

previewEl?.addEventListener('loadedmetadata', () => {
  markPreviewDirty();
  updateCanvasDimensions();
  updatePlayerOrientation();
});

previewEl?.addEventListener('pause', handlePreviewChange);
previewEl?.addEventListener('play', handlePreviewChange);
previewEl?.addEventListener('seeked', handlePreviewChange);

// ── Transport bar ──
const transportPlay = document.getElementById('transport-play');
const transportPlayIcon = document.getElementById('transport-play-icon');
const transportPauseIcon = document.getElementById('transport-pause-icon');
const transportTimeline = document.getElementById('transport-timeline');
const transportProgress = document.getElementById('transport-progress');
const transportTime = document.getElementById('transport-time');

const fmtTime = (s) => {
  if (!Number.isFinite(s) || s <= 0) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
};

const updateTransport = () => {
  if (!previewEl) return;
  if (liveMode) {
    if (transportTime) transportTime.textContent = 'LIVE • ' + fmtTime(previewEl.currentTime || 0);
    return;
  }
  const cur = previewEl.currentTime || 0;
  const dur = previewEl.duration || 0;
  if (transportProgress && dur) transportProgress.style.width = (cur / dur * 100) + '%';
  if (transportTime) transportTime.textContent = fmtTime(cur) + ' / ' + fmtTime(dur);
  const paused = previewEl.paused;
  if (transportPlayIcon) transportPlayIcon.hidden = !paused;
  if (transportPauseIcon) transportPauseIcon.hidden = paused;
};

previewEl?.addEventListener('timeupdate', updateTransport);
previewEl?.addEventListener('pause', updateTransport);
previewEl?.addEventListener('play', updateTransport);
previewEl?.addEventListener('loadedmetadata', updateTransport);

transportPlay?.addEventListener('click', () => {
  if (!previewEl) return;
  if (previewEl.paused) previewEl.play(); else previewEl.pause();
});

transportTimeline?.addEventListener('click', (e) => {
  if (!previewEl || !previewEl.duration) return;
  const rect = transportTimeline.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  previewEl.currentTime = pct * previewEl.duration;
});

// Click canvas to toggle play/pause
document.getElementById('landmark-canvas')?.addEventListener('click', () => {
  if (isStaticImage) return;
  if (!previewEl || !previewHasVideo()) return;
  if (previewEl.paused) previewEl.play(); else previewEl.pause();
});

window.addEventListener('resize', () => {
  updateCanvasDimensions();
  updatePlayerOrientation();
});

const drawFaceLandmarks = (result) => {
  if (!landmarkCtx || !drawingUtils || !result) return;
  const faces = result.faceLandmarks || [];
  landmarkCtx.save();
  faces.forEach((landmarks) => {
    if (faceRenderMode === 'dots') {
      const width = landmarkCanvas.width || 1;
      const height = landmarkCanvas.height || 1;
      landmarks.forEach((point) => {
        landmarkCtx.beginPath();
        landmarkCtx.arc(point.x * width, point.y * height, 1.5 * renderScale, 0, Math.PI * 2);
        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fill();
      });
      return;
    }

    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: '#FFFFFF',
      lineWidth: 1 * renderScale
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, {
      color: '#FFFFFF'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, {
      color: '#FFFFFF'
    });
  });
  landmarkCtx.restore();
};

const drawHandLandmarks = (result, gestureResult) => {
  if (!landmarkCtx || !result) {
    updateHandGesturePanel(null);
    return;
  }
  const hands = result.landmarks || [];
  const width = landmarkCanvas.width || 1;
  const height = landmarkCanvas.height || 1;

  landmarkCtx.save();
  landmarkCtx.lineCap = 'round';
  landmarkCtx.lineJoin = 'round';
  const summaries = [];
  hands.forEach((landmarks, handIndex) => {
    HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
      const start = landmarks[startIdx];
      const end = landmarks[endIdx];
      if (!start || !end) return;
      landmarkCtx.beginPath();
      landmarkCtx.moveTo(start.x * width, start.y * height);
      landmarkCtx.lineTo(end.x * width, end.y * height);
      landmarkCtx.strokeStyle = '#FFFFFF';
      landmarkCtx.lineWidth = 4 * renderScale;
      landmarkCtx.stroke();
    });

    landmarks.forEach((point) => {
      landmarkCtx.beginPath();
      landmarkCtx.arc(point.x * width, point.y * height, 4 * renderScale, 0, Math.PI * 2);
      landmarkCtx.fillStyle = '#FFFFFF';
      landmarkCtx.fill();
    });

    const gesture = gestureResult?.gestures?.[handIndex]?.[0];
    const handedness = gestureResult?.handednesses?.[handIndex]?.[0]?.displayName;
    if (gesture) {
      const handLabel = handedness ? `${handedness} hand` : `Hand ${handIndex + 1}`;
      const text = `${gesture.categoryName} ${(gesture.score * 100).toFixed(1)}%`;
      const wrist = landmarks[0];
      
      if (wrist) {
        const fontSize = Math.round(12 * renderScale);
        landmarkCtx.font = fontSize + 'px "OCR A Extended", monospace';
        const labelX = wrist.x * width;
        const labelY = wrist.y * height - 10 * renderScale;
        const textWidth = landmarkCtx.measureText(text).width + 16 * renderScale;
        const labelH = 24 * renderScale;
        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fillRect(labelX - 8 * renderScale, labelY - labelH - 2 * renderScale, textWidth, labelH);
        landmarkCtx.strokeStyle = '#000';
        landmarkCtx.lineWidth = 1 * renderScale;
        landmarkCtx.strokeRect(labelX - 8 * renderScale, labelY - labelH - 2 * renderScale, textWidth, labelH);
        landmarkCtx.fillStyle = '#000';
        landmarkCtx.fillText(text, labelX - 4 * renderScale, labelY - 10 * renderScale);
      }
      summaries.push({ handLabel, gesture: gesture.categoryName, confidence: gesture.score });
    }
  });
  landmarkCtx.restore();
  updateHandGesturePanel(summaries);
};

const hasVisiblePoseLandmarks = (result) => {
  const poses = result?.landmarks || [];
  if (!poses.length) return false;
  return poses.some((landmarks) =>
    landmarks.some((point) => {
      const visibility = typeof point.visibility === 'number' ? point.visibility : null;
      const presence = typeof point.presence === 'number' ? point.presence : null;
      const confidence = visibility ?? presence;
      return typeof confidence === 'number' ? confidence > POSE_VISIBILITY_THRESHOLD : true;
    })
  );
};

const updatePoseTrailHistory = (poses = []) => {
  if (!isPoseTrailsEnabled()) {
    poseTrails.clear();
    return;
  }
  const primaryPose = poses[0];
  if (!primaryPose) return;
  POSE_TRAIL_INDICES.forEach((index) => {
    const point = primaryPose[index];
    if (!point) return;
    const history = poseTrails.get(index) || [];
    history.push({ x: point.x, y: point.y });
    if (history.length > POSE_TRAIL_LENGTH) {
      history.shift();
    }
    poseTrails.set(index, history);
  });
};

const drawPoseTrailsOverlay = (width, height) => {
  if (!isPoseTrailsEnabled() || !poseTrails.size) return;
  landmarkCtx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  landmarkCtx.lineWidth = 2 * renderScale;
  poseTrails.forEach((points) => {
    if (points.length < 2) return;
    landmarkCtx.beginPath();
    points.forEach((point, index) => {
      const x = point.x * width;
      const y = point.y * height;
      if (index === 0) {
        landmarkCtx.moveTo(x, y);
      } else {
        landmarkCtx.lineTo(x, y);
      }
    });
    landmarkCtx.stroke();
  });
};

const drawTorsoOverlay = (landmarks, width, height) => {
  const torsoIndices = [11, 12, 24, 23];
  const torsoPoints = torsoIndices
    .map((index) => landmarks[index])
    .filter((point) => point && typeof point.x === 'number' && typeof point.y === 'number');
  if (torsoPoints.length < 4) return;

  landmarkCtx.beginPath();
  torsoPoints.forEach((point, index) => {
    const x = point.x * width;
    const y = point.y * height;
    if (index === 0) {
      landmarkCtx.moveTo(x, y);
    } else {
      landmarkCtx.lineTo(x, y);
    }
  });
  landmarkCtx.closePath();
  landmarkCtx.fillStyle = 'rgba(255, 255, 255, 0.08)';
  landmarkCtx.fill();
  landmarkCtx.strokeStyle = '#FFFFFF';
  landmarkCtx.lineWidth = 1 * renderScale;
  landmarkCtx.stroke();
};

const drawPoseLandmarks = (result) => {
  if (!landmarkCtx || !drawingUtils || !result || !hasVisiblePoseLandmarks(result)) return;
  const poses = result.landmarks || [];
  const width = landmarkCanvas.width || 1;
  const height = landmarkCanvas.height || 1;
  const trailsEnabled = isPoseTrailsEnabled();
  if (trailsEnabled) {
    updatePoseTrailHistory(poses);
  } else {
    poseTrails.clear();
  }
  landmarkCtx.save();
  poses.forEach((landmarks) => {
    if (trailsEnabled) {
      drawingUtils.drawConnectors(landmarks, POSE_CONNECTIONS, {
        color: '#FFFFFF',
        lineWidth: 3 * renderScale
      });
      drawTorsoOverlay(landmarks, width, height);
    }
    if (poseJointsEnabled) {
      drawingUtils.drawLandmarks(landmarks, {
        color: '#FFFFFF',
        radius: 3 * renderScale
      });
    }
  });
  if (trailsEnabled) {
    drawPoseTrailsOverlay(width, height);
  } else {
    poseTrails.clear();
  }
  landmarkCtx.restore();
};

const drawObjectDetections = (result) => {
  if (!landmarkCtx || !result) return;
  const detections = result.detections || [];
  const frameWidth = previewEl.videoWidth || landmarkCanvas.width || 1;
  const frameHeight = previewEl.videoHeight || landmarkCanvas.height || 1;
  const scaleX = landmarkCanvas.width / frameWidth;
  const scaleY = landmarkCanvas.height / frameHeight;
  landmarkCtx.save();
  detections.forEach((detection) => {
    let { originX, originY, width, height } = detection.boundingBox;
    if (width <= 2 && height <= 2) {
      originX *= frameWidth;
      originY *= frameHeight;
      width *= frameWidth;
      height *= frameHeight;
    }
    originX *= scaleX;
    originY *= scaleY;
    width *= scaleX;
    height *= scaleY;
    landmarkCtx.strokeStyle = '#FFFFFF';
    landmarkCtx.lineWidth = 4 * renderScale;
      landmarkCtx.strokeRect(originX, originY, width, height);
      const label = detection.categories?.[0];
      if (label) {
        const text = `${label.categoryName || 'Object'} ${(label.score * 100).toFixed(1)}%`;
        const fontSize = Math.round(16 * renderScale);
        landmarkCtx.font = fontSize + 'px "OCR A Extended", monospace';
        const textWidth = landmarkCtx.measureText(text).width;
        const labelHeight = 30 * renderScale;
        const padding = 10 * renderScale;
        const boxWidth = textWidth + padding * 2;
        let boxX = originX;
        const boxY = Math.max(originY - labelHeight - 4 * renderScale, 0);

        // Clamp boxX to be within canvas width
        if (boxX + boxWidth > landmarkCanvas.width) {
          boxX = landmarkCanvas.width - boxWidth;
        }
        if (boxX < 0) boxX = 0;

        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fillRect(boxX, boxY, boxWidth, labelHeight);
        landmarkCtx.strokeStyle = '#000';
        landmarkCtx.lineWidth = 2 * renderScale;
        landmarkCtx.strokeRect(boxX, boxY, boxWidth, labelHeight);
        landmarkCtx.fillStyle = '#000';
        landmarkCtx.fillText(text, boxX + padding, boxY + labelHeight - 10 * renderScale);
      }
    });
  landmarkCtx.restore();
};

const drawFaceDetections = (result) => {
  if (!landmarkCtx || !result) return;
  const detections = result.detections || [];
  const frameWidth = previewEl.videoWidth || landmarkCanvas.width || 1;
  const frameHeight = previewEl.videoHeight || landmarkCanvas.height || 1;
  const scaleX = landmarkCanvas.width / frameWidth;
  const scaleY = landmarkCanvas.height / frameHeight;
  landmarkCtx.save();
  detections.forEach((detection) => {
    let { originX, originY, width, height } = detection.boundingBox;
    if (width <= 2 && height <= 2) {
      originX *= frameWidth;
      originY *= frameHeight;
      width *= frameWidth;
      height *= frameHeight;
    }
    originX *= scaleX;
    originY *= scaleY;
    width *= scaleX;
    height *= scaleY;
    landmarkCtx.strokeStyle = '#FFFFFF';
    landmarkCtx.lineWidth = 4 * renderScale;
    landmarkCtx.strokeRect(originX, originY, width, height);
    const label = detection.categories?.[0];
    const text = label
      ? `${label.categoryName || 'Face'} ${(label.score * 100).toFixed(1)}%`
      : 'Face';
    const fontSize = Math.round(16 * renderScale);
    landmarkCtx.font = fontSize + 'px "OCR A Extended", monospace';
    const textWidth = landmarkCtx.measureText(text).width;
    const padding = 10 * renderScale;
    const labelHeight = 30 * renderScale;
    const boxWidth = textWidth + padding * 2;
    let boxX = originX;
    const boxY = Math.max(originY - labelHeight - 4 * renderScale, 0);

    // Clamp boxX to be within canvas width
    if (boxX + boxWidth > landmarkCanvas.width) {
      boxX = landmarkCanvas.width - boxWidth;
    }
    if (boxX < 0) boxX = 0;

    landmarkCtx.fillStyle = '#FFFFFF';
    landmarkCtx.fillRect(boxX, boxY, boxWidth, labelHeight);
    landmarkCtx.strokeStyle = '#000';
    landmarkCtx.lineWidth = 2 * renderScale;
    landmarkCtx.strokeRect(boxX, boxY, boxWidth, labelHeight);
    landmarkCtx.fillStyle = '#000';
    landmarkCtx.fillText(text, boxX + padding, boxY + labelHeight - 10 * renderScale);
  });
  landmarkCtx.restore();
};

const drawBlendShapesList = (blendShapes = []) => {
  if (!blendShapeList) return;
  if (!faceEnabled) {
    setBlendShapesMessage('Face landmarks disabled.');
    return;
  }
  if (!blendShapes.length || !blendShapes[0]?.categories?.length) {
    setBlendShapesMessage('Not enough facial data to assess.');
    return;
  }
  const items = blendShapes[0].categories
    .slice(0, 12)
    .map(
      (shape) => `
        <li class="blend-shapes-item">
          <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
          <span class="blend-shapes-value">${(+shape.score).toFixed(4)}</span>
        </li>
      `
    )
    .join('');
  blendShapeList.innerHTML = items;
};

const analyzeFaceFrame = () => {
  requestAnimationFrame(analyzeFaceFrame);

  if (!landmarkCtx) return;

  if (!previewHasVideo()) {
    if (faceLandmarker || handLandmarker || poseLandmarker || objectDetector) {
      resetFaceOutputs();
    }
    return;
  }

  if (!isStaticImage && (!previewEl.videoWidth || !previewEl.videoHeight)) return;

  updatePlayerOrientation();
  updateCanvasDimensions();

  const mediaSrc = isStaticImage ? imagePreviewEl : previewEl;
  const invertActive = invertedModeEnabled && workspaceMode === 'edit';
  if (landmarkCanvas.classList.contains('inverted-mode') !== invertActive) {
    landmarkCanvas.classList.toggle('inverted-mode', invertActive);
  }

  if (showVideoBackground) {
    landmarkCtx.drawImage(mediaSrc, 0, 0, landmarkCanvas.width, landmarkCanvas.height);
  } else {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    // Fill with black background
    landmarkCtx.fillStyle = '#000000';
    landmarkCtx.fillRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    // Draw background image if available
    if (backgroundImage) {
      landmarkCtx.drawImage(backgroundImage, 0, 0, landmarkCanvas.width, landmarkCanvas.height);
    }
  }

  if (
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    setBlendShapesMessage('All landmarks disabled.');
    updateHandGesturePanel(null);
    clearEmotionWheel('Emotion wheel disabled.');
    return;
  }

  const startTimeMs = performance.now();
  const shouldDetect = isStaticImage
    ? lastVideoTime === -1
    : lastVideoTime !== previewEl.currentTime;
  if (shouldDetect) {
    lastVideoTime = isStaticImage ? 0 : previewEl.currentTime;
    pipelineState.face =
      faceEnabled && faceLandmarker ? faceLandmarker.detectForVideo(mediaSrc, startTimeMs) : null;
    pipelineState.hands =
      handEnabled && handLandmarker ? handLandmarker.detectForVideo(mediaSrc, startTimeMs) : null;
    if (poseEnabled && poseLandmarker) {
      const poseResult = poseLandmarker.detectForVideo(mediaSrc, startTimeMs);
      pipelineState.pose = hasVisiblePoseLandmarks(poseResult) ? poseResult : null;
    } else {
      pipelineState.pose = null;
    }
    pipelineState.objects =
      objectEnabled && objectDetector ? objectDetector.detectForVideo(mediaSrc, startTimeMs) : null;
    pipelineState.gestures =
      gestureEnabled && gestureRecognizer
        ? gestureRecognizer.recognizeForVideo(mediaSrc, startTimeMs)
        : null;
    pipelineState.faceDetections =
      faceDetectionEnabled && faceDetector
        ? faceDetector.detectForVideo(mediaSrc, startTimeMs)
        : null;
  }

  const faceResult = pipelineState.face;
  if (faceEnabled && faceResult) {
    drawFaceLandmarks(faceResult);
    const blendShapes = faceResult.faceBlendshapes || [];
    drawBlendShapesList(blendShapes);
    updateEmotionWheel(blendShapes);
  } else if (faceEnabled) {
    setBlendShapesMessage('Detecting face landmarks…');
    if (emotionWheelEnabled) {
      clearEmotionWheel('Waiting for blendshapes…');
    }
  } else {
    setBlendShapesMessage('Face landmarks disabled.');
    clearEmotionWheel('Emotion wheel disabled.');
  }

  if (handEnabled && pipelineState.hands) {
    drawHandLandmarks(pipelineState.hands, pipelineState.gestures);
  } else {
    updateHandGesturePanel(null);
  }

  if (poseEnabled && pipelineState.pose) {
    drawPoseLandmarks(pipelineState.pose);
  } else if (!poseEnabled) {
    poseTrails.clear();
  }

  if (objectEnabled && pipelineState.objects) {
    drawObjectDetections(pipelineState.objects);
  }

  if (faceDetectionEnabled && pipelineState.faceDetections) {
    drawFaceDetections(pipelineState.faceDetections);
  }
};

// ── MediaPipe model loading: local-first with CDN fallback ──
// Models are served from /assets/models/mediapipe/ when present (see
// scripts/download-mediapipe-models.sh). If a local file is missing the
// loader transparently falls back to the original Google CDN URL.
const MEDIAPIPE_LOCAL_BASE = '/assets/models/mediapipe';
let mediapipeCdnFallbackWarned = false;

// `create(modelAssetPath)` builds and returns the task; we try local then CDN.
const createModelWithFallback = async (localFile, cdnUrl, create) => {
  try {
    return await create(`${MEDIAPIPE_LOCAL_BASE}/${localFile}`);
  } catch (localErr) {
    if (!mediapipeCdnFallbackWarned) {
      console.warn('[MediaPipe] local model unavailable — using CDN fallback', localErr);
      mediapipeCdnFallbackWarned = true;
    }
    return create(cdnUrl);
  }
};

const initFaceLandmarker = async () => {
  if (!landmarkCtx) return;
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    faceLandmarker = await createModelWithFallback(
      'face_landmarker.task',
      'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      (modelAssetPath) => FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
      })
    );
    drawingUtils = new DrawingUtils(landmarkCtx);
    await faceLandmarker.setOptions({ runningMode: 'VIDEO' });
    runningMode = 'VIDEO';
    if (!faceLoopStarted) {
      faceLoopStarted = true;
    }
  } catch (error) {
    console.error('MediaPipe failed to load', error);
    setBlendShapesMessage('MediaPipe unavailable.');
  }
};

initFaceLandmarker();
resetFaceOutputs();

const initHandLandmarker = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    handLandmarker = await createModelWithFallback(
      'hand_landmarker.task',
      'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      (modelAssetPath) => HandLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        runningMode,
        numHands: 2
      })
    );
    await handLandmarker.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Hand Landmarker failed to load', error);
  }
};

initHandLandmarker();

const initPoseLandmarker = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    poseLandmarker = await createModelWithFallback(
      'pose_landmarker_lite.task',
      'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      (modelAssetPath) => PoseLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        runningMode,
        numPoses: 2
      })
    );
    await poseLandmarker.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Pose Landmarker failed to load', error);
  }
};

initPoseLandmarker();

const initObjectDetector = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm'
    );
    objectDetector = await createModelWithFallback(
      'efficientdet_lite0.tflite',
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
      (modelAssetPath) => ObjectDetector.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        runningMode,
        scoreThreshold: 0.5
      })
    );
    await objectDetector.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Object detector failed to load', error);
  }
};

const initFaceDetector = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );
    faceDetector = await createModelWithFallback(
      'blaze_face_short_range.tflite',
      'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
      (modelAssetPath) => FaceDetector.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        runningMode
      })
    );
    await faceDetector.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Face detector failed to load', error);
  }
};

const initGestureRecognizer = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    gestureRecognizer = await createModelWithFallback(
      'gesture_recognizer.task',
      'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
      (modelAssetPath) => GestureRecognizer.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        runningMode
      })
    );
    await gestureRecognizer.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Gesture recognizer failed to load', error);
  }
};

// Opt-in detectors (off by default) initialise lazily on first enable so the
// page doesn't fetch three extra models at boot. Each ensure* runs init once.
let objectDetectorRequested = false;
let faceDetectorRequested = false;
let gestureRecognizerRequested = false;

const ensureObjectDetector = () => {
  if (objectDetectorRequested) return;
  objectDetectorRequested = true;
  initObjectDetector();
};
const ensureFaceDetector = () => {
  if (faceDetectorRequested) return;
  faceDetectorRequested = true;
  initFaceDetector();
};
const ensureGestureRecognizer = () => {
  if (gestureRecognizerRequested) return;
  gestureRecognizerRequested = true;
  initGestureRecognizer();
};

// Cover the case where a checkbox is already checked at load (bfcache restore).
if (objectEnabled) ensureObjectDetector();
if (gestureEnabled) ensureGestureRecognizer();
if (faceDetectionEnabled) ensureFaceDetector();

// Start render loop immediately (independent of model loading)
analyzeFaceFrame();

const updatePromptVisibility = () => {
  if (!promptField || !togglePromptBtn) return;
  if (promptVisible) {
    promptField.hidden = false;
    togglePromptBtn.textContent = 'Hide prompt';
  } else {
    promptField.hidden = true;
    togglePromptBtn.textContent = 'Customize prompt';
  }
};

togglePromptBtn?.addEventListener('click', () => {
  promptVisible = !promptVisible;
  updatePromptVisibility();
  if (promptVisible) {
    promptField.focus();
  }
});

populatePresetSelect();
if (promptField) {
  promptField.value = getPresetById(DEFAULT_PRESET_ID).prompt;
}
promptPresetSelect?.addEventListener('change', (event) => {
  const preset = getPresetById(event.target.value);
  if (promptField) {
    promptField.value = preset.prompt;
  }
});

updatePromptVisibility();

toggleVideoBg?.addEventListener('change', (event) => {
  showVideoBackground = event.target.checked;
  // Show/hide background image button when background is off
  if (backgroundImageBtn) {
    backgroundImageBtn.style.display = event.target.checked ? 'none' : 'inline-flex';
  }
  markPreviewDirty();
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

// Handle background image selection
backgroundImageInput?.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      backgroundImage = img;
      markPreviewDirty();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

// Initialize: hide background image button when background is on
if (toggleVideoBg && backgroundImageBtn) {
  showVideoBackground = toggleVideoBg.checked;
  backgroundImageBtn.style.display = toggleVideoBg.checked ? 'none' : 'inline-flex';
}

toggleInvertedMode?.addEventListener('change', (event) => {
  invertedModeEnabled = Boolean(event.target.checked);
  if (landmarkCanvas) {
    landmarkCanvas.classList.toggle('inverted-mode', invertedModeEnabled && workspaceMode === 'edit');
  }
  markPreviewDirty();
});

toggleFace?.addEventListener('change', (event) => {
  faceEnabled = Boolean(event.target.checked);
  markPreviewDirty();
  if (!faceEnabled) {
    setBlendShapesMessage('Face landmarks disabled.');
    pipelineState.face = null;
    clearEmotionWheel('Emotion wheel disabled.');
  }
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

toggleHand?.addEventListener('change', (event) => {
  handEnabled = Boolean(event.target.checked);
  markPreviewDirty();
  if (!handEnabled) {
    pipelineState.hands = null;
    updateHandGesturePanel(null);
  }
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

togglePose?.addEventListener('change', (event) => {
  poseEnabled = Boolean(event.target.checked);
  markPreviewDirty();
  if (!poseEnabled) {
    pipelineState.pose = null;
    poseTrails.clear();
  }
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

toggleObject?.addEventListener('change', (event) => {
  objectEnabled = Boolean(event.target.checked);
  if (objectEnabled) ensureObjectDetector();
  markPreviewDirty();
  if (!objectEnabled) {
    pipelineState.objects = null;
  }
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

toggleGesture?.addEventListener('change', (event) => {
  gestureEnabled = Boolean(event.target.checked);
  if (gestureEnabled) ensureGestureRecognizer();
  markPreviewDirty();
  if (!gestureEnabled) {
    pipelineState.gestures = null;
    updateHandGesturePanel(null);
  }
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

toggleFaceDetect?.addEventListener('change', (event) => {
  faceDetectionEnabled = Boolean(event.target.checked);
  if (faceDetectionEnabled) ensureFaceDetector();
  markPreviewDirty();
  if (!faceDetectionEnabled) {
    pipelineState.faceDetections = null;
  }
  if (
    !showVideoBackground &&
    !faceEnabled &&
    !handEnabled &&
    !poseEnabled &&
    !objectEnabled &&
    !gestureEnabled &&
    !faceDetectionEnabled
  ) {
    resetFaceOutputs();
  }
});

faceStyleSelect?.addEventListener('change', (event) => {
  faceRenderMode = event.target.value === 'dots' ? 'dots' : 'mesh';
});

togglePoseJoints?.addEventListener('change', (event) => {
  poseJointsEnabled = Boolean(event.target.checked);
});

togglePoseTrails?.addEventListener('change', () => {
  if (!isPoseTrailsEnabled()) {
    poseTrails.clear();
  }
});

toggleEmotionWheel?.addEventListener('change', (event) => {
  emotionWheelEnabled = Boolean(event.target.checked);
  if (emotionWheelContainer) {
    emotionWheelContainer.hidden = !emotionWheelEnabled;
  }
  if (!emotionWheelEnabled) {
    clearEmotionWheel('Emotion wheel disabled.');
  } else {
    clearEmotionWheel('Emotion wheel enabled. Waiting for blendshapes…');
  }
});

if (emotionWheelContainer && toggleEmotionWheel) {
  emotionWheelContainer.hidden = !toggleEmotionWheel.checked;
}

// Ensure outputs panel is hidden by default on page load
if (outputsPanel) {
  outputsPanel.hidden = true;
}
const workspace = document.querySelector('.workspace');
if (workspace) {
  workspace.classList.remove('analytics-visible');
}

const enableFaceLandmarks = () => {
  if (toggleFace && !toggleFace.checked) {
    toggleFace.checked = true;
    toggleFace.dispatchEvent(new Event('change'));
  }
};

const enablePoseLandmarks = () => {
  if (togglePose && !togglePose.checked) {
    togglePose.checked = true;
    togglePose.dispatchEvent(new Event('change'));
  }
};

const enableHandLandmarks = () => {
  if (toggleHand && !toggleHand.checked) {
    toggleHand.checked = true;
    toggleHand.dispatchEvent(new Event('change'));
  }
};

showAnalyticsBtn?.addEventListener('click', toggleAnalyticsPanel);
archiveAnalyticsBtn?.addEventListener('click', toggleAnalyticsPanel);

tabData?.addEventListener('click', () => {
  tabData.classList.add('active');
  tabAi?.classList.remove('active');
  if (viewData) viewData.hidden = false;
  if (viewAi) viewAi.hidden = true;
});

tabAi?.addEventListener('click', () => {
  tabAi.classList.add('active');
  tabData?.classList.remove('active');
  if (viewAi) viewAi.hidden = false;
  if (viewData) viewData.hidden = true;
});

const revokePreviewUrl = () => {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
};

// Tracks the load/play listeners attached by showBlobInPreview so they can be
// detached on the next upload or clearPreview — otherwise stale listeners pile
// up on the shared #preview element across re-uploads.
const previewListeners = {};
const detachPreviewListeners = () => {
  if (!previewEl) return;
  for (const [event, fn] of Object.entries(previewListeners)) {
    if (fn) previewEl.removeEventListener(event, fn);
    previewListeners[event] = null;
  }
};

const showBlobInPreview = (blob, statusMessage) => {
  if (!blob || !previewEl) return;

  // Reset any previous stream, object URLs and stale listeners
  detachPreviewListeners();
  revokePreviewUrl();
  previewEl.srcObject = null;

  // Prepare element for muted/inline autoplay before setting src
  previewEl.muted = true;
  previewEl.playsInline = true;
  previewEl.autoplay = true;
  previewEl.loop = false;

  const applySrc = (src) => {
    previewEl.src = src;
    previewEl.currentTime = 0;
    previewEl.load();
    previewEl.play?.().catch(() => {});
  };

  // Attach diagnostics to help users know what's happening
  const onLoaded = () => {
    setStatus('Press Send for Analysis to analyse non verbal behavior and get AI summary.', 'info');
    previewEl.play?.().catch(() => {});
    updatePlaceholderVisibility();
    previewEl.removeEventListener('loadeddata', onLoaded);
    previewListeners.loadeddata = null;
  };
  const onError = () => {
    // Fallback to FileReader data URL if object URL fails
    const reader = new FileReader();
    reader.onload = () => {
      applySrc(reader.result);
      previewEl.play?.().catch(() => {});
    };
    reader.readAsDataURL(blob);
    previewEl.removeEventListener('error', onError);
    previewListeners.error = null;
  };
  previewListeners.loadeddata = onLoaded;
  previewListeners.error = onError;
  previewEl.addEventListener('loadeddata', onLoaded);
  previewEl.addEventListener('error', onError);

  // Create and set fresh object URL
  previewObjectUrl = URL.createObjectURL(blob);
  applySrc(previewObjectUrl);

  // On metadata ready, attempt playback (helps when initial play() is blocked)
  const tryPlay = () => {
    previewEl.play?.().catch(() => {});
    previewEl.removeEventListener('loadedmetadata', tryPlay);
    previewListeners.loadedmetadata = null;
  };
  previewListeners.loadedmetadata = tryPlay;
  previewEl.addEventListener('loadedmetadata', tryPlay);

  // On canplay, attempt playback again (some browsers need this)
  const tryPlayCanPlay = () => {
    previewEl.play?.().catch(() => {});
    previewEl.removeEventListener('canplay', tryPlayCanPlay);
    previewListeners.canplay = null;
  };
  previewListeners.canplay = tryPlayCanPlay;
  previewEl.addEventListener('canplay', tryPlayCanPlay);

  // Try to start playback immediately; if blocked, the user can press play
  previewEl.play?.().catch(() => {});

  handlePreviewChange();
  updatePlaceholderVisibility();
  if (statusMessage) {
    setStatus(statusMessage, 'info');
  } else {
    setStatus('Loading selected video…', 'info');
  }
};

const updatePlaceholderVisibility = () => {
  if (!videoPlaceholder) return;
  const hasVideo = isStaticImage
    ? !!imagePreviewEl?.src
    : previewEl && (previewEl.src || previewEl.srcObject);
  videoPlaceholder.style.display = hasVideo ? 'none' : 'block';
  
  // Show canvas placeholder when no video is loaded
  if (canvasPlaceholder) {
    canvasPlaceholder.style.display = hasVideo ? 'none' : 'block';
  }
};

const clearPreview = () => {
  if (!previewEl) return;
  previewEl.pause?.();
  previewEl.removeAttribute('src');
  previewEl.srcObject = null;
  detachPreviewListeners();
  revokePreviewUrl();
  if (imagePreviewEl) {
    imagePreviewEl.removeAttribute('src');
    imagePreviewEl.hidden = true;
  }
  isStaticImage = false;
  updatePlaceholderVisibility();
};

const transportBar = document.getElementById('transport-bar');

const showImageInPreview = (file) => {
  clearPreview();
  isStaticImage = true;
  previewObjectUrl = URL.createObjectURL(file);
  imagePreviewEl.src = previewObjectUrl;
  imagePreviewEl.onload = () => {
    updateCanvasDimensions();
    updatePlayerOrientation();
    markPreviewDirty();
    updatePlaceholderVisibility();
    setStatus('Image loaded. Press Send for Analysis to analyse non verbal behavior and get AI summary.', 'info');
  };
  if (transportBar) transportBar.hidden = true;
  handlePreviewChange();
};

const handleVideoSelection = () => {
  const file = videoInput?.files?.[0];
  if (selectedFileHint) {
    if (file) {
      selectedFileHint.hidden = false;
      selectedFileHint.textContent = file.name;
    } else {
      selectedFileHint.hidden = true;
      selectedFileHint.textContent = '';
    }
  }
  if (uploadButtonLabel) {
    uploadButtonLabel.textContent = file ? 'Change file' : 'Select file';
  }
  if (!file) {
    clearPreview();
    handlePreviewChange();
    if (playersPanel) playersPanel.hidden = true;
    if (captureFrameGroup) captureFrameGroup.style.display = 'none';
    if (captureFramesetBtn) captureFramesetBtn.style.display = 'none';
    if (transportBar) transportBar.hidden = false;
    return;
  }

  const isImage = file.type.startsWith('image/');

  if (playersPanel) playersPanel.hidden = false;
  if (captureFrameGroup) captureFrameGroup.style.display = 'inline-flex';
  if (captureFramesetBtn) captureFramesetBtn.style.display = isImage ? 'none' : 'inline-flex';

  if (isImage) {
    showImageInPreview(file);
  } else {
    isStaticImage = false;
    if (transportBar) transportBar.hidden = false;
    showBlobInPreview(file, 'Uploaded clip ready');
  }
};

videoInput?.addEventListener('change', handleVideoSelection);

// Make placeholder clickable to trigger video upload
videoPlaceholder?.addEventListener('click', () => {
  videoInput?.click();
});

// Initialize placeholder visibility
updatePlaceholderVisibility();

// Initialize sidebar as hidden
form.classList.add('hidden');

// ── Loading screen / workspace entry ─────────────────────────────────────────

const workspaceRoot = document.getElementById('workspace-root');

// App states: 'loading' | 'workspace'
let appState = 'loading';

function showWorkspace() {
  if (workspaceRoot) workspaceRoot.hidden = false;
  form.classList.remove('hidden');
  if (showAnalyticsBtn) {
    showAnalyticsBtn.style.display = 'inline-flex';
    showAnalyticsBtn.textContent = 'View Analytics';
  }
  // Reset analytics panel to closed state
  if (outputsPanel) outputsPanel.hidden = true;
  if (submitBtn) submitBtn.style.display = 'none';
  document.querySelector('.workspace')?.classList.remove('analytics-visible');
  appState = 'workspace';
}

// ── Analytics panel helpers ──

function closeAnalyticsPanel() {
  if (outputsPanel) outputsPanel.hidden = true;
  if (submitBtn) submitBtn.style.display = 'none';
  if (tabAi) tabAi.hidden = true;
  if (showAnalyticsBtn) showAnalyticsBtn.textContent = 'View Analytics';
  if (archiveAnalyticsBtn) archiveAnalyticsBtn.textContent = 'Nonverbal analysis';
  document.querySelector('.workspace')?.classList.remove('analytics-visible');
}

function toggleAnalyticsPanel() {
  if (!outputsPanel) return;
  if (outputsPanel.hidden) {
    outputsPanel.hidden = false;
    // Show Behavior Analysis button only in processing mode
    if (submitBtn) submitBtn.style.display = workspaceMode === 'edit' ? 'inline-flex' : 'none';
    enableFaceLandmarks();
    if (tabData) tabData.classList.add('active');
    if (tabAi) { tabAi.classList.remove('active'); tabAi.hidden = true; }
    if (viewData) viewData.hidden = false;
    if (viewAi) viewAi.hidden = true;
    if (showAnalyticsBtn) showAnalyticsBtn.textContent = 'Hide Analytics';
    if (archiveAnalyticsBtn) archiveAnalyticsBtn.textContent = 'Hide analysis';
    document.querySelector('.workspace')?.classList.add('analytics-visible');
  } else {
    closeAnalyticsPanel();
  }
}

// ── Webcam (Exhibition mode) ──

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    clearPreview();
    isStaticImage = false;
    previewEl.srcObject = webcamStream;
    previewEl.muted = true;
    previewEl.playsInline = true;
    previewEl.autoplay = true;
    await previewEl.play().catch(() => {});
    enableFaceLandmarks();
    enablePoseLandmarks();
    enableHandLandmarks();
    markPreviewDirty();
    updatePlaceholderVisibility();
    startCacheRecording(webcamStream);
  } catch (err) {
    console.error('Webcam access denied:', err);
    if (exhibitionStatus) {
      exhibitionStatus.textContent = 'Camera access denied. Please allow camera permissions.';
    }
  }
}

function stopWebcam() {
  stopCacheRecording();
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  if (previewEl) previewEl.srcObject = null;
  liveMode = false;
  transportBar?.classList.remove('transport-bar--live');
}

// Pick the first container/codec the browser actually supports. Hardcoding
// 'video/webm' throws on Safari; mp4 is the fallback there.
const RECORDER_MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4;codecs=h264',
  'video/mp4',
];

function pickRecorderMime() {
  if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) return '';
  return RECORDER_MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || '';
}

function startCacheRecording(stream) {
  stopCacheRecording();
  cacheChunks = [];
  const mime = pickRecorderMime();
  // 2.5 Mbps cap keeps cacheChunks RAM growth bounded (was unbounded).
  const options = { videoBitsPerSecond: 2500000 };
  if (mime) options.mimeType = mime;
  try {
    mediaRecorder = new MediaRecorder(stream, options);
  } catch (e) {
    console.warn('MediaRecorder not supported:', e);
    mediaRecorder = null;
    return;
  }
  cacheRecorderMime = mediaRecorder.mimeType || mime || 'video/webm';
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) cacheChunks.push(e.data);
  };
  mediaRecorder.start(1000);
}

function stopCacheRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    try { mediaRecorder.stop(); } catch (_) {}
  }
  mediaRecorder = null;
}

// ── Mode switching (Processing / Archive / Exhibition) ──

function switchMode(mode) {
  if (mode === workspaceMode) return;

  // Stop webcam when leaving exhibition mode
  if (workspaceMode === 'exhibition') {
    stopWebcam();
  }

  workspaceMode = mode;

  // Close analytics panel + capture menu on mode switch (the latter also
  // detaches the capture menu's document-level click/keydown listeners).
  closeAnalyticsPanel();
  closeCaptureMenu();

  // Hide all sidebar panels + overlays first
  form.classList.add('hidden');
  if (libraryPanel) libraryPanel.hidden = true;
  if (exhibitionOverlay) exhibitionOverlay.hidden = true;
  if (archiveAnalyticsBtn) archiveAnalyticsBtn.style.display = 'none';

  if (mode === 'edit') {
    form.classList.remove('hidden');
  } else if (mode === 'archive') {
    if (libraryPanel) libraryPanel.hidden = false;
    if (archiveAnalyticsBtn) archiveAnalyticsBtn.style.display = 'inline-flex';
    fetchAndRenderLibrary();
  } else if (mode === 'exhibition') {
    if (exhibitionOverlay) exhibitionOverlay.hidden = false;
    if (exhibitionStatus) exhibitionStatus.innerHTML = '';
    // Show player with transport bar in LIVE mode (no timeline / play-pause)
    if (playersPanel) playersPanel.hidden = false;
    liveMode = true;
    if (transportBar) {
      transportBar.hidden = false;
      transportBar.classList.add('transport-bar--live');
    }
    if (transportTime) transportTime.textContent = 'LIVE • 0:00';
    // Hide capture buttons
    if (captureFrameGroup) captureFrameGroup.style.display = 'none';
    if (captureFramesetBtn) captureFramesetBtn.style.display = 'none';
    if (archiveAnalyticsBtn) archiveAnalyticsBtn.style.display = 'none';
    startWebcam();
  }

  // Update toggle buttons
  modeBar?.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
}

modeBar?.addEventListener('click', (e) => {
  const btn = e.target.closest('.mode-btn');
  if (!btn) return;
  switchMode(btn.dataset.mode);
});

async function fetchAndRenderLibrary() {
  if (!libraryGrid) return;
  if (libraryCache) {
    renderLibrary(libraryCache);
    return;
  }
  try {
    const res = await fetch('/api/library');
    libraryCache = await res.json();
    renderLibrary(libraryCache);
  } catch (err) {
    console.error('Failed to fetch library:', err);
    libraryGrid.innerHTML = '<span style="color:rgba(255,255,255,0.5);font-size:0.8rem;">Failed to load library.</span>';
  }
}

function renderLibrary(items) {
  if (!libraryGrid) return;
  libraryGrid.innerHTML = '';
  if (!items.length) {
    libraryGrid.innerHTML = '<span style="color:rgba(255,255,255,0.5);font-size:0.8rem;">No files in library.</span>';
    return;
  }
  items.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'library-item';
    div.dataset.path = item.path;
    div.dataset.type = item.type;

    if (item.type === 'video') {
      const vid = document.createElement('video');
      vid.src = item.path;
      vid.preload = 'metadata';
      vid.muted = true;
      // Show first frame once metadata loaded
      vid.addEventListener('loadeddata', () => { vid.currentTime = 0.01; });
      div.appendChild(vid);
    } else {
      const img = document.createElement('img');
      img.src = item.path;
      div.appendChild(img);
    }

    const name = document.createElement('span');
    name.className = 'library-item-name';
    name.textContent = item.name;
    div.appendChild(name);

    libraryGrid.appendChild(div);
  });
}

libraryGrid?.addEventListener('click', async (e) => {
  const item = e.target.closest('.library-item');
  if (!item) return;
  const itemPath = item.dataset.path;
  const itemType = item.dataset.type;

  try {
    const res = await fetch(itemPath);
    const blob = await res.blob();

    if (playersPanel) playersPanel.hidden = false;
    if (captureFrameGroup) captureFrameGroup.style.display = 'inline-flex';
    if (archiveAnalyticsBtn) archiveAnalyticsBtn.style.display = 'inline-flex';
    enableFaceLandmarks();

    if (itemType === 'image') {
      isStaticImage = true;
      if (captureFramesetBtn) captureFramesetBtn.style.display = 'none';
      showImageInPreview(blob);
    } else {
      isStaticImage = false;
      if (captureFramesetBtn) captureFramesetBtn.style.display = 'inline-flex';
      if (transportBar) transportBar.hidden = false;
      showBlobInPreview(blob, 'Archive clip loaded');
    }
  } catch (err) {
    console.error('Failed to load library item:', err);
  }
});

// ── Exhibition: Archive button — save last 10s ──

exhibitionArchiveBtn?.addEventListener('click', async () => {
  if (!mediaRecorder || !webcamStream) {
    if (exhibitionStatus) exhibitionStatus.textContent = 'No recording active.';
    return;
  }
  if (!cacheChunks.length) {
    if (exhibitionStatus) exhibitionStatus.textContent = 'Wait a moment for the recording to buffer.';
    return;
  }

  exhibitionArchiveBtn.disabled = true;
  if (exhibitionStatus) exhibitionStatus.textContent = 'Saving...';

  const recorder = mediaRecorder;
  const blobType = cacheRecorderMime || 'video/webm';
  const finalBlob = await new Promise((resolve) => {
    recorder.addEventListener('stop', () => {
      resolve(new Blob(cacheChunks, { type: blobType }));
    }, { once: true });
    try { recorder.stop(); } catch (_) { resolve(new Blob(cacheChunks, { type: blobType })); }
  });

  if (webcamStream) startCacheRecording(webcamStream);

  try {
    // Send the actual recorded container type (webm on Chrome, mp4 on Safari).
    const postType = blobType.startsWith('video/mp4') ? 'video/mp4' : 'video/webm';
    const res = await fetch('/api/archive-clip', {
      method: 'POST',
      headers: { 'Content-Type': postType },
      body: finalBlob,
    });
    const data = await res.json();
    if (data.ok) {
      libraryCache = null;
      if (exhibitionStatus) {
        exhibitionStatus.innerHTML = `Saved! <a id="open-archive-link">Open in Archive</a>`;
        document.getElementById('open-archive-link')?.addEventListener('click', () => {
          switchMode('archive');
        });
      }
    } else {
      if (exhibitionStatus) exhibitionStatus.textContent = 'Failed to save clip.';
    }
  } catch (err) {
    console.error('Failed to archive clip:', err);
    if (exhibitionStatus) exhibitionStatus.textContent = 'Error saving clip.';
  } finally {
    exhibitionArchiveBtn.disabled = false;
  }
});

// Loading screen: play video, then enter workspace
const loaderVideo = document.getElementById('loader-video');
const loaderOverlay = document.getElementById('loader-overlay');

function endLoader() {
  if (appState === 'workspace') return;
  if (loaderOverlay) loaderOverlay.hidden = true;
  showWorkspace();
}

const loaderProgressBar = document.getElementById('loader-progress-bar');
let loaderProgress = 0;
let loaderTargetProgress = 0;
let loaderRafId = null;

function animateLoaderProgress() {
  loaderProgress += (loaderTargetProgress - loaderProgress) * 0.08;
  if (loaderProgressBar) loaderProgressBar.style.width = loaderProgress + '%';
  if (Math.abs(loaderTargetProgress - loaderProgress) > 0.1) {
    loaderRafId = requestAnimationFrame(animateLoaderProgress);
  } else {
    if (loaderProgressBar) loaderProgressBar.style.width = loaderTargetProgress + '%';
    loaderRafId = null;
  }
}

if (loaderVideo) {
  loaderVideo.addEventListener('timeupdate', () => {
    if (loaderVideo.duration) {
      loaderTargetProgress = (loaderVideo.currentTime / loaderVideo.duration) * 100;
      if (!loaderRafId) loaderRafId = requestAnimationFrame(animateLoaderProgress);
    }
  });
  loaderVideo.play().catch(() => endLoader());
  loaderVideo.addEventListener('ended', endLoader);
  loaderOverlay?.addEventListener('click', endLoader);
} else {
  endLoader();
}


const refreshCaptureDestLabel = () => {
  if (!captureFrameMenuCurrent) return;
  if (captureFramePickedDirHandle) {
    captureFrameMenuCurrent.textContent = `Saving to: ${captureFramePickedDirHandle.name}`;
    if (captureFrameDestClearBtn) captureFrameDestClearBtn.hidden = false;
    if (captureFrameDestBtn) captureFrameDestBtn.textContent = 'Pick different folder…';
  } else {
    captureFrameMenuCurrent.textContent = 'Saving to: Default';
    if (captureFrameDestClearBtn) captureFrameDestClearBtn.hidden = true;
    if (captureFrameDestBtn) captureFrameDestBtn.textContent = 'Pick folder…';
  }
};

const onDocClickForCaptureMenu = (e) => {
  if (!captureFrameMenu) return;
  if (e.target.closest('#capture-frame-group')) return;
  closeCaptureMenu();
};

const onKeyForCaptureMenu = (e) => {
  if (e.key === 'Escape') closeCaptureMenu();
};

function closeCaptureMenu() {
  if (!captureFrameMenu || captureFrameMenu.hidden) return;
  captureFrameMenu.hidden = true;
  captureFrameMenuToggle?.setAttribute('aria-expanded', 'false');
  document.removeEventListener('click', onDocClickForCaptureMenu, true);
  document.removeEventListener('keydown', onKeyForCaptureMenu);
}

captureFrameMenuToggle?.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!captureFrameMenu) return;
  if (captureFrameMenu.hidden) {
    captureFrameMenu.hidden = false;
    captureFrameMenuToggle.setAttribute('aria-expanded', 'true');
    document.addEventListener('click', onDocClickForCaptureMenu, true);
    document.addEventListener('keydown', onKeyForCaptureMenu);
  } else {
    closeCaptureMenu();
  }
});

captureFrameDestBtn?.addEventListener('click', async () => {
  if (!window.showDirectoryPicker) {
    alert('Folder picker not supported in this browser.');
    return;
  }
  try {
    const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
    captureFramePickedDirHandle = handle;
    refreshCaptureDestLabel();
  } catch (err) {
    if (err?.name !== 'AbortError') console.error('Folder pick failed:', err);
  } finally {
    closeCaptureMenu();
  }
});

captureFrameDestClearBtn?.addEventListener('click', () => {
  captureFramePickedDirHandle = null;
  refreshCaptureDestLabel();
  closeCaptureMenu();
});

captureFrameBtn?.addEventListener('click', () => {
  if (!previewHasVideo() || !landmarkCanvas) return;
  const videoFile = videoInput?.files?.[0];
  const baseName = videoFile ? videoFile.name.replace(/\.[^/.]+$/, '') : 'capture';
  const time = previewEl?.currentTime ?? 0;
  const mm = String(Math.floor(time / 60)).padStart(2, '0');
  const ss = String(Math.floor(time % 60)).padStart(2, '0');
  const filename = `frame_${baseName}_${mm}-${ss}.png`;

  runDetectionsAtCurrentTime();
  const capCanvas = render4KFrame();
  capCanvas.toBlob(async (blob) => {
    if (!blob) return;
    try {
      if (captureFramePickedDirHandle) {
        await writeFrameToDir(captureFramePickedDirHandle, filename, blob);
      } else {
        const res = await fetch(`/api/capture-frame?filename=${encodeURIComponent(filename)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'image/png' },
          body: blob,
        });
        const data = await res.json();
        if (!data.ok) console.error('Capture failed:', data.error);
      }
    } catch (err) {
      console.error('Capture failed:', err);
    }
  }, 'image/png');
});

// ============ FRAME SET EXPORT ============

const framesetPopup = document.getElementById('frameset-popup');
const framesetFrom = document.getElementById('frameset-from');
const framesetTo = document.getElementById('frameset-to');
const framesetDest = document.getElementById('frameset-dest');
const framesetPrefix = document.getElementById('frameset-prefix');
const framesetPickBtn = document.getElementById('frameset-pick-btn');
const framesetWholeBtn = document.getElementById('frameset-whole-btn');
const framesetCancelBtn = document.getElementById('frameset-cancel-btn');
const framesetStartBtn = document.getElementById('frameset-start-btn');
const exportOverlay = document.getElementById('export-overlay');
const exportStatus = document.getElementById('export-status');
const exportProgressBar = document.getElementById('export-progress-bar');
const exportStopBtn = document.getElementById('export-stop-btn');

let framesetExportAborted = false;
let pickedDirHandle = null;

const parseTimeInput = (val) => {
  const parts = val.trim().split(':');
  if (parts.length !== 2) return NaN;
  const m = parseInt(parts[0], 10);
  const s = parseInt(parts[1], 10);
  if (isNaN(m) || isNaN(s)) return NaN;
  return m * 60 + s;
};

const fmtMmSs = (sec) => {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(Math.floor(sec % 60)).padStart(2, '0');
  return `${m}-${s}`;
};

const clearPickedDir = () => {
  pickedDirHandle = null;
  if (framesetDest) {
    framesetDest.readOnly = false;
    framesetDest.classList.remove('picked');
  }
  if (framesetPickBtn) framesetPickBtn.textContent = 'Pick…';
};

captureFramesetBtn?.addEventListener('click', () => {
  if (!previewHasVideo() || isStaticImage) return;
  const dur = previewEl.duration || 0;
  const durMm = String(Math.floor(dur / 60)).padStart(2, '0');
  const durSs = String(Math.floor(dur % 60)).padStart(2, '0');
  framesetFrom.value = '00:00';
  framesetTo.value = `${durMm}:${durSs}`;

  const videoFile = videoInput?.files?.[0];
  const baseName = videoFile ? videoFile.name.replace(/\.[^/.]+$/, '') : 'capture';
  clearPickedDir();
  if (framesetDest) {
    framesetDest.value = `assets/export/frames/${baseName}_00-00_${durMm}-${durSs}_frameset`;
  }
  if (framesetPrefix) {
    framesetPrefix.value = `frame_${baseName}`;
  }

  framesetPopup.hidden = false;
});

framesetPickBtn?.addEventListener('click', async () => {
  if (pickedDirHandle) {
    clearPickedDir();
    return;
  }
  if (!window.showDirectoryPicker) {
    alert('Folder picker not supported in this browser — type a path instead.');
    return;
  }
  try {
    const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
    pickedDirHandle = handle;
    framesetDest.value = handle.name;
    framesetDest.readOnly = true;
    framesetDest.classList.add('picked');
    framesetPickBtn.textContent = 'Clear ✕';
  } catch (err) {
    if (err?.name !== 'AbortError') console.error('Folder pick failed:', err);
  }
});

framesetWholeBtn?.addEventListener('click', () => {
  const dur = previewEl?.duration || 0;
  framesetFrom.value = '00:00';
  const durMm = String(Math.floor(dur / 60)).padStart(2, '0');
  const durSs = String(Math.floor(dur % 60)).padStart(2, '0');
  framesetTo.value = `${durMm}:${durSs}`;
});

framesetCancelBtn?.addEventListener('click', () => {
  framesetPopup.hidden = true;
});

// Cache the offline capture canvas + its DrawingUtils so repeated exports
// (frame-set) don't allocate a new canvas/DrawingUtils per frame.
let capFrameCache = null; // { canvas, ctx, drawingUtils }

const render4KFrame = () => {
  const origCanvas = landmarkCanvas;
  const origCtx = landmarkCtx;
  const origDrawingUtils = drawingUtils;
  const origScale = renderScale;

  const origW = origCanvas.width || 640;
  const origH = origCanvas.height || 360;
  const scale = Math.min(3840 / origW, 2160 / origH);
  const capW = Math.round(origW * scale);
  const capH = Math.round(origH * scale);

  let capCanvas;
  let capCtx;
  let capDrawingUtils;
  if (capFrameCache && capFrameCache.canvas.width === capW && capFrameCache.canvas.height === capH) {
    capCanvas = capFrameCache.canvas;
    capCtx = capFrameCache.ctx;
    capDrawingUtils = capFrameCache.drawingUtils;
    capCtx.clearRect(0, 0, capW, capH);
  } else {
    capCanvas = document.createElement('canvas');
    capCanvas.width = capW;
    capCanvas.height = capH;
    capCtx = capCanvas.getContext('2d');
    capDrawingUtils = new DrawingUtils(capCtx);
    capFrameCache = { canvas: capCanvas, ctx: capCtx, drawingUtils: capDrawingUtils };
  }

  landmarkCanvas = capCanvas;
  landmarkCtx = capCtx;
  drawingUtils = capDrawingUtils;
  renderScale = OVERLAY_RENDER_SCALE;

  const renderSrc = isStaticImage ? imagePreviewEl : previewEl;
  if (showVideoBackground) {
    capCtx.drawImage(renderSrc, 0, 0, capCanvas.width, capCanvas.height);
  } else if (backgroundImage) {
    capCtx.drawImage(backgroundImage, 0, 0, capCanvas.width, capCanvas.height);
  }
  // else: canvas stays transparent (PNG alpha)

  if (faceEnabled && pipelineState.face) drawFaceLandmarks(pipelineState.face);
  if (handEnabled && pipelineState.hands) drawHandLandmarks(pipelineState.hands, pipelineState.gestures);
  if (poseEnabled && pipelineState.pose) drawPoseLandmarks(pipelineState.pose);
  if (objectEnabled && pipelineState.objects) drawObjectDetections(pipelineState.objects);
  if (faceDetectionEnabled && pipelineState.faceDetections) drawFaceDetections(pipelineState.faceDetections);

  landmarkCanvas = origCanvas;
  landmarkCtx = origCtx;
  drawingUtils = origDrawingUtils;
  renderScale = origScale;

  if (invertedModeEnabled && workspaceMode === 'edit') {
    const outCanvas = document.createElement('canvas');
    outCanvas.width = capCanvas.width;
    outCanvas.height = capCanvas.height;
    const outCtx = outCanvas.getContext('2d');
    outCtx.filter = 'grayscale(100%) invert(100%)';
    outCtx.drawImage(capCanvas, 0, 0);
    return outCanvas;
  }

  return capCanvas;
};

const runDetectionsAtCurrentTime = () => {
  const startTimeMs = performance.now();
  const src = isStaticImage ? imagePreviewEl : previewEl;
  pipelineState.face = faceEnabled && faceLandmarker ? faceLandmarker.detectForVideo(src, startTimeMs) : null;
  pipelineState.hands = handEnabled && handLandmarker ? handLandmarker.detectForVideo(src, startTimeMs) : null;
  if (poseEnabled && poseLandmarker) {
    const poseResult = poseLandmarker.detectForVideo(src, startTimeMs);
    pipelineState.pose = hasVisiblePoseLandmarks(poseResult) ? poseResult : null;
  } else {
    pipelineState.pose = null;
  }
  pipelineState.objects = objectEnabled && objectDetector ? objectDetector.detectForVideo(src, startTimeMs) : null;
  pipelineState.gestures = gestureEnabled && gestureRecognizer ? gestureRecognizer.recognizeForVideo(src, Date.now()) : null;
  pipelineState.faceDetections = faceDetectionEnabled && faceDetector ? faceDetector.detectForVideo(src, startTimeMs) : null;
};

const seekTo = (time) => new Promise((resolve) => {
  previewEl.currentTime = time;
  previewEl.addEventListener('seeked', resolve, { once: true });
});

const canvasToBlob = (canvas) => new Promise((resolve) => {
  canvas.toBlob((blob) => resolve(blob), 'image/png');
});

const sanitizePrefix = (raw) => {
  const cleaned = (raw || '').trim().replace(/[^A-Za-z0-9_\-]/g, '_');
  return cleaned || 'frame';
};

const writeFrameToDir = async (handle, filename, blob) => {
  const fh = await handle.getFileHandle(filename, { create: true });
  const writable = await fh.createWritable();
  await writable.write(blob);
  await writable.close();
};

const startFramesetExport = async (fromSec, toSec) => {
  framesetPopup.hidden = true;
  framesetExportAborted = false;

  const destPath = (framesetDest?.value || '').trim();
  const prefix = sanitizePrefix(framesetPrefix?.value);
  const useHandle = !!pickedDirHandle;

  if (!useHandle && !destPath) {
    alert('Destination folder is required.');
    return;
  }

  const interval = 3;
  const times = [];
  for (let t = fromSec; t <= toSec; t += interval) {
    times.push(t);
  }
  const totalFrames = times.length;

  exportOverlay.hidden = false;
  exportProgressBar.style.width = '0%';
  exportStatus.textContent = `Frame 0 / ${totalFrames}`;

  const wasPlaying = !previewEl.paused;
  previewEl.pause();

  for (let i = 0; i < times.length; i++) {
    if (framesetExportAborted) break;

    const t = times[i];
    await seekTo(t);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

    runDetectionsAtCurrentTime();
    const capCanvas = render4KFrame();

    const blob = await canvasToBlob(capCanvas);
    if (!blob || framesetExportAborted) break;

    const mm = String(Math.floor(t / 60)).padStart(2, '0');
    const ss = String(Math.floor(t % 60)).padStart(2, '0');
    const filename = `${prefix}_${mm}-${ss}.png`;

    try {
      if (useHandle) {
        await writeFrameToDir(pickedDirHandle, filename, blob);
      } else {
        const res = await fetch(`/api/capture-frameset-frame-v2?dir=${encodeURIComponent(destPath)}&filename=${encodeURIComponent(filename)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'image/png' },
          body: blob,
        });
        const data = await res.json();
        if (!data.ok) console.error('Frame save failed:', data.error);
      }
    } catch (err) {
      console.error('Frame save failed:', err);
    }

    const pct = ((i + 1) / totalFrames) * 100;
    exportProgressBar.style.width = pct + '%';
    exportStatus.textContent = `Frame ${i + 1} / ${totalFrames}`;
  }

  exportOverlay.hidden = true;

  if (wasPlaying) previewEl.play();
};

framesetStartBtn?.addEventListener('click', () => {
  const fromSec = parseTimeInput(framesetFrom.value);
  const toSec = parseTimeInput(framesetTo.value);
  const dur = previewEl?.duration || 0;

  if (isNaN(fromSec) || isNaN(toSec) || fromSec < 0 || toSec <= fromSec || toSec > Math.ceil(dur)) {
    alert('Invalid time range. Use mm:ss format.');
    return;
  }

  const frameCount = Math.floor((toSec - fromSec) / 3) + 1;
  if (!confirm(`This will export ${frameCount} frame${frameCount === 1 ? '' : 's'}. Continue?`)) {
    return;
  }

  startFramesetExport(fromSec, toSec);
});

exportStopBtn?.addEventListener('click', () => {
  framesetExportAborted = true;
});

form.addEventListener('submit', (e) => e.preventDefault());

const aiControls = document.getElementById('ai-controls');
const sendAnalysisBtn = document.getElementById('send-analysis-btn');
submitBtn?.addEventListener('click', () => {
  if (tabAi && tabData && viewAi && viewData) {
    tabAi.hidden = false;
    tabAi.classList.add('active');
    tabData.classList.remove('active');
    viewAi.hidden = false;
    viewData.hidden = true;
  }
  if (aiControls) aiControls.hidden = false;
});

const runAnalysis = async () => {
  if (!form.video.files.length) {
    setStatus('Please choose a file first.', 'error');
    return;
  }

  const videoFile = form.video.files[0];
  const MAX_SIZE_MB = 100;
  const fileSizeMB = videoFile.size / 1024 / 1024;
  if (fileSizeMB > MAX_SIZE_MB) {
    setStatus(`File too large: ${fileSizeMB.toFixed(1)} MB. Maximum allowed size is ${MAX_SIZE_MB} MB.`, 'error');
    return;
  }

  if (aiControls) aiControls.hidden = true;
  resultSection.hidden = true;
  setStatus('Uploading file and contacting AI…', 'info');
  if (sendAnalysisBtn) sendAnalysisBtn.disabled = true;

  const formData = new FormData();
  formData.append('video', form.video.files[0]);
  const promptValue = promptField?.value?.trim() || '';
  formData.append('prompt', promptValue);

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    let payload;
    try {
      payload = await response.json();
    } catch (_) {
      throw new Error(`Server returned HTTP ${response.status} with no valid response. Check server logs for details.`);
    }
    if (!response.ok) {
      const message = payload?.error || `Analysis failed (HTTP ${response.status}). Please try again or use a smaller video.`;
      if (payload?.geminiResponse) {
        resultText.innerHTML = `<h4>Gemini API Response</h4><pre style="white-space:pre-wrap;color:#fff;font-size:0.8rem;">${payload.geminiResponse}</pre>`;
        resultSection.hidden = false;
      }
      throw new Error(message);
    }

    resultText.innerHTML = formatAnalysisResponse(payload.resultText);
    resultSection.hidden = false;
    setStatus('AI response ready.', 'success');
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Unexpected error. Check your network connection and try again.', 'error');
  } finally {
    if (sendAnalysisBtn) sendAnalysisBtn.disabled = false;
  }
};

sendAnalysisBtn?.addEventListener('click', runAnalysis);

const fullscreenOverlay = document.getElementById('fullscreen-overlay');
const fullscreenContent = document.getElementById('fullscreen-result-content');
const fullscreenOpenBtn = document.getElementById('fullscreen-result-btn');
const fullscreenCloseBtn = document.getElementById('fullscreen-close-btn');

fullscreenOpenBtn?.addEventListener('click', () => {
  if (fullscreenOverlay && fullscreenContent && resultText) {
    fullscreenContent.innerHTML = resultText.innerHTML;
    fullscreenOverlay.hidden = false;
  }
});

fullscreenCloseBtn?.addEventListener('click', () => {
  if (fullscreenOverlay) fullscreenOverlay.hidden = true;
});

fullscreenOverlay?.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && fullscreenOverlay) fullscreenOverlay.hidden = true;
});

const fsPlayerCard = document.querySelector('.players-panel .player-card');
const PAGE_FS_KEY = 'pageFullscreen';

const isTypingTarget = (t) =>
  !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);

document.addEventListener('keydown', (e) => {
  if (e.key !== 'f' && e.key !== 'F') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (isTypingTarget(e.target)) return;

  if (e.shiftKey) {
    e.preventDefault();
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      document.documentElement.requestFullscreen?.();
    }
    return;
  }

  if (!fsPlayerCard || playersPanel?.hidden) return;
  e.preventDefault();
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else {
    fsPlayerCard.requestFullscreen?.();
  }
});

document.addEventListener('fullscreenchange', () => {
  try {
    if (document.fullscreenElement === document.documentElement) {
      localStorage.setItem(PAGE_FS_KEY, '1');
    } else {
      localStorage.removeItem(PAGE_FS_KEY);
    }
  } catch {}
});

// Browsers require a user gesture to enter fullscreen, so we can't auto-restore
// on reload — instead, re-enter on the next keydown/pointerdown if the flag is set.
try {
  if (localStorage.getItem(PAGE_FS_KEY) === '1') {
    const restore = () => {
      document.removeEventListener('keydown', restore, true);
      document.removeEventListener('pointerdown', restore, true);
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen?.().catch(() => {});
      }
    };
    document.addEventListener('keydown', restore, true);
    document.addEventListener('pointerdown', restore, true);
  }
} catch {}
