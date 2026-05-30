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
const backgroundImageInput = document.getElementById('background-image');
const aemaBackgroundBtn = document.getElementById('aema-background-btn');
const videoBgCustomBtns = document.getElementById('video-bg-custom-btns');
const AEMA_BACKGROUND_URL = '/assets/AEMA_logo.svg';
const toggleFace = document.getElementById('toggle-face');
const toggleHand = document.getElementById('toggle-hand');
const togglePose = document.getElementById('toggle-pose');
const toggleObject = document.getElementById('toggle-object');
const toggleGesture = document.getElementById('toggle-gesture');
const toggleFaceDetect = document.getElementById('toggle-face-detect');
const faceStyleSelect = document.getElementById('face-style');
const overlayThicknessSlider = document.getElementById('overlay-thickness');
const overlayThicknessValue = document.getElementById('overlay-thickness-value');
const faceDotDensitySlider = document.getElementById('face-dot-density');
const faceDotDensityValue = document.getElementById('face-dot-density-value');
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
// Analyse-detail unified capture: sidebar button + in-popup single-frame button.
const analyseCaptureBtn = document.getElementById('analyse-capture-btn');
let captureFramePickedDirHandle = null;
let landmarkCtx = landmarkCanvas?.getContext('2d');
if (landmarkCtx) {
  landmarkCtx.imageSmoothingEnabled = true;
  landmarkCtx.imageSmoothingQuality = 'high';
}
const OVERLAY_RENDER_SCALE = 1.5;
let renderScale = OVERLAY_RENDER_SCALE;

const libraryPanel = document.getElementById('library-panel');
const libraryGrid = document.getElementById('library-grid');
const archiveAnalyticsBtn = document.getElementById('archive-analytics-btn');
const appFooter = document.getElementById('app-footer');
const liveIndicator = document.getElementById('live-indicator');
const liveClock = document.getElementById('live-clock');

// Workspace root element used as a per-mode CSS hook (data-mode="live|edit|archive").
// All Analyse-mode layout rules are scoped under .workspace[data-mode="edit"].
const workspaceEl = document.querySelector('.workspace');

// Analyse-mode (edit) preset controls.
const analyseControls = document.getElementById('analyse-controls');
const analysePresets = document.getElementById('analyse-presets');
const analyseTypedesc = document.getElementById('analyse-typedesc');
const analyseMediaInput = document.getElementById('analyse-media-input');
let analyseMediaKind = 'video';
let selectedPresetIndex = null; // no TYPE selected until the user picks one
// Most-recently loaded Analyse clip (Blob), set by loadClipIntoAnalyse so the
// preset-driven run can re-POST it without relying on a file input.
let analyseClipBlob = null;

// Analyse sub-view inside edit mode: 'detail' (#5, opened from Archive — CV
// options + full-width video + transport) or 'main' (#4, opened from Live or
// from #5's ANALISE — TYPE buttons + circumplex + AI response).
let analyseView = 'main';

let workspaceMode = 'live';
let libraryCache = null;
let webcamStream = null;
let webcamWarmupPromise = null; // in-flight getUserMedia, so warm-up is idempotent
let mediaRecorder = null;
let cacheChunks = [];
let cacheRecorderMime = 'video/webm';
let liveMode = false;
let previousWindowBlob = null;
let rotationTimer = null;
let liveClockTimer = null;
const LIVE_WINDOW_MS = 15000;

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
let backgroundImageAlpha = 1;
let backgroundImagePreserveAspect = false; // contain-fit (AEMA bg only); user images stretch to fill
let faceEnabled = toggleFace ? toggleFace.checked : true;
let handEnabled = toggleHand ? toggleHand.checked : true;
let poseEnabled = togglePose ? togglePose.checked : true;
let objectEnabled = toggleObject ? toggleObject.checked : false;
let gestureEnabled = toggleGesture ? toggleGesture.checked : false;
let faceDetectionEnabled = toggleFaceDetect ? toggleFaceDetect.checked : false;
let faceLoopStarted = false;
let faceRenderMode = faceStyleSelect ? faceStyleSelect.value : 'dots';
// Live overlay tuning: global landmark/line size multiplier, and the dots-mode
// stride (draw every Nth face landmark). Both adjusted via sidebar sliders.
let overlayThickness = overlayThicknessSlider ? Number(overlayThicknessSlider.value) || 1 : 1;
let faceDotStep = faceDotDensitySlider ? Math.max(1, Math.round(Number(faceDotDensitySlider.value) || 1)) : 1;
let poseJointsEnabled = togglePoseJoints ? togglePoseJoints.checked : true;
let emotionWheelEnabled = toggleEmotionWheel ? toggleEmotionWheel.checked : true;

// Workspace emotion tracking
// Workspace emotion — smoothed via lerp
let wsTargetValence = 0;
let wsTargetArousal = 0;
let wsValence = 0;
let wsArousal = 0;
const WS_LERP = 0.08;

const isPoseTrailsEnabled = () => (togglePoseTrails ? togglePoseTrails.checked : true);

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
    description: 'Full nonverbal communication, emotion and behavior analysis.',
    prompt: DEFAULT_PROMPT,
  },
  {
    id: 'ekman-naturalness',
    name: 'Ekman + naturalness score',
    description: 'Dominant Ekman emotion plus a naturalness/authenticity score.',
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
  {
    id: 'preset-3',
    name: 'Preset 3',
    description: 'Preset 3 (TODO)',
    prompt: DEFAULT_PROMPT,
  },
  {
    id: 'preset-4',
    name: 'Preset 4',
    description: 'Preset 4 (TODO)',
    prompt: DEFAULT_PROMPT,
  },
  {
    id: 'preset-5',
    name: 'Preset 5',
    description: 'Preset 5 (TODO)',
    prompt: DEFAULT_PROMPT,
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

// Transient awareness toast (top-center, auto-dismissing). Repeated calls reset
// the timer; the element is non-interactive (pointer-events:none via CSS).
const toastEl = document.getElementById('toast');
let toastTimer = null;
let toastHideTimer = null;
function showToast(message, ms = 3500) {
  if (!toastEl) return;
  clearTimeout(toastTimer);
  clearTimeout(toastHideTimer);
  toastEl.textContent = message;
  toastEl.hidden = false;
  // Next frame so the fade-in transition runs from the hidden state.
  requestAnimationFrame(() => toastEl.classList.add('is-visible'));
  toastTimer = setTimeout(() => {
    toastEl.classList.remove('is-visible');
    toastHideTimer = setTimeout(() => { toastEl.hidden = true; }, 250);
  }, ms);
}

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
      drawCustomBackground(landmarkCtx, landmarkCanvas.width, landmarkCanvas.height);
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
    // Mirror the backing-store aspect into CSS so `width:100%; height:auto`
    // resolves to (container.width × video aspect) — guarantees 100% width-fill
    // and proportional height regardless of any flex/grid stretching.
    landmarkCanvas.style.aspectRatio = `${targetW} / ${targetH}`;
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

// Live wall-clock HH:MM:SS (24h, zero-padded) shown in the top-right indicator.
const updateLiveClock = () => {
  if (!liveClock) return;
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, '0');
  const mm = String(now.getMinutes()).padStart(2, '0');
  const ss = String(now.getSeconds()).padStart(2, '0');
  liveClock.textContent = `${hh}:${mm}:${ss}`;
};

function startLiveClock() {
  stopLiveClock();
  updateLiveClock();
  liveClockTimer = setInterval(updateLiveClock, 1000);
}

function stopLiveClock() {
  if (liveClockTimer) { clearInterval(liveClockTimer); liveClockTimer = null; }
}

const updateTransport = () => {
  if (!previewEl) return;
  if (liveMode) {
    // Clock text is driven by the live-clock interval, not playback time.
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
      const dotRadius = 1.5 * renderScale * overlayThickness;
      landmarks.forEach((point, idx) => {
        // Dot density: skip all but every faceDotStep-th landmark.
        if (idx % faceDotStep !== 0) return;
        landmarkCtx.beginPath();
        landmarkCtx.arc(point.x * width, point.y * height, dotRadius, 0, Math.PI * 2);
        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fill();
      });
      return;
    }

    const meshLineWidth = 1 * renderScale * overlayThickness;
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, {
      color: '#FFFFFF',
      lineWidth: meshLineWidth
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
      landmarkCtx.lineWidth = 4 * renderScale * overlayThickness;
      landmarkCtx.stroke();
    });

    landmarks.forEach((point) => {
      landmarkCtx.beginPath();
      landmarkCtx.arc(point.x * width, point.y * height, 4 * renderScale * overlayThickness, 0, Math.PI * 2);
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
  landmarkCtx.lineWidth = 2 * renderScale * overlayThickness;
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
  landmarkCtx.lineWidth = 1 * renderScale * overlayThickness;
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
        lineWidth: 3 * renderScale * overlayThickness
      });
      drawTorsoOverlay(landmarks, width, height);
    }
    if (poseJointsEnabled) {
      drawingUtils.drawLandmarks(landmarks, {
        color: '#FFFFFF',
        radius: 3 * renderScale * overlayThickness
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
      drawCustomBackground(landmarkCtx, landmarkCanvas.width, landmarkCanvas.height);
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

function updateVideoBgCustomControlsVisibility() {
  const showCustom = toggleVideoBg && !toggleVideoBg.checked;
  if (videoBgCustomBtns) videoBgCustomBtns.hidden = !showCustom;
}

function drawCustomBackground(ctx, width, height) {
  if (!backgroundImage) return;
  ctx.save();
  ctx.globalAlpha = backgroundImageAlpha;
  if (backgroundImagePreserveAspect) {
    // Contain-fit: keep the source aspect ratio, centered (AEMA bg only).
    const iw = backgroundImage.naturalWidth || backgroundImage.width || width;
    const ih = backgroundImage.naturalHeight || backgroundImage.height || height;
    const scale = Math.min(width / iw, height / ih);
    const dw = iw * scale;
    const dh = ih * scale;
    ctx.drawImage(backgroundImage, (width - dw) / 2, (height - dh) / 2, dw, dh);
  } else {
    // Default: stretch to fill the canvas.
    ctx.drawImage(backgroundImage, 0, 0, width, height);
  }
  ctx.restore();
}

function applyCustomBackgroundImage(img, alpha = 1, preserveAspect = false) {
  backgroundImage = img;
  backgroundImageAlpha = alpha;
  backgroundImagePreserveAspect = preserveAspect;
  markPreviewDirty();
}

function enableCustomBackgroundMode() {
  if (toggleVideoBg?.checked) {
    toggleVideoBg.checked = false;
    showVideoBackground = false;
  }
  updateVideoBgCustomControlsVisibility();
}

function loadCustomBackgroundFromUrl(url, alpha = 1, preserveAspect = false) {
  const img = new Image();
  img.onload = () => applyCustomBackgroundImage(img, alpha, preserveAspect);
  img.onerror = () => console.error('Failed to load background image:', url);
  img.src = url;
}

toggleVideoBg?.addEventListener('change', (event) => {
  showVideoBackground = event.target.checked;
  updateVideoBgCustomControlsVisibility();
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
      enableCustomBackgroundMode();
      applyCustomBackgroundImage(img);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

aemaBackgroundBtn?.addEventListener('click', () => {
  enableCustomBackgroundMode();
  // 87% transparent → 0.13 opacity; preserve the logo's aspect ratio (contain).
  loadCustomBackgroundFromUrl(AEMA_BACKGROUND_URL, 0.13, true);
});

// Initialize: hide custom background controls when video background is on
if (toggleVideoBg) {
  showVideoBackground = toggleVideoBg.checked;
  updateVideoBgCustomControlsVisibility();
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
    // Mood/emotion analysis relies on face blendshapes — let the user know.
    showToast('Face landmarks off — mood analysis is unavailable.');
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

// Overlay thickness / dot density — the rAF render loop redraws every frame,
// so updating the variable is enough for a live ("on the fly") change.
overlayThicknessSlider?.addEventListener('input', (event) => {
  overlayThickness = Number(event.target.value) || 1;
  if (overlayThicknessValue) overlayThicknessValue.textContent = `${overlayThickness.toFixed(2)}×`;
});

faceDotDensitySlider?.addEventListener('input', (event) => {
  faceDotStep = Math.max(1, Math.round(Number(event.target.value) || 1));
  if (faceDotDensityValue) faceDotDensityValue.textContent = String(faceDotStep);
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

// Analyse mode: the sidebar "Analyse" button runs analysis on the loaded clip
// using the selected preset (no longer a panel toggle). Archive keeps its toggle.
showAnalyticsBtn?.addEventListener('click', () => runSelectedAnalysis());
archiveAnalyticsBtn?.addEventListener('click', toggleAnalyticsPanel);

// ── Analyse-mode media picker (photo / video) ──

// The single "Select media" picker accepts both video and images; the kind is
// derived automatically from the chosen file (see loadMediaIntoAnalysePreview)
// and tracked here only as a fallback for runAnalysis's isImage check.
function setAnalyseMediaKind(kind) {
  analyseMediaKind = kind === 'photo' ? 'photo' : 'video';
}

function handleAnalyseMediaSelection() {
  const file = analyseMediaInput?.files?.[0];
  if (!file) return;
  loadMediaIntoAnalysePreview(file, null);
  if (workspaceMode !== 'edit') switchMode('edit');
  else applyAnalyseView();
}

analyseMediaInput?.addEventListener('change', handleAnalyseMediaSelection);
setAnalyseMediaKind('video');

// ── Analyse-mode preset controls (edit mode only) ──

// Inject TYPE_01..TYPE_05 preset buttons; clicking selects one (tracked in
// selectedPresetIndex) AND immediately runs that preset. Re-renders are idempotent.
function renderAnalysePresets() {
  if (!analysePresets) return;
  analysePresets.innerHTML = '';
  PROMPT_PRESETS.forEach((preset, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'analyse-num' + (i === selectedPresetIndex ? ' is-selected' : '');
    btn.dataset.index = String(i);
    btn.textContent = `TYPE_0${i + 1}`;
    btn.title = preset.name;
    analysePresets.appendChild(btn);
  });
  updateAnalyseTypedesc();
}

// Show the selected preset's description (what that TYPE analyses) in the
// Analyse-main type-description box.
function updateAnalyseTypedesc() {
  if (!analyseTypedesc) return;
  const p = selectedPresetIndex == null ? null : PROMPT_PRESETS[selectedPresetIndex];
  analyseTypedesc.textContent = p?.description || '';
}

// ── TYPE-selection cooldown ──
// Tied to the /api/analyze request lifecycle (not a fixed click timer): the
// cooldown begins when a request is actually sent (beginTypeCooldown, called
// from runAnalysis) and releases only when BOTH the response has settled AND a
// 10 s minimum has elapsed — i.e. lock duration = max(responseTime, 10 s). It
// never unlocks while a request is still in flight. During cooldown the TYPE
// buttons are locked (unselected text 50%-transparent, selected solid black via
// .is-cooldown) and the status section reflects the state. A token guards a
// stale request settling after a newer cooldown has begun.
const TYPE_COOLDOWN_MIN_MS = 10000;
let typeCooldownOn = false;
let typeCooldownToken = 0;
let typeCooldownStart = 0;
let typeCooldownResponded = false;
let typeCooldownTicker = null;
let typeCooldownMinTimer = null;

function typeCooldownActive() {
  return typeCooldownOn;
}

function renderTypeCooldownStatus() {
  if (!typeCooldownResponded) { setStatus('Analysing…', 'info'); return; }
  const secs = Math.max(0, Math.ceil((typeCooldownStart + TYPE_COOLDOWN_MIN_MS - Date.now()) / 1000));
  setStatus(`Cooldown — ${secs}s`, 'info');
}

// Begins the cooldown for an in-flight request; returns a token to release with.
function beginTypeCooldown() {
  if (!analysePresets) return 0;
  const token = ++typeCooldownToken;
  typeCooldownOn = true;
  typeCooldownStart = Date.now();
  typeCooldownResponded = false;
  analysePresets.classList.add('is-cooldown');
  renderTypeCooldownStatus();
  clearInterval(typeCooldownTicker);
  typeCooldownTicker = setInterval(renderTypeCooldownStatus, 250);
  clearTimeout(typeCooldownMinTimer);
  typeCooldownMinTimer = setTimeout(() => maybeReleaseTypeCooldown(token), TYPE_COOLDOWN_MIN_MS);
  return token;
}

// The request settled (response / error / abort). Release once the 10 s
// minimum has also elapsed; otherwise hold the lock until that mark.
function notifyTypeCooldownResponse(token) {
  if (token !== typeCooldownToken) return;
  typeCooldownResponded = true;
  maybeReleaseTypeCooldown(token);
}

function maybeReleaseTypeCooldown(token) {
  if (token !== typeCooldownToken || !typeCooldownOn) return;
  if (!typeCooldownResponded) return; // still waiting for the response → stay locked
  const remain = typeCooldownStart + TYPE_COOLDOWN_MIN_MS - Date.now();
  if (remain <= 0) {
    releaseTypeCooldown(token);
  } else {
    clearTimeout(typeCooldownMinTimer);
    typeCooldownMinTimer = setTimeout(() => releaseTypeCooldown(token), remain);
  }
}

function releaseTypeCooldown(token) {
  if (token !== typeCooldownToken) return;
  typeCooldownOn = false;
  clearInterval(typeCooldownTicker);
  typeCooldownTicker = null;
  clearTimeout(typeCooldownMinTimer);
  typeCooldownMinTimer = null;
  analysePresets?.classList.remove('is-cooldown');
  setStatus('Ready — select a type', 'info');
}

analysePresets?.addEventListener('click', (e) => {
  if (typeCooldownActive()) return; // locked out while a request is in flight / cooling down
  const btn = e.target.closest('.analyse-num');
  if (!btn) return;
  const idx = Number(btn.dataset.index);
  if (Number.isNaN(idx) || idx < 0 || idx >= PROMPT_PRESETS.length) return;
  selectedPresetIndex = idx;
  analysePresets.querySelectorAll('.analyse-num').forEach((b) => {
    b.classList.toggle('is-selected', Number(b.dataset.index) === idx);
  });
  updateAnalyseTypedesc();
  // Paint the selection first, then kick off the (slow) upload on the next tick.
  // runAnalysis starts the cooldown once the request is actually sent.
  setTimeout(runSelectedAnalysis, 0);
});

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

  isStaticImage = false;
  if (imagePreviewEl) {
    imagePreviewEl.removeAttribute('src');
    imagePreviewEl.hidden = true;
  }

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

  const onLoaded = () => {
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

// ── Clip thumbnail + Analyse helpers (Agent A) ───────────────────────────────
// Grabs a single random frame from a video blob and returns it as a PNG Blob.
// Never throws — resolves null on any failure.
async function captureRandomThumbnail(blob) {
  if (!blob) return null;
  return new Promise((resolve) => {
    let url = null;
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      if (url) {
        try { URL.revokeObjectURL(url); } catch (_) {}
      }
      resolve(result);
    };
    try {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'metadata';
      url = URL.createObjectURL(blob);
      video.src = url;

      video.addEventListener('error', () => finish(null));

      video.addEventListener('loadedmetadata', () => {
        const dur = video.duration;
        let seekTo = 0.1;
        if (Number.isFinite(dur) && dur > 0) {
          seekTo = Math.random() * dur;
          if (!Number.isFinite(seekTo)) seekTo = 0.1;
        }
        try {
          video.currentTime = seekTo;
        } catch (_) {
          finish(null);
        }
      });

      video.addEventListener('seeked', () => {
        try {
          const w = video.videoWidth || 640;
          const h = video.videoHeight || 360;
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, w, h);
          canvas.toBlob((png) => finish(png || null), 'image/png');
        } catch (_) {
          finish(null);
        }
      });

      video.load();
    } catch (_) {
      finish(null);
    }
  });
}

// POSTs a PNG thumbnail into the library folder beside its video, reusing the
// existing capture-frameset-frame-v2 endpoint. Returns true/false; never throws.
async function saveThumbnail(videoName, pngBlob) {
  if (!videoName || !pngBlob) return false;
  try {
    const dot = videoName.lastIndexOf('.');
    const base = dot >= 0 ? videoName.slice(0, dot) : videoName;
    const thumbName = `${base}.png`;
    const res = await fetch(
      '/api/capture-frameset-frame-v2?dir=assets/archive/library&filename=' + encodeURIComponent(thumbName),
      { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: pngBlob }
    );
    return res.ok;
  } catch (err) {
    console.error('saveThumbnail failed', err);
    return false;
  }
}

// Loads a clip (path string or Blob) into the shared player, sets it looping,
// and switches to the Analyse (edit) mode. Never throws.
function isImageBlobOrPath(blob, pathOrBlob) {
  if (blob?.type?.startsWith('image/')) return true;
  if (typeof pathOrBlob === 'string') return /\.(jpe?g|png|webp|gif)$/i.test(pathOrBlob);
  return false;
}

function loadMediaIntoAnalysePreview(blob, pathOrBlob) {
  if (!blob) return;
  analyseClipBlob = blob;
  if (isImageBlobOrPath(blob, pathOrBlob)) {
    const file = blob instanceof File
      ? blob
      : new File([blob], 'photo.png', { type: blob.type || 'image/png' });
    showImageInPreview(file);
    setAnalyseMediaKind('photo');
  } else {
    isStaticImage = false;
    showBlobInPreview(blob, 'Clip loaded');
    setAnalyseMediaKind('video');
  }
  if (playersPanel) playersPanel.hidden = false;
}

async function loadClipIntoAnalyse(pathOrBlob) {
  try {
    let blob = pathOrBlob;
    if (typeof pathOrBlob === 'string') {
      const res = await fetch(pathOrBlob);
      blob = await res.blob();
    }
    if (!blob) return;
    loadMediaIntoAnalysePreview(blob, pathOrBlob);
    switchMode('edit'); // applyAnalyseView() (in the edit branch) sets loop per sub-view
  } catch (err) {
    console.error('loadClipIntoAnalyse failed', err);
  }
}

// Apply the current analyse sub-view: CSS hook + per-view playback. Both the
// detail (scrubbable) and main views loop the clip. A2/A3 scope their layouts
// under [data-analyse="…"].
function applyAnalyseView() {
  if (workspaceEl) workspaceEl.dataset.analyse = analyseView;
  if (previewEl) previewEl.loop = true;
  // Detail (#5) shows the Computer-Vision options expanded; a closed <details>
  // can't be reliably un-hidden by CSS alone, so open it in detail.
  document.querySelector('#analyze-form .cv-dropdown')?.toggleAttribute('open', analyseView === 'detail');
}

// Open the Analyse experience at a given sub-view. `clip` (path or Blob) is
// loaded when provided; otherwise the current clip is kept (e.g. detail→main).
function openAnalyse(view, clip) {
  analyseView = view || 'main';
  if (clip !== undefined && clip !== null) {
    loadClipIntoAnalyse(clip); // → switchMode('edit') → applyAnalyseView()
  } else if (workspaceMode === 'edit') {
    applyAnalyseView();
    renderFooter('edit');
  } else {
    switchMode('edit');
  }
}

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

// Placeholder click: Analyse page uses the media picker; legacy path uses #video if present.
videoPlaceholder?.addEventListener('click', () => {
  if (workspaceMode === 'edit' && analyseMediaInput) {
    analyseMediaInput.click();
    return;
  }
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
    // Sidebar button is the Analyse action button — keep its label fixed.
    showAnalyticsBtn.textContent = 'Analyse';
  }
  // Reset analytics panel to closed state
  if (outputsPanel) outputsPanel.hidden = true;
  if (submitBtn) submitBtn.style.display = 'none';
  document.querySelector('.workspace')?.classList.remove('analytics-visible');
  appState = 'workspace';
  // Boot straight into Live mode. switchMode early-returns when mode === workspaceMode,
  // so null it first to force the setup (webcam start, overlay show, sidebar hide).
  workspaceMode = null;
  switchMode('live');
}

// ── Analytics panel helpers ──

function closeAnalyticsPanel() {
  if (outputsPanel) outputsPanel.hidden = true;
  if (submitBtn) submitBtn.style.display = 'none';
  if (tabAi) tabAi.hidden = true;
  // #show-analytics-btn is the Analyse action button; do not relabel it.
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
    // #show-analytics-btn label is fixed ("Analyse"); only the Archive button toggles.
    if (archiveAnalyticsBtn) archiveAnalyticsBtn.textContent = 'Hide analysis';
    document.querySelector('.workspace')?.classList.add('analytics-visible');
  } else {
    closeAnalyticsPanel();
  }
}

// ── Webcam (Live mode) ──

// Acquire the camera once and keep it alive for the whole session, so the first
// switch to Live — and every later one — attaches an already-live stream with no
// getUserMedia lag. Idempotent; returns the live stream or null if access denied.
async function ensureWebcamStream() {
  if (webcamStream && webcamStream.active) return webcamStream;
  if (webcamWarmupPromise) return webcamWarmupPromise;
  if (!navigator.mediaDevices?.getUserMedia) return null;
  webcamWarmupPromise = navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => { webcamStream = stream; return stream; })
    .catch((err) => { console.error('Webcam access denied:', err); return null; })
    .finally(() => { webcamWarmupPromise = null; });
  return webcamWarmupPromise;
}

async function startWebcam() {
  const stream = await ensureWebcamStream();
  if (!stream) return;
  // Clear any clip that was loaded (e.g. from Analyse) so nothing else plays —
  // the warm webcam stream is attached and on screen instantly.
  clearPreview();
  isStaticImage = false;
  previewEl.srcObject = stream;
  previewEl.muted = true;
  previewEl.playsInline = true;
  previewEl.autoplay = true;
  await previewEl.play().catch(() => {});
  enableFaceLandmarks();
  enablePoseLandmarks();
  enableHandLandmarks();
  markPreviewDirty();
  updatePlaceholderVisibility();
  startCacheRecording(stream);
}

// Leaving Live: detach the stream from the player and stop the rolling-buffer
// recorder, but KEEP the camera tracks alive in the background so re-entering
// Live is instant. The tracks are released only on page unload.
function stopWebcam() {
  stopCacheRecording();
  if (previewEl) previewEl.srcObject = null;
  liveMode = false;
  stopLiveClock();
  transportBar?.classList.remove('transport-bar--live');
}

// Warm the camera up right away (during the loader) and release it on unload.
ensureWebcamStream();
window.addEventListener('pagehide', () => {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
}, { once: true });

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
  scheduleRotation(stream);
}

function scheduleRotation(stream) {
  if (rotationTimer) clearTimeout(rotationTimer);
  rotationTimer = setTimeout(() => rotateCacheRecording(stream), LIVE_WINDOW_MS);
}

// Every 10s, finalize the current recorder (so its container is fully playable),
// snapshot it as `previousWindowBlob`, then start a fresh recorder. This is what
// makes "save last 10s" produce a playable WebM/MP4 — a naive chunk-slice would
// miss the initial EBML header and be undecodable.
async function rotateCacheRecording(stream) {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
  const recorder = mediaRecorder;
  const blobType = cacheRecorderMime || 'video/webm';
  const chunksSnapshot = cacheChunks;
  await new Promise((resolve) => {
    recorder.addEventListener('stop', resolve, { once: true });
    try { recorder.stop(); } catch (_) { resolve(); }
  });
  if (chunksSnapshot.length > 0) {
    previousWindowBlob = new Blob(chunksSnapshot, { type: blobType });
  }
  // Only keep the rolling buffer going while still in Live (the stream now stays
  // warm across modes, so webcamStream alone is no longer a "live" signal).
  if (liveMode) startCacheRecording(stream);
}

function stopCacheRecording() {
  if (rotationTimer) { clearTimeout(rotationTimer); rotationTimer = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    try { mediaRecorder.stop(); } catch (_) {}
  }
  mediaRecorder = null;
  previousWindowBlob = null;
}

// ── Mode switching (Processing / Archive / Live) ──

function switchMode(mode) {
  if (mode === workspaceMode) return;

  // Stop webcam when leaving live mode
  if (workspaceMode === 'live') {
    stopWebcam();
  }

  workspaceMode = mode;

  // Per-mode CSS hook: drives all .workspace[data-mode="edit"] layout rules.
  if (workspaceEl) workspaceEl.dataset.mode = mode;

  // Analyse-mode preset controls are only present in edit mode.
  if (analyseControls) analyseControls.hidden = (mode !== 'edit');

  // Shared silver footer (per-page buttons, current page's button inactive).
  renderFooter(mode);
  // Top-right LIVE indicator (red dot + clock) shows only in Live.
  if (liveIndicator) liveIndicator.hidden = (mode !== 'live');

  // Close analytics panel + capture menu on mode switch (the latter also
  // detaches the capture menu's document-level click/keydown listeners).
  closeAnalyticsPanel();
  closeCaptureMenu();

  // Hide all sidebar panels + overlays first
  form.classList.add('hidden');
  if (libraryPanel) libraryPanel.hidden = true;
  if (archiveAnalyticsBtn) archiveAnalyticsBtn.style.display = 'none';

  if (mode === 'edit') {
    form.classList.remove('hidden');
    // Apply the analyse sub-view (#5 detail vs #4 main) CSS hook + playback.
    applyAnalyseView();
    if (playersPanel) playersPanel.hidden = false;
    enableFaceLandmarks();
    // Force the analytics panel open and reveal BOTH views (circumplex + result).
    // CSS scoped to .workspace[data-mode="edit"] hides the tabs/blendshapes/ai-controls
    // and lays the two views out together; the [hidden] toggling from the tab JS is
    // neutralised by un-hiding both here so existing animation/render code keeps running.
    if (outputsPanel) outputsPanel.hidden = false;
    if (viewData) viewData.hidden = false;
    if (viewAi) viewAi.hidden = false;
    workspaceEl?.classList.add('analytics-visible');
    renderAnalysePresets();
  } else if (mode === 'archive') {
    // Full-screen thumbnail grid only — no player, no analytics, no sidebar.
    // (.controls-panel + .center-column are hidden via CSS scoped to archive.)
    if (libraryPanel) libraryPanel.hidden = false;
    fetchAndRenderLibrary();
  } else if (mode === 'live') {
    // Full-bleed live video; the bottom #app-footer + top indicator are the
    // only chrome — the old in-player transport bar is not used here.
    if (playersPanel) playersPanel.hidden = false;
    liveMode = true;
    if (transportBar) {
      transportBar.hidden = true;
      transportBar.classList.remove('transport-bar--live');
    }
    startLiveClock();
    // Hide capture buttons
    if (captureFrameGroup) captureFrameGroup.style.display = 'none';
    if (captureFramesetBtn) captureFramesetBtn.style.display = 'none';
    if (archiveAnalyticsBtn) archiveAnalyticsBtn.style.display = 'none';
    startWebcam();
  }

}

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

// Derive a human-readable creation timestamp from a clip filename.
// Matches the `live_YYYY-MM-DD_HH-mm-ss` (or legacy `exhibition_…`) pattern and
// renders it as `YYYY-MM-DD HH:MM:SS`; otherwise returns the raw filename.
function clipTimestampLabel(name) {
  // Saved clips are named `<prefix>_YYYYMMDD_HH-mm-ss.<ext>` (see server.js
  // /api/archive-clip); also tolerate a dashed date `YYYY-MM-DD_HH-mm-ss`.
  const n = name || '';
  let m = /(\d{4})-?(\d{2})-?(\d{2})_(\d{2})-(\d{2})-(\d{2})/.exec(n);
  if (m) {
    const [, y, mo, d, h, mi, s] = m;
    return `${y}-${mo}-${d} ${h}:${mi}:${s}`;
  }
  return n;
}

function renderLibrary(items) {
  if (!libraryGrid) return;
  libraryGrid.innerHTML = '';
  // Archive grid shows VIDEO clips only.
  const clips = (items || []).filter((it) => it.type === 'video');
  if (!clips.length) {
    libraryGrid.innerHTML = '<p class="library-empty">No clips yet</p>';
    return;
  }
  clips.forEach((item) => {
    const tile = document.createElement('div');
    tile.className = 'archive-tile';
    tile.dataset.path = item.path;

    const thumb = document.createElement('div');
    thumb.className = 'archive-tile-thumb';

    if (item.thumb) {
      const img = document.createElement('img');
      img.src = item.thumb;
      img.alt = '';
      img.loading = 'lazy';
      thumb.appendChild(img);
    } else {
      // No sibling thumbnail PNG → fall back to a metadata first-frame.
      const vid = document.createElement('video');
      vid.src = item.path;
      vid.preload = 'metadata';
      vid.muted = true;
      vid.addEventListener('loadeddata', () => { vid.currentTime = 0.01; });
      thumb.appendChild(vid);
    }
    tile.appendChild(thumb);

    const bar = document.createElement('div');
    bar.className = 'archive-tile-bar';
    bar.textContent = clipTimestampLabel(item.name);
    tile.appendChild(bar);

    libraryGrid.appendChild(tile);
  });
}

libraryGrid?.addEventListener('click', (e) => {
  const tile = e.target.closest('.archive-tile');
  if (!tile) return;
  const itemPath = tile.dataset.path;
  if (!itemPath) return;
  // Open the clip in the Analyse-detail (#5) screen.
  openAnalyse('detail', itemPath);
});

// ── Live: Save & analise button — save last 10s + analyze ──

// Opens analytics panel and switches to the Behavior Analysis (AI) tab.
function openAnalyticsAiTab() {
  if (outputsPanel) outputsPanel.hidden = false;
  document.querySelector('.workspace')?.classList.add('analytics-visible');
  if (tabAi) { tabAi.hidden = false; tabAi.classList.add('active'); }
  if (tabData) tabData.classList.remove('active');
  if (viewAi) viewAi.hidden = false;
  if (viewData) viewData.hidden = true;
  // #show-analytics-btn label is fixed ("Analyse"); do not relabel it here.
}

// Save the last ~15s of live webcam (raw, no overlay) + a random-frame
// thumbnail to the library, then open the clip in the Analyse page.
let liveSaveBusy = false;
async function saveLiveClipAndAnalyse() {
  if (liveSaveBusy || !mediaRecorder || !webcamStream) return;
  if (!cacheChunks.length && !previousWindowBlob) return;

  liveSaveBusy = true;
  // Cancel pending rotation so it can't race the manual stop.
  if (rotationTimer) { clearTimeout(rotationTimer); rotationTimer = null; }

  const recorder = mediaRecorder;
  const blobType = cacheRecorderMime || 'video/webm';
  const currentChunks = cacheChunks;
  const currentBlob = await new Promise((resolve) => {
    recorder.addEventListener('stop', () => {
      resolve(currentChunks.length ? new Blob(currentChunks, { type: blobType }) : null);
    }, { once: true });
    try { recorder.stop(); } catch (_) {
      resolve(currentChunks.length ? new Blob(currentChunks, { type: blobType }) : null);
    }
  });

  // Prefer the just-finalized current window if it has ≥3s of material;
  // otherwise fall back to the previous fully-finalized window.
  const finalBlob = (currentChunks.length >= 3 && currentBlob) ? currentBlob
    : (previousWindowBlob || currentBlob);

  if (liveMode && webcamStream) startCacheRecording(webcamStream);

  if (!finalBlob || finalBlob.size === 0) { liveSaveBusy = false; return; }

  try {
    const postType = blobType.startsWith('video/mp4') ? 'video/mp4' : 'video/webm';
    const saveRes = await fetch('/api/archive-clip', {
      method: 'POST',
      headers: { 'Content-Type': postType },
      body: finalBlob,
    });
    const saveData = await saveRes.json();
    if (!saveData.ok) { console.error('Failed to save clip'); return; }
    libraryCache = null;
    const png = await captureRandomThumbnail(finalBlob);
    if (png) await saveThumbnail(saveData.name, png);
    analyseView = 'main'; // Live ANALISE opens the #4 main analyse screen
    await loadClipIntoAnalyse(finalBlob);
  } catch (err) {
    console.error('Failed to save clip:', err);
  } finally {
    liveSaveBusy = false;
  }
}

// ── Shared silver footer (all pages). The current page's own nav button is
// shown inactive; Archive has no ANALISE button. ──
const FOOTER_SPEC = {
  live:    [['live', 'LIVE'], ['analyse', 'ANALISE'], ['archive', 'ARCHIVE']],
  edit:    [['live', 'LIVE'], ['analyse', 'ANALISE'], ['archive', 'ARCHIVE']],
  archive: [['live', 'LIVE'], ['archive', 'ARCHIVE']],
};

function renderFooter(mode) {
  if (!appFooter) return;
  appFooter.innerHTML = '';
  (FOOTER_SPEC[mode] || FOOTER_SPEC.live).forEach(([action, label]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'footer-btn';
    b.dataset.action = action;
    b.textContent = label;
    let inactive = (action === 'live' && mode === 'live')
                || (action === 'archive' && mode === 'archive');
    // On the main analyse screen ANALISE is the current page → inactive;
    // on the detail screen it stays active (advances to main).
    if (action === 'analyse' && mode === 'edit' && analyseView === 'main') inactive = true;
    if (inactive) { b.classList.add('is-inactive'); b.disabled = true; }
    appFooter.appendChild(b);
  });
}

appFooter?.addEventListener('click', (e) => {
  const b = e.target.closest('.footer-btn');
  if (!b || b.disabled) return;
  const action = b.dataset.action;
  if (action === 'live') switchMode('live');
  else if (action === 'archive') switchMode('archive');
  else if (action === 'analyse') {
    // Live → save 15s + open #4 main. Detail (#5) → advance to #4 main.
    if (workspaceMode === 'live') saveLiveClipAndAnalyse();
    else openAnalyse('main');
  }
});

// Loading screen: spinning AEMA logo. Dismiss on click or 2s timeout.
const loaderOverlay = document.getElementById('loader-overlay');

function endLoader() {
  if (appState === 'workspace') return;
  if (loaderOverlay) loaderOverlay.hidden = true;
  showWorkspace();
}

// Clicking the loader to enter the app is a user gesture, so we can request
// document fullscreen here — this hides the browser headbar/chrome. (The 2 s
// auto-dismiss path can't: the Fullscreen API requires a user gesture.) The
// existing fullscreenchange handler persists the state via PAGE_FS_KEY, and
// Shift+F still toggles it manually.
loaderOverlay?.addEventListener('click', () => {
  document.documentElement.requestFullscreen?.().catch(() => {});
  endLoader();
});
setTimeout(endLoader, 2000);


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

// Capture the current frame at up to 4K (overlays included) and save it to the
// server default dir (assets/export/frames/). Works for video and still images.
function captureCurrentFrame() {
  const hasMedia = isStaticImage ? !!imagePreviewEl?.src : previewHasVideo();
  if (!hasMedia || !landmarkCanvas) return;
  const mediaFile = analyseMediaInput?.files?.[0] || videoInput?.files?.[0];
  const baseName = mediaFile ? mediaFile.name.replace(/\.[^/.]+$/, '') : 'capture';
  const time = previewEl?.currentTime ?? 0;
  const mm = String(Math.floor(time / 60)).padStart(2, '0');
  const ss = String(Math.floor(time % 60)).padStart(2, '0');
  const filename = `frame_${baseName}_${mm}-${ss}.png`;

  runDetectionsAtCurrentTime();
  const capCanvas = render4KFrame();
  capCanvas.toBlob(async (blob) => {
    if (!blob) return;
    try {
      const res = await fetch(`/api/capture-frame?filename=${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'image/png' },
        body: blob,
      });
      const data = await res.json();
      if (!data.ok) console.error('Capture failed:', data.error);
    } catch (err) {
      console.error('Capture failed:', err);
    }
  }, 'image/png');
}

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
const framesetCaptureFrameBtn = document.getElementById('frameset-capture-frame-btn');
const framesetSetSection = document.getElementById('frameset-set-section');
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

// Open the unified capture popup. Always offers "Capture current frame"; the
// frame-set (range export) section is shown only for video, hidden for images.
function openCapturePopup() {
  const hasMedia = isStaticImage ? !!imagePreviewEl?.src : previewHasVideo();
  if (!hasMedia) return;

  const showSet = !isStaticImage;
  if (framesetSetSection) framesetSetSection.hidden = !showSet;
  if (framesetStartBtn) framesetStartBtn.hidden = !showSet;

  if (showSet) {
    const dur = previewEl.duration || 0;
    const durMm = String(Math.floor(dur / 60)).padStart(2, '0');
    const durSs = String(Math.floor(dur % 60)).padStart(2, '0');
    framesetFrom.value = '00:00';
    framesetTo.value = `${durMm}:${durSs}`;

    const mediaFile = analyseMediaInput?.files?.[0] || videoInput?.files?.[0];
    const baseName = mediaFile ? mediaFile.name.replace(/\.[^/.]+$/, '') : 'capture';
    clearPickedDir();
    if (framesetDest) {
      framesetDest.value = `assets/export/frames/${baseName}_00-00_${durMm}-${durSs}_frameset`;
    }
    if (framesetPrefix) {
      framesetPrefix.value = `frame_${baseName}`;
    }
  }

  framesetPopup.hidden = false;
}

analyseCaptureBtn?.addEventListener('click', openCapturePopup);
captureFramesetBtn?.addEventListener('click', openCapturePopup);

framesetCaptureFrameBtn?.addEventListener('click', () => {
  captureCurrentFrame();
  framesetPopup.hidden = true;
  showToast('Frame captured.');
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
    drawCustomBackground(capCtx, capCanvas.width, capCanvas.height);
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

// Aborts the previous in-flight /api/analyze fetch when a new TYPE_0x click
// fires, so rapid clicks can't pile up concurrent uploads (Gemini 429 rate-limit).
let analyseAbortController = null;

const runAnalysis = async () => {
  const fileFromPicker = analyseMediaInput?.files?.[0] || null;
  const fileFromLegacy = form?.video?.files?.[0] || null;
  const mediaFile = fileFromPicker || fileFromLegacy || analyseClipBlob;
  if (!mediaFile) {
    setStatus('Select a photo or video first.', 'error');
    return;
  }
  // Cancel any previous request still in flight.
  if (analyseAbortController) { try { analyseAbortController.abort(); } catch (_) {} }
  analyseAbortController = new AbortController();
  const signal = analyseAbortController.signal;

  const isImage =
    mediaFile.type?.startsWith('image/') ||
    (fileFromPicker && analyseMediaKind === 'photo') ||
    (fileFromLegacy && fileFromLegacy.type?.startsWith('image/'));

  const MAX_SIZE_MB = 250;
  const fileSizeMB = mediaFile.size / 1024 / 1024;
  if (fileSizeMB > MAX_SIZE_MB) {
    setStatus(`File too large: ${fileSizeMB.toFixed(1)} MB. Maximum allowed size is ${MAX_SIZE_MB} MB.`, 'error');
    return;
  }

  if (aiControls) aiControls.hidden = true;
  resultSection.hidden = true;
  setStatus('Sending for analysis', 'info');
  if (sendAnalysisBtn) sendAnalysisBtn.disabled = true;
  if (showAnalyticsBtn) showAnalyticsBtn.disabled = true;

  const formData = new FormData();
  // Multer accepts a Blob; give it a filename so the extension/mime survive.
  let mediaName = fileFromPicker?.name || fileFromLegacy?.name;
  if (!mediaName) {
    if (isImage) {
      const ext = (mediaFile.type || '').includes('jpeg') ? 'jpg' : 'png';
      mediaName = `photo.${ext}`;
    } else {
      mediaName = `clip.${(mediaFile.type || 'video/webm').includes('mp4') ? 'mp4' : 'webm'}`;
    }
  }
  formData.append('video', mediaFile, mediaName);
  const promptValue = promptField?.value?.trim() || '';
  formData.append('prompt', promptValue);

  // The request is now being sent — start the TYPE cooldown (Analyse-main only).
  const cdToken = (workspaceMode === 'edit' && analyseView === 'main') ? beginTypeCooldown() : 0;

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
      signal,
    });

    let payload;
    try {
      payload = await response.json();
    } catch (_) {
      throw new Error(`Server returned HTTP ${response.status} with no valid response. Check server logs for details.`);
    }
    if (!response.ok) {
      const message = payload?.error || `Analysis failed (HTTP ${response.status}). Please try again or use a smaller file.`;
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
    // Aborted requests are expected when the user re-clicks; don't show an error.
    if (error?.name === 'AbortError' || signal.aborted) return;
    console.error(error);
    setStatus(error.message || 'Unexpected error. Check your network connection and try again.', 'error');
  } finally {
    if (sendAnalysisBtn) sendAnalysisBtn.disabled = false;
    if (showAnalyticsBtn) showAnalyticsBtn.disabled = false;
    // Response settled — release the cooldown once the 10 s minimum has also passed.
    if (cdToken) notifyTypeCooldownResponse(cdToken);
  }
};

// Analyse-mode entry point: apply the selected preset's prompt, then run.
// Triggered by a TYPE_0x preset click and the sidebar #show-analytics-btn.
function runSelectedAnalysis() {
  const preset = PROMPT_PRESETS[selectedPresetIndex] || PROMPT_PRESETS[0];
  if (promptField && preset) promptField.value = preset.prompt;
  return runAnalysis();
}

sendAnalysisBtn?.addEventListener('click', runAnalysis);

const fullscreenOverlay = document.getElementById('fullscreen-overlay');
const fullscreenCloseBtn = document.getElementById('fullscreen-close-btn');

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

// Hidden global font-size shortcut: Cmd/Ctrl + (+/=) grows, (-) shrinks, (0) resets.
// Persists across reloads. Skipped when typing in inputs/textareas.
const FONT_SIZE_KEY = 'fontSizePx';
const FONT_SIZE_BASE = 16;
const FONT_SIZE_MIN = 10;
const FONT_SIZE_MAX = 32;

const applyFontSize = (px) => {
  document.documentElement.style.fontSize = `${px}px`;
};

try {
  const stored = parseInt(localStorage.getItem(FONT_SIZE_KEY), 10);
  if (Number.isFinite(stored) && stored >= FONT_SIZE_MIN && stored <= FONT_SIZE_MAX) {
    applyFontSize(stored);
  }
} catch {}

document.addEventListener('keydown', (e) => {
  if (!(e.metaKey || e.ctrlKey)) return;
  if (isTypingTarget(e.target)) return;
  const k = e.key;
  if (k !== '+' && k !== '=' && k !== '-' && k !== '0') return;
  e.preventDefault();
  const current = parseInt(getComputedStyle(document.documentElement).fontSize, 10) || FONT_SIZE_BASE;
  let next = current;
  if (k === '+' || k === '=') next = Math.min(FONT_SIZE_MAX, current + 1);
  else if (k === '-') next = Math.max(FONT_SIZE_MIN, current - 1);
  else if (k === '0') next = FONT_SIZE_BASE;
  applyFontSize(next);
  try { localStorage.setItem(FONT_SIZE_KEY, String(next)); } catch {}
});

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
