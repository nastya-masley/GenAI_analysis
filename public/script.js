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
const menuToggle = document.getElementById('menu-toggle');
const statusEl = document.getElementById('status');
const resultSection = document.getElementById('result');
const resultText = document.getElementById('result-text');
const submitBtn = document.getElementById('submit-btn');
const showAnalyticsBtn = document.getElementById('show-analytics-btn');
const outputsPanel = document.querySelector('.outputs-panel');
const previewEl = document.getElementById('preview');
const videoPlaceholder = document.getElementById('video-placeholder');
const canvasPlaceholder = document.getElementById('canvas-placeholder');
const videoInput = document.getElementById('video');
const uploadButtonLabel = document.getElementById('upload-btn-label');
const selectedFileHint = document.getElementById('selected-file-hint');
const togglePromptBtn = document.getElementById('toggle-prompt');
const promptField = document.getElementById('prompt');
const toggleVideoBg = document.getElementById('toggle-video-bg');
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
const landmarkCanvas = document.getElementById('landmark-canvas');
const blendShapeList = document.getElementById('blend-shape-list');
const emotionWheelCanvas = document.getElementById('emotion-wheel-canvas');
const emotionWheelCtx = emotionWheelCanvas?.getContext('2d');
const emotionWheelContainer = document.getElementById('emotion-wheel-container');
const emotionWheelName = document.getElementById('emotion-wheel-name');
const ekmanLegendWorkspace = document.getElementById('ekman-legend-workspace');
const emotionResultValence = document.getElementById('emotion-result-valence');
const emotionResultArousal = document.getElementById('emotion-result-arousal');
const tabData = document.getElementById('tab-data');
const tabAi = document.getElementById('tab-ai');
const viewData = document.getElementById('view-data');
const viewAi = document.getElementById('view-ai');
const verdictCard = document.getElementById('verdict-card');
const landmarkCtx = landmarkCanvas?.getContext('2d');

let promptVisible = false;
let previewObjectUrl = null;

let faceLandmarker;
let handLandmarker;
let poseLandmarker;
let objectDetector;
let gestureRecognizer;
let faceDetector;
let drawingUtils = null;
let runningMode = 'IMAGE';
let lastVideoTime = -1;
let showVideoBackground = false;
let backgroundImage = null;
let faceEnabled = toggleFace ? toggleFace.checked : true;
let handEnabled = toggleHand ? toggleHand.checked : true;
let poseEnabled = togglePose ? togglePose.checked : true;
let objectEnabled = toggleObject ? toggleObject.checked : true;
let gestureEnabled = toggleGesture ? toggleGesture.checked : true;
let faceDetectionEnabled = toggleFaceDetect ? toggleFaceDetect.checked : true;
let faceLoopStarted = false;
let faceRenderMode = faceStyleSelect ? faceStyleSelect.value : 'dots';
let poseJointsEnabled = togglePoseJoints ? togglePoseJoints.checked : true;
let emotionWheelEnabled = toggleEmotionWheel ? toggleEmotionWheel.checked : true;

// Workspace emotion tracking (mirrors webcam trail)
// Workspace emotion — smoothed via lerp
let wsTargetValence = 0;
let wsTargetArousal = 0;
let wsValence = 0;
let wsArousal = 0;
const wsEmotionTrail = [];
const WS_MAX_TRAIL = 30;
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

const EKMAN_EMOTIONS = [
  { label: 'HAPPINESS', v:  0.82, a:  0.20 },
  { label: 'SURPRISE',  v:  0.05, a:  0.85 },
  { label: 'FEAR',      v: -0.55, a:  0.72 },
  { label: 'ANGER',     v: -0.68, a:  0.44 },
  { label: 'DISGUST',   v: -0.72, a:  0.02 },
  { label: 'SADNESS',   v: -0.50, a: -0.60 },
];

const getDominantEmotion = (v, a) => {
  let closest = EKMAN_EMOTIONS[0];
  let minDist = Infinity;
  EKMAN_EMOTIONS.forEach(e => {
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

const clearEmotionWheel = () => {
  if (!emotionWheelCanvas || !emotionWheelCtx) return;
  emotionWheelCtx.clearRect(0, 0, emotionWheelCanvas.width, emotionWheelCanvas.height);
};

// Set target values — the animation loop will lerp towards them
const renderEmotionWheel = ({ valence, arousal }) => {
  wsTargetValence = valence;
  wsTargetArousal = arousal;
};

let wsTrailTimer = 0;

function drawEmotionWheel(timestamp) {
  if (!emotionWheelCanvas || !emotionWheelCtx) return;

  // Lerp towards target
  wsValence += (wsTargetValence - wsValence) * WS_LERP;
  wsArousal += (wsTargetArousal - wsArousal) * WS_LERP;

  // Push trail point every ~5 frames
  wsTrailTimer++;
  if (wsTrailTimer >= 5) {
    wsTrailTimer = 0;
    wsEmotionTrail.push({ valence: wsValence, arousal: wsArousal });
    if (wsEmotionTrail.length > WS_MAX_TRAIL) wsEmotionTrail.shift();
  }

  const ctx = emotionWheelCtx;
  const size = emotionWheelCanvas.width;
  const center = size / 2;
  const radius = center - 24;

  ctx.clearRect(0, 0, size, size);

  // Circular background
  ctx.beginPath();
  ctx.arc(center, center, radius + 20, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fill();

  // Quadrant tints
  const quadrants = [
    { startAngle: -Math.PI / 2, color: 'rgba(100,200,100,0.04)' },
    { startAngle: 0, color: 'rgba(100,100,200,0.04)' },
    { startAngle: Math.PI / 2, color: 'rgba(200,100,100,0.04)' },
    { startAngle: Math.PI, color: 'rgba(200,200,100,0.04)' },
  ];
  quadrants.forEach(({ startAngle, color }) => {
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.arc(center, center, radius, startAngle, startAngle + Math.PI / 2);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  });

  // Dashed crosshair
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.moveTo(center - radius, center);
  ctx.lineTo(center + radius, center);
  ctx.moveTo(center, center - radius);
  ctx.lineTo(center, center + radius);
  ctx.stroke();
  ctx.setLineDash([]);

  // Dashed ring
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.arc(center, center, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Ekman markers
  const dominant = getDominantEmotion(wsValence, wsArousal);
  EKMAN_EMOTIONS.forEach(e => {
    const ex = center + e.v * radius;
    const ey = center - e.a * radius;
    const isDominant = e.label === dominant.label;

    if (isDominant) {
      ctx.beginPath();
      ctx.arc(ex, ey, 14, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(ex, ey, isDominant ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isDominant ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.3)';
    ctx.fill();
  });

  // Trail
  wsEmotionTrail.forEach((point, i) => {
    const opacity = ((i + 1) / wsEmotionTrail.length) * 0.5;
    const px = center + point.valence * radius;
    const py = center - point.arousal * radius;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${opacity})`;
    ctx.fill();
  });

  // Pulsing pointer
  const pulse = 0.5 + 0.5 * Math.sin(timestamp / 400);
  const px = center + wsValence * radius;
  const py = center - wsArousal * radius;
  ctx.beginPath();
  ctx.arc(px, py, 8 + pulse * 4, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(255,255,255,${0.15 + pulse * 0.15})`;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(px, py, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();

  // Update labels
  if (emotionWheelName) emotionWheelName.textContent = dominant.label;
  if (emotionResultValence) emotionResultValence.textContent = `V ${wsValence.toFixed(2)}`;
  if (emotionResultArousal) emotionResultArousal.textContent = `A ${wsArousal.toFixed(2)}`;
  updateWorkspaceLegend(dominant);
}

// Continuous animation loop for smooth workspace circumplex
function animateEmotionWheel(timestamp) {
  drawEmotionWheel(timestamp);
  requestAnimationFrame(animateEmotionWheel);
}
requestAnimationFrame(animateEmotionWheel);

// Workspace Ekman legend
function buildWorkspaceLegend() {
  if (!ekmanLegendWorkspace) return;
  ekmanLegendWorkspace.innerHTML = '';
  EKMAN_EMOTIONS.forEach(e => {
    const li = document.createElement('li');
    li.textContent = e.label;
    li.dataset.emotion = e.label;
    ekmanLegendWorkspace.appendChild(li);
  });
}

function updateWorkspaceLegend(dominant) {
  if (!ekmanLegendWorkspace) return;
  ekmanLegendWorkspace.querySelectorAll('li').forEach(li => {
    li.classList.toggle('active', li.dataset.emotion === dominant.label);
  });
}

buildWorkspaceLegend();

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

const previewHasVideo = () =>
  previewEl && (previewEl.readyState >= 2 || !!previewEl.srcObject);

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

const updateCanvasDimensions = () => {
  if (!previewEl || !landmarkCanvas) return;
  // Match canvas dimensions to video natural dimensions for accurate rendering
  // CSS object-fit: contain will handle aspect ratio fitting
  const width = previewEl.videoWidth || previewEl.clientWidth || 640;
  const height = previewEl.videoHeight || previewEl.clientHeight || 360;
  if (landmarkCanvas.width !== width || landmarkCanvas.height !== height) {
    landmarkCanvas.width = width;
    landmarkCanvas.height = height;
  }
};

const updatePlayerOrientation = () => {
  if (!playersPanel || !previewEl) return;
  const videoWidth = previewEl.videoWidth || previewEl.clientWidth;
  const videoHeight = previewEl.videoHeight || previewEl.clientHeight;
  if (!videoWidth || !videoHeight) return;
  const isLandscape = videoWidth / Math.max(videoHeight, 1) >= 1;
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
        landmarkCtx.arc(point.x * width, point.y * height, 1.5, 0, Math.PI * 2);
        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fill();
      });
      return;
    }

    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: '#FFFFFF',
      lineWidth: 1
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
      landmarkCtx.lineWidth = 4;
      landmarkCtx.stroke();
    });

    landmarks.forEach((point) => {
      landmarkCtx.beginPath();
      landmarkCtx.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
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
        landmarkCtx.font = '12px "OCR A Extended", monospace';
        const labelX = wrist.x * width;
        const labelY = wrist.y * height - 10;
        const textWidth = landmarkCtx.measureText(text).width + 16;
        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fillRect(labelX - 8, labelY - 26, textWidth, 24);
        landmarkCtx.strokeStyle = '#000';
        landmarkCtx.lineWidth = 1;
        landmarkCtx.strokeRect(labelX - 8, labelY - 26, textWidth, 24);
        landmarkCtx.fillStyle = '#000';
        landmarkCtx.fillText(text, labelX - 4, labelY - 10);
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
  landmarkCtx.lineWidth = 2;
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
  landmarkCtx.lineWidth = 1;
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
        lineWidth: 3
      });
      drawTorsoOverlay(landmarks, width, height);
    }
    if (poseJointsEnabled) {
      drawingUtils.drawLandmarks(landmarks, {
        color: '#FFFFFF',
        radius: 3
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
    landmarkCtx.lineWidth = 4;
      landmarkCtx.strokeRect(originX, originY, width, height);
      const label = detection.categories?.[0];
      if (label) {
        const text = `${label.categoryName || 'Object'} ${(label.score * 100).toFixed(1)}%`;
        landmarkCtx.font = '16px "OCR A Extended", monospace';
        const textWidth = landmarkCtx.measureText(text).width;
        const labelHeight = 30;
        const padding = 10;
        const boxWidth = textWidth + padding * 2;
        let boxX = originX;
        const boxY = Math.max(originY - labelHeight - 4, 0);

        // Clamp boxX to be within canvas width
        if (boxX + boxWidth > landmarkCanvas.width) {
          boxX = landmarkCanvas.width - boxWidth;
        }
        if (boxX < 0) boxX = 0;

        landmarkCtx.fillStyle = '#FFFFFF';
        landmarkCtx.fillRect(boxX, boxY, boxWidth, labelHeight);
        landmarkCtx.strokeStyle = '#000';
        landmarkCtx.lineWidth = 2;
        landmarkCtx.strokeRect(boxX, boxY, boxWidth, labelHeight);
        landmarkCtx.fillStyle = '#000';
        landmarkCtx.fillText(text, boxX + padding, boxY + labelHeight - 10);
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
    landmarkCtx.lineWidth = 4;
    landmarkCtx.strokeRect(originX, originY, width, height);
    const label = detection.categories?.[0];
    const text = label
      ? `${label.categoryName || 'Face'} ${(label.score * 100).toFixed(1)}%`
      : 'Face';
    landmarkCtx.font = '16px "OCR A Extended", monospace';
    const textWidth = landmarkCtx.measureText(text).width;
    const padding = 10;
    const labelHeight = 30;
    const boxWidth = textWidth + padding * 2;
    let boxX = originX;
    const boxY = Math.max(originY - labelHeight - 4, 0);

    // Clamp boxX to be within canvas width
    if (boxX + boxWidth > landmarkCanvas.width) {
      boxX = landmarkCanvas.width - boxWidth;
    }
    if (boxX < 0) boxX = 0;

    landmarkCtx.fillStyle = '#FFFFFF';
    landmarkCtx.fillRect(boxX, boxY, boxWidth, labelHeight);
    landmarkCtx.strokeStyle = '#000';
    landmarkCtx.lineWidth = 2;
    landmarkCtx.strokeRect(boxX, boxY, boxWidth, labelHeight);
    landmarkCtx.fillStyle = '#000';
    landmarkCtx.fillText(text, boxX + padding, boxY + labelHeight - 10);
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

  if ((!faceLandmarker && !handLandmarker && !poseLandmarker && !objectDetector) || !landmarkCtx) {
    return;
  }

  if (!previewHasVideo()) {
    resetFaceOutputs();
    return;
  }

  updatePlayerOrientation();
  updateCanvasDimensions();

  if (showVideoBackground) {
    landmarkCtx.drawImage(previewEl, 0, 0, landmarkCanvas.width, landmarkCanvas.height);
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
  const shouldDetect = lastVideoTime !== previewEl.currentTime;
  if (shouldDetect) {
    lastVideoTime = previewEl.currentTime;
    pipelineState.face =
      faceEnabled && faceLandmarker ? faceLandmarker.detectForVideo(previewEl, startTimeMs) : null;
    pipelineState.hands =
      handEnabled && handLandmarker ? handLandmarker.detectForVideo(previewEl, startTimeMs) : null;
    if (poseEnabled && poseLandmarker) {
      const poseResult = poseLandmarker.detectForVideo(previewEl, startTimeMs);
      pipelineState.pose = hasVisiblePoseLandmarks(poseResult) ? poseResult : null;
    } else {
      pipelineState.pose = null;
    }
    pipelineState.objects =
      objectEnabled && objectDetector ? objectDetector.detectForVideo(previewEl, startTimeMs) : null;
    pipelineState.gestures =
      gestureEnabled && gestureRecognizer
        ? gestureRecognizer.recognizeForVideo(previewEl, Date.now())
        : null;
    pipelineState.faceDetections =
      faceDetectionEnabled && faceDetector
        ? faceDetector.detectForVideo(previewEl, startTimeMs)
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

const initFaceLandmarker = async () => {
  if (!landmarkCtx) return;
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      outputFaceBlendshapes: true,
      runningMode,
      numFaces: 1
    });
    drawingUtils = new DrawingUtils(landmarkCtx);
    await faceLandmarker.setOptions({ runningMode: 'VIDEO' });
    runningMode = 'VIDEO';
    if (!faceLoopStarted) {
      faceLoopStarted = true;
      analyzeFaceFrame();
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
    handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode,
      numHands: 2
    });
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
    poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU'
      },
      runningMode,
      numPoses: 2
    });
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
    objectDetector = await ObjectDetector.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
        delegate: 'GPU'
      },
      runningMode,
      scoreThreshold: 0.5
    });
    await objectDetector.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Object detector failed to load', error);
  }
};

initObjectDetector();

const initFaceDetector = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );
    faceDetector = await FaceDetector.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
        delegate: 'GPU'
      },
      runningMode
    });
    await faceDetector.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Face detector failed to load', error);
  }
};

initFaceDetector();

const initGestureRecognizer = async () => {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
        delegate: 'GPU'
      },
      runningMode
    });
    await gestureRecognizer.setOptions({ runningMode: 'VIDEO' });
  } catch (error) {
    console.error('Gesture recognizer failed to load', error);
  }
};

initGestureRecognizer();

const updatePromptVisibility = () => {
  if (!promptField || !togglePromptBtn) return;
  if (promptVisible) {
    promptField.hidden = false;
    if (!promptField.value) {
      promptField.value = DEFAULT_PROMPT;
    }
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

updatePromptVisibility();

toggleVideoBg?.addEventListener('change', (event) => {
  showVideoBackground = !event.target.checked;
  // Show/hide background image button (hidden when "don't show" is checked)
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

// Initialize: hide background button when "don't show" is checked
if (toggleVideoBg && backgroundImageBtn) {
  showVideoBackground = !toggleVideoBg.checked;
  backgroundImageBtn.style.display = toggleVideoBg.checked ? 'none' : 'inline-flex';
}

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

showAnalyticsBtn?.addEventListener('click', () => {
  if (!outputsPanel) return;
  const isHidden = outputsPanel.hidden;

  if (isHidden) {
    // --- Open panel ---
    outputsPanel.hidden = false;
    if (submitBtn) submitBtn.style.display = 'inline-flex';
    enableFaceLandmarks();
    // Default to Emotions AI tab
    if (tabData) tabData.classList.add('active');
    if (tabAi) { tabAi.classList.remove('active'); tabAi.hidden = true; }
    if (viewData) viewData.hidden = false;
    if (viewAi) viewAi.hidden = true;
    showAnalyticsBtn.textContent = 'Hide Analytics';
    document.querySelector('.workspace')?.classList.add('analytics-visible');
    setTimeout(() => outputsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
  } else {
    // --- Close panel ---
    outputsPanel.hidden = true;
    if (submitBtn) submitBtn.style.display = 'none';
    if (tabAi) tabAi.hidden = true;
    showAnalyticsBtn.textContent = 'View Analytics';
    document.querySelector('.workspace')?.classList.remove('analytics-visible');
  }
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

const showBlobInPreview = (blob, statusMessage) => {
  if (!blob || !previewEl) return;

  // Reset any previous stream and object URLs
  revokePreviewUrl();
  previewEl.srcObject = null;

  // Prepare element for muted/inline autoplay before setting src
  previewEl.muted = true;
  previewEl.playsInline = true;
  previewEl.autoplay = true;
  previewEl.controls = true;
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
  };
  previewEl.addEventListener('loadeddata', onLoaded);
  previewEl.addEventListener('error', onError);

  // Create and set fresh object URL
  previewObjectUrl = URL.createObjectURL(blob);
  applySrc(previewObjectUrl);

  // On metadata ready, attempt playback (helps when initial play() is blocked)
  const tryPlay = () => {
    previewEl.play?.().catch(() => {});
    previewEl.removeEventListener('loadedmetadata', tryPlay);
  };
  previewEl.addEventListener('loadedmetadata', tryPlay);

  // On canplay, attempt playback again (some browsers need this)
  const tryPlayCanPlay = () => {
    previewEl.play?.().catch(() => {});
    previewEl.removeEventListener('canplay', tryPlayCanPlay);
  };
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
  const hasVideo = previewEl && (previewEl.src || previewEl.srcObject);
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
  previewEl.controls = false;
  revokePreviewUrl();
  updatePlaceholderVisibility();
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
    uploadButtonLabel.textContent = file ? 'Change video' : 'Select video';
  }
  if (!file) {
    clearPreview();
    handlePreviewChange();
    if (playersPanel) playersPanel.hidden = true;
    return;
  }

  if (playersPanel) playersPanel.hidden = false;
  showBlobInPreview(file, 'Uploaded clip ready');
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

// ── Split-screen / webcam pipeline ────────────────────────────────────────────

const splitScreen = document.getElementById('split-screen');
const webcamVideo = document.getElementById('webcam-stream');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamCtx = webcamCanvas?.getContext('2d');
const webcamEmotionCanvas = document.getElementById('webcam-emotion-canvas');
const webcamEmotionCtx = webcamEmotionCanvas?.getContext('2d');
const webcamEmotionName = document.getElementById('webcam-emotion-name');
const typingTextEl = document.getElementById('typing-text');
const webcamSection = document.getElementById('webcam-section');
const rightPanel = document.getElementById('right-panel');
const workspaceRoot = document.getElementById('workspace-root');
const ekmanLegendEl = document.getElementById('ekman-legend');

// App states: 'initial' | 'webcam' | 'workspace'
let appState = 'initial';
let webcamFaceLandmarker = null;
let webcamRunning = false;

// Emotion tracking state (used by renderWebcamCircumplex)
let liveTargetValence = 0;
let liveTargetArousal = 0;
let liveValence = 0;
let liveArousal = 0;
const emotionTrail = [];
const MAX_TRAIL_LENGTH = 30;
const LIVE_LERP = 0.08;
let liveTrailTimer = 0;

function pushEmotionFrame(coords) {
  liveTargetValence = coords.valence;
  liveTargetArousal = coords.arousal;
}

// Build Ekman legend
function buildEkmanLegend() {
  if (!ekmanLegendEl) return;
  ekmanLegendEl.innerHTML = '';
  EKMAN_EMOTIONS.forEach(e => {
    const li = document.createElement('li');
    li.textContent = e.label;
    li.dataset.emotion = e.label;
    ekmanLegendEl.appendChild(li);
  });
}

function updateEkmanLegend() {
  if (!ekmanLegendEl) return;
  const dominant = getDominantEmotion(liveValence, liveArousal);
  ekmanLegendEl.querySelectorAll('li').forEach(li => {
    li.classList.toggle('active', li.dataset.emotion === dominant.label);
  });
}

buildEkmanLegend();

// Typing effect
async function typeText(element, messages, charDelay = 60, pauseDelay = 1200) {
  if (!element) return;
  element.hidden = false;
  element.classList.remove('done');
  for (let i = 0; i < messages.length; i++) {
    element.textContent = '';
    for (const char of messages[i]) {
      element.textContent += char;
      await new Promise(r => setTimeout(r, charDelay));
    }
    if (i < messages.length - 1) {
      await new Promise(r => setTimeout(r, pauseDelay));
    }
  }
  element.classList.add('done');
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    // If state changed while awaiting camera permission, release immediately
    if (appState !== 'webcam') {
      stream.getTracks().forEach(t => t.stop());
      return;
    }
    if (webcamVideo) {
      webcamVideo.srcObject = stream;
      webcamVideo.play();
    }
    await initWebcamPipeline();
  } catch (err) {
    console.error('Webcam error:', err);
  }
}

async function initWebcamPipeline() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
  );
  webcamFaceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1
  });
  if (webcamEmotionCanvas) {
    webcamEmotionCanvas.width = 420;
    webcamEmotionCanvas.height = 420;
  }
  webcamRunning = true;
  requestAnimationFrame(analyzeWebcamFrame);
}

function analyzeWebcamFrame(timestamp) {
  if (!webcamRunning || !webcamVideo || webcamVideo.paused || webcamVideo.ended) {
    if (webcamRunning) requestAnimationFrame(analyzeWebcamFrame);
    return;
  }

  // Draw face dots on webcam canvas
  if (webcamCanvas && webcamCtx && webcamVideo.videoWidth) {
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
  }

  if (webcamFaceLandmarker) {
    const results = webcamFaceLandmarker.detectForVideo(webcamVideo, timestamp);

    // Draw face dots
    if (results.faceLandmarks && webcamCtx) {
      results.faceLandmarks.forEach(landmarks => {
        landmarks.forEach(point => {
          const x = point.x * webcamCanvas.width;
          const y = point.y * webcamCanvas.height;
          webcamCtx.fillStyle = 'rgba(255,255,255,0.85)';
          webcamCtx.fillRect(Math.round(x), Math.round(y), 1, 1);
        });
      });
    }

    // Update emotion targets from blendshapes
    if (results.faceBlendshapes?.length) {
      const categories = results.faceBlendshapes[0].categories || [];
      const coords = computeEmotionCoordinates(categories);
      if (coords) pushEmotionFrame(coords);
    }
  }

  requestAnimationFrame(analyzeWebcamFrame);
}

function renderWebcamCircumplex(timestamp) {
  if (!webcamEmotionCanvas || !webcamEmotionCtx) return;

  // Lerp towards targets
  liveValence += (liveTargetValence - liveValence) * LIVE_LERP;
  liveArousal += (liveTargetArousal - liveArousal) * LIVE_LERP;

  // Trail at reduced rate
  liveTrailTimer++;
  if (liveTrailTimer >= 5) {
    liveTrailTimer = 0;
    emotionTrail.push({ valence: liveValence, arousal: liveArousal });
    if (emotionTrail.length > MAX_TRAIL_LENGTH) emotionTrail.shift();
    updateEkmanLegend();
  }

  const ctx = webcamEmotionCtx;
  const size = webcamEmotionCanvas.width;
  const center = size / 2;
  const radius = center - 24;

  ctx.clearRect(0, 0, size, size);

  // Background
  ctx.beginPath();
  ctx.arc(center, center, radius + 20, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fill();

  // Quadrant tints
  const quadrants = [
    { startAngle: -Math.PI / 2, color: 'rgba(100,200,100,0.04)' },
    { startAngle: 0, color: 'rgba(100,100,200,0.04)' },
    { startAngle: Math.PI / 2, color: 'rgba(200,100,100,0.04)' },
    { startAngle: Math.PI, color: 'rgba(200,200,100,0.04)' },
  ];
  quadrants.forEach(({ startAngle, color }) => {
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.arc(center, center, radius, startAngle, startAngle + Math.PI / 2);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  });

  // Crosshair
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.moveTo(center - radius, center);
  ctx.lineTo(center + radius, center);
  ctx.moveTo(center, center - radius);
  ctx.lineTo(center, center + radius);
  ctx.stroke();
  ctx.setLineDash([]);

  // Dashed ring
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.arc(center, center, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Ekman markers
  const dominant = getDominantEmotion(liveValence, liveArousal);
  EKMAN_EMOTIONS.forEach(e => {
    const ex = center + e.v * radius;
    const ey = center - e.a * radius;
    const isDominant = e.label === dominant.label;

    if (isDominant) {
      ctx.beginPath();
      ctx.arc(ex, ey, 14, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(ex, ey, isDominant ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isDominant ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.3)';
    ctx.fill();
  });

  // Trail
  emotionTrail.forEach((point, i) => {
    const opacity = ((i + 1) / emotionTrail.length) * 0.5;
    const px = center + point.valence * radius;
    const py = center - point.arousal * radius;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${opacity})`;
    ctx.fill();
  });

  // Pulsing pointer
  const pulse = 0.5 + 0.5 * Math.sin(timestamp / 400);
  const px = center + liveValence * radius;
  const py = center - liveArousal * radius;
  ctx.beginPath();
  ctx.arc(px, py, 8 + pulse * 4, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(255,255,255,${0.15 + pulse * 0.15})`;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(px, py, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();

  if (webcamEmotionName) webcamEmotionName.textContent = dominant.label;
}

// Continuous animation loop for smooth webcam circumplex
function animateWebcamCircumplex(timestamp) {
  if (webcamRunning) renderWebcamCircumplex(timestamp);
  requestAnimationFrame(animateWebcamCircumplex);
}
requestAnimationFrame(animateWebcamCircumplex);

function stopWebcam() {
  const stream = webcamVideo?.srcObject;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    webcamVideo.srcObject = null;
  }
  webcamRunning = false;
}

function showWorkspace() {
  stopWebcam();
  if (webcamSection) {
    webcamSection.classList.remove('visible');
    webcamSection.hidden = true;
  }
  // Clear typing text
  if (typingTextEl) { typingTextEl.textContent = ''; typingTextEl.hidden = true; }
  // Hide split-screen entirely, show standalone workspace
  if (splitScreen) splitScreen.hidden = true;
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

function showWebcam() {
  if (workspaceRoot) workspaceRoot.hidden = true;
  form.classList.add('hidden');
  // Show split-screen with webcam
  if (splitScreen) splitScreen.hidden = false;
  if (webcamSection) {
    webcamSection.hidden = false;
    requestAnimationFrame(() => webcamSection.classList.add('visible'));
  }
  startWebcam();
  appState = 'webcam';
}

document.addEventListener('bust-click', async () => {
  if (appState === 'initial') {
    // First click: split screen, animate bust to left
    splitScreen?.classList.add('activated');
    window.bust3d?.activate();
    appState = 'webcam';

    // Wait for split animation to complete (matches 0.8s CSS transition)
    await new Promise(r => setTimeout(r, 900));

    await typeText(typingTextEl, [
      'Hi...',
      'Let me reveal how I see your emotions right now...',
      'Look at the camera...'
    ]);

    // Show webcam feed (face landmarks)
    if (webcamSection) {
      webcamSection.hidden = false;
      requestAnimationFrame(() => webcamSection.classList.add('visible'));
    }
    startWebcam();

    // Fade in diagram after 1s delay
    const webcamDiagram = document.getElementById('webcam-diagram');
    setTimeout(() => {
      webcamDiagram?.classList.add('diagram-visible');
    }, 1000);

    // After 15s, hint text
    setTimeout(() => {
      if (appState === 'webcam' && typingTextEl) {
        typingTextEl.classList.remove('done');
        typeText(typingTextEl, ['To reveal even more click on me...']).then(() => {
          typingTextEl.classList.add('done');
        });
      }
    }, 15000);

  } else if (appState === 'webcam') {
    showWorkspace();

  } else if (appState === 'workspace') {
    showWebcam();
  }
});


form.addEventListener('submit', (e) => e.preventDefault());

const aiControls = document.getElementById('ai-controls');
const sendAnalysisBtn = document.getElementById('send-analysis-btn');
let trueFalseEnabled = false;
const TRUE_FALSE_PROMPT = `You are an expert in nonverbal communication, deception detection and behavioral analysis.

Analyze this video and determine whether the person is likely telling the **truth** or **lying/being deceptive**.

Base your analysis ONLY on observable nonverbal cues. Do NOT guess from context or content.

---

VERDICT

State one of: **LIKELY TRUTHFUL** or **LIKELY DECEPTIVE** or **INCONCLUSIVE**

CONFIDENCE: Give a percentage (0-100%) of how confident you are.

---

1. Deception Indicators Observed

* List each specific nonverbal signal you observed that suggests truth or deception.
* For each signal, note the timestamp or moment if possible.
* Use **bold** for the signal name.

---

2. Truthful Indicators Observed

* List each specific nonverbal signal that supports truthfulness.
* Use **bold** for the signal name.

---

3. Baseline Behavior

* Describe the person's baseline demeanor (calm, nervous, animated, etc.).
* Note any shifts from baseline that may indicate deception.

---

4. Summary

* 3-5 bullet points summarizing your analysis.
* Final verdict with reasoning.

---

Formatting Rules:

* Use numbered sections (1-4) and subsections as headings.
* Use * for bullet points, each on its own line.
* Use **bold** for key terms and signals.
* Separate major sections with --- on its own line.
* Keep blank lines between sections.
* The VERY FIRST line must be the verdict: "LIKELY TRUTHFUL" or "LIKELY DECEPTIVE" or "INCONCLUSIVE".
* The SECOND line must be: "CONFIDENCE: XX%"
* Be concise: each bullet max 1-2 short sentences.
* Do NOT invent details. If something cannot be assessed, write: "Not enough visual data."`;

const toggleTrueFalse = document.getElementById('toggle-true-false');

toggleTrueFalse?.addEventListener('change', (event) => {
  trueFalseEnabled = Boolean(event.target.checked);
});

const renderVerdictCard = (text) => {
  if (!verdictCard) return;
  const verdictLabel = document.getElementById('verdict-label');
  const verdictConfidence = document.getElementById('verdict-confidence');
  const verdictSignals = document.getElementById('verdict-signals');

  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

  // Parse verdict from first meaningful line
  let verdict = 'INCONCLUSIVE';
  let confidence = '';
  for (const line of lines) {
    const upper = line.replace(/\*\*/g, '').toUpperCase();
    if (upper.includes('LIKELY TRUTHFUL')) { verdict = 'LIKELY TRUTHFUL'; }
    else if (upper.includes('LIKELY DECEPTIVE')) { verdict = 'LIKELY DECEPTIVE'; }
    else if (upper.includes('INCONCLUSIVE')) { verdict = 'INCONCLUSIVE'; }
    const confMatch = line.match(/CONFIDENCE[:\s]*(\d+%?)/i);
    if (confMatch) confidence = confMatch[1].includes('%') ? confMatch[1] : confMatch[1] + '%';
    if (verdict !== 'INCONCLUSIVE' || confidence) break;
  }

  if (verdictLabel) {
    verdictLabel.textContent = verdict;
    verdictLabel.className = 'verdict-label';
    if (verdict === 'LIKELY TRUTHFUL') verdictLabel.classList.add('verdict-truth');
    else if (verdict === 'LIKELY DECEPTIVE') verdictLabel.classList.add('verdict-lie');
    else verdictLabel.classList.add('verdict-inconclusive');
  }
  if (verdictConfidence) {
    verdictConfidence.textContent = confidence ? `Confidence: ${confidence}` : '';
  }

  // Collect deception/truth indicator bullets
  if (verdictSignals) {
    verdictSignals.innerHTML = '';
    let collecting = false;
    for (const line of lines) {
      if (/^\d+\.\s*(Deception|Truthful)\s*Indicators/i.test(line)) { collecting = true; continue; }
      if (/^\d+\.\s/.test(line) && collecting) { collecting = false; }
      if (collecting && /^[-*]\s+/.test(line)) {
        const li = document.createElement('li');
        li.innerHTML = line.replace(/^[-*]\s+/, '').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        verdictSignals.appendChild(li);
      }
    }
  }

  verdictCard.hidden = false;
};

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
    setStatus('Please choose a video first.', 'error');
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
  setStatus('Uploading video and contacting AI…', 'info');
  if (sendAnalysisBtn) sendAnalysisBtn.disabled = true;

  const formData = new FormData();
  formData.append('video', form.video.files[0]);
  let promptValue = '';
  if (trueFalseEnabled) {
    promptValue = TRUE_FALSE_PROMPT;
  } else {
    promptValue = promptField?.value?.trim() || '';
  }
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
    if (trueFalseEnabled) {
      renderVerdictCard(payload.resultText);
    } else if (verdictCard) {
      verdictCard.hidden = true;
    }
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
