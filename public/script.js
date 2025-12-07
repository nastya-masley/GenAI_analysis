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
const enableCameraBtn = document.getElementById('enable-camera');
const startRecordingBtn = document.getElementById('start-recording');
const stopRecordingBtn = document.getElementById('stop-recording');
const discardRecordingBtn = document.getElementById('discard-recording');
const recordingStatus = document.getElementById('recording-status');
const previewEl = document.getElementById('preview');
const videoInput = document.getElementById('video');
const uploadButtonLabel = document.getElementById('upload-btn-label');
const selectedFileHint = document.getElementById('selected-file-hint');
const togglePromptBtn = document.getElementById('toggle-prompt');
const promptField = document.getElementById('prompt');
const toggleVideoBg = document.getElementById('toggle-video-bg');
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
const handGestureList = document.getElementById('hand-gesture-list');
const handGestureEmpty = document.getElementById('hand-gesture-empty');
const emotionWheelCanvas = document.getElementById('emotion-wheel-canvas');
const emotionWheelCtx = emotionWheelCanvas?.getContext('2d');
const emotionWheelStatus = document.getElementById('emotion-wheel-status');
const landmarkCtx = landmarkCanvas?.getContext('2d');

let mediaStream;
let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;
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
let showVideoBackground = true;
let faceEnabled = toggleFace ? toggleFace.checked : true;
let handEnabled = toggleHand ? toggleHand.checked : true;
let poseEnabled = togglePose ? togglePose.checked : true;
let objectEnabled = toggleObject ? toggleObject.checked : true;
let gestureEnabled = toggleGesture ? toggleGesture.checked : true;
let faceDetectionEnabled = toggleFaceDetect ? toggleFaceDetect.checked : true;
let faceLoopStarted = false;
let faceRenderMode = faceStyleSelect ? faceStyleSelect.value : 'mesh';
let poseJointsEnabled = togglePoseJoints ? togglePoseJoints.checked : true;
let poseTrailsEnabled = togglePoseTrails ? togglePoseTrails.checked : false;
let emotionWheelEnabled = toggleEmotionWheel ? toggleEmotionWheel.checked : true;

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
const VALENCE_POSITIVE = ['mouthSmileLeft', 'mouthSmileRight', 'cheekPuff'];
const VALENCE_NEGATIVE = ['mouthFrownLeft', 'mouthFrownRight', 'browDownLeft', 'browDownRight'];
const AROUSAL_POSITIVE = ['eyeWideLeft', 'eyeWideRight', 'jawOpen', 'mouthOpen'];
const AROUSAL_NEGATIVE = ['eyeBlinkLeft', 'eyeBlinkRight', 'mouthClose'];

const clamp = (value, min = -1, max = 1) => Math.min(Math.max(value, min), max);

const getBlendshapeScore = (categories = [], targetName) => {
  const match = categories.find(
    (shape) => shape.categoryName === targetName || shape.displayName === targetName
  );
  return match ? Number(match.score) : 0;
};

const computeEmotionCoordinates = (categories = []) => {
  if (!categories.length) return null;
  const avg = (names) =>
    names.reduce((sum, name) => sum + getBlendshapeScore(categories, name), 0) / names.length;

  const valence = clamp(avg(VALENCE_POSITIVE) - avg(VALENCE_NEGATIVE));
  const arousal = clamp(avg(AROUSAL_POSITIVE) - avg(AROUSAL_NEGATIVE));
  return { valence, arousal };
};

const setEmotionWheelMessage = (message) => {
  if (emotionWheelStatus) {
    emotionWheelStatus.textContent = message;
  }
};

const clearEmotionWheel = (message) => {
  if (!emotionWheelCanvas || !emotionWheelCtx) return;
  emotionWheelCtx.clearRect(0, 0, emotionWheelCanvas.width, emotionWheelCanvas.height);
  emotionWheelCtx.fillStyle = '#000';
  emotionWheelCtx.fillRect(0, 0, emotionWheelCanvas.width, emotionWheelCanvas.height);
  setEmotionWheelMessage(message);
};

const renderEmotionWheel = ({ valence, arousal }) => {
  if (!emotionWheelCanvas || !emotionWheelCtx) return;
  const ctx = emotionWheelCtx;
  const size = emotionWheelCanvas.width;
  const center = size / 2;
  const radius = center - 20;

  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);

  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(center, center, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(center - radius, center);
  ctx.lineTo(center + radius, center);
  ctx.moveTo(center, center - radius);
  ctx.lineTo(center, center + radius);
  ctx.stroke();

  const pointerX = center + valence * radius;
  const pointerY = center - arousal * radius;

  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(center, center);
  ctx.lineTo(pointerX, pointerY);
  ctx.stroke();

  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(pointerX, pointerY, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.font = '14px "OCR A Extended", monospace';
  ctx.fillText('Positive', center + radius - 60, center - 6);
  ctx.fillText('Negative', center - radius + 10, center - 6);
  ctx.save();
  ctx.translate(center + 6, center - radius + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('High arousal', 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(center + 6, center + radius - 10);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Low arousal', 0, 0);
  ctx.restore();
};

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
  setEmotionWheelMessage(
    `Valence ${coords.valence.toFixed(2)} · Arousal ${coords.arousal.toFixed(2)}`
  );
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

* Always use this 1-4 structure and subpoints.
* Everything have to be in raw text format. No markdown or html tags.
* No line breaks between bullets.
* No line breaks between sections.
* No line breaks between subpoints.
* No line breaks between paragraphs.
* No line breaks between sentences.
* No line breaks between words.
* No line breaks between characters.
* Be concise: each bullet max 1-2 short sentences.
* Do NOT invent details. If something cannot be seen or judged, write: "Not enough visual data to assess."`;

const setStatus = (message, variant = 'info') => {
  statusEl.textContent = message;
  statusEl.dataset.variant = variant;
  statusEl.hidden = false;
};

const setBlendShapesMessage = (message) => {
  if (!blendShapeList) return;
  blendShapeList.innerHTML = `<li class="blend-shapes-item"><span class="blend-shapes-label">${message}</span></li>`;
};

const updateHandGesturePanel = (entries) => {
  if (!handGestureList || !handGestureEmpty) return;
  if (!handEnabled) {
    handGestureList.innerHTML = '';
    handGestureEmpty.hidden = false;
    handGestureEmpty.textContent = 'Hand landmarks disabled.';
    return;
  }
  if (!gestureEnabled) {
    handGestureList.innerHTML = '';
    handGestureEmpty.hidden = false;
    handGestureEmpty.textContent = 'Gesture recognition disabled.';
    return;
  }
  if (!entries || !entries.length) {
    handGestureList.innerHTML = '';
    handGestureEmpty.hidden = false;
    handGestureEmpty.textContent = 'Detecting hands & gestures…';
    return;
  }

  const items = entries
    .map(
      ({ handLabel, gesture, confidence }) => `
        <li class="hand-gesture-item">
          ${handLabel}: ${gesture}
          <span>Confidence ${(confidence * 100).toFixed(1)}%</span>
        </li>
      `
    )
    .join('');

  handGestureList.innerHTML = items;
  handGestureEmpty.hidden = true;
};

const resetFaceOutputs = () => {
  if (landmarkCtx && landmarkCanvas) {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
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
        landmarkCtx.arc(point.x * width, point.y * height, 2.5, 0, Math.PI * 2);
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
        landmarkCtx.fillStyle = 'rgba(255, 255, 255, 0.92)';
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
  if (!poseTrailsEnabled) {
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
  if (!poseTrailsEnabled || !poseTrails.size) return;
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
  updatePoseTrailHistory(poses);
  landmarkCtx.save();
  poses.forEach((landmarks) => {
    drawingUtils.drawConnectors(landmarks, POSE_CONNECTIONS, {
      color: '#FFFFFF',
      lineWidth: 3
    });
    if (poseJointsEnabled) {
      drawingUtils.drawLandmarks(landmarks, {
        color: '#FFFFFF',
        radius: 3
      });
    }
    drawTorsoOverlay(landmarks, width, height);
  });
  if (poseTrailsEnabled) {
    drawPoseTrailsOverlay(width, height);
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
    landmarkCtx.lineWidth = 3;
    landmarkCtx.strokeRect(originX, originY, width, height);
    const label = detection.categories?.[0];
    if (label) {
      const text = `${label.categoryName || 'Object'} ${(label.score * 100).toFixed(1)}%`;
      landmarkCtx.font = '16px "OCR A Extended", monospace';
      const textWidth = landmarkCtx.measureText(text).width;
      const labelHeight = 30;
      const padding = 10;
      const boxWidth = textWidth + padding * 2;
      const boxX = originX;
      const boxY = Math.max(originY - labelHeight - 4, 0);
      landmarkCtx.fillStyle = 'rgba(255, 255, 255, 0.95)';
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
    landmarkCtx.lineWidth = 3;
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
    const boxX = originX;
    const boxY = Math.max(originY - labelHeight - 4, 0);
    landmarkCtx.fillStyle = 'rgba(255, 255, 255, 0.95)';
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
  showVideoBackground = Boolean(event.target.checked);
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

togglePoseTrails?.addEventListener('change', (event) => {
  poseTrailsEnabled = Boolean(event.target.checked);
  if (!poseTrailsEnabled) {
    poseTrails.clear();
  }
});

toggleEmotionWheel?.addEventListener('change', (event) => {
  emotionWheelEnabled = Boolean(event.target.checked);
  if (!emotionWheelEnabled) {
    clearEmotionWheel('Emotion wheel disabled.');
  } else {
    clearEmotionWheel('Emotion wheel enabled. Waiting for blendshapes…');
  }
});

const setRecordingStatus = (message) => {
  if (recordingStatus) {
    recordingStatus.textContent = message;
  }
};

const resetRecording = (statusMessage = 'Camera idle') => {
  recordedBlob = null;
  recordedChunks = [];
  discardRecordingBtn.disabled = true;
  if (statusMessage && recordingStatus) {
    setRecordingStatus(statusMessage);
  }
};

const revokePreviewUrl = () => {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
};

const showBlobInPreview = (blob, statusMessage) => {
  if (!blob || !previewEl) return;
  revokePreviewUrl();
  previewEl.srcObject = null;
  previewObjectUrl = URL.createObjectURL(blob);
  previewEl.src = previewObjectUrl;
  previewEl.controls = true;
  previewEl.muted = false;
  previewEl.play?.().catch(() => {});
  handlePreviewChange();
  if (statusMessage) {
    setRecordingStatus(statusMessage);
  }
};

const clearPreview = () => {
  if (!previewEl) return;
  previewEl.pause?.();
  previewEl.removeAttribute('src');
  previewEl.srcObject = null;
  previewEl.controls = false;
  revokePreviewUrl();
};

const stopTracks = () => {
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;
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
    if (!recordedBlob && !mediaStream) {
      clearPreview();
      setRecordingStatus('Camera idle');
      handlePreviewChange();
    }
    return;
  }

  stopTracks();
  resetRecording(null);
  showBlobInPreview(file, 'Uploaded clip ready');
};

const enableCamera = async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    clearPreview();
    previewEl.srcObject = mediaStream;
    previewEl.controls = false;
    previewEl.muted = true;
    previewEl.play?.().catch(() => {});
    startRecordingBtn.disabled = false;
    setRecordingStatus('Camera ready');
    handlePreviewChange();
  } catch (error) {
    console.error(error);
    setStatus('Unable to access camera. Check browser permissions.', 'error');
  }
};

const startRecording = () => {
  if (!mediaStream) {
    setStatus('Enable the webcam before recording.', 'error');
    return;
  }

  recordedChunks = [];
  recordedBlob = null;
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm;codecs=vp9,opus' });

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
    discardRecordingBtn.disabled = false;
    showBlobInPreview(recordedBlob, 'Recorded clip ready');
  };

  previewEl.controls = false;
  previewEl.srcObject = mediaStream;
  mediaRecorder.start();
  startRecordingBtn.disabled = true;
  stopRecordingBtn.disabled = false;
  setRecordingStatus('Recording…');
};

const stopRecording = () => {
  if (mediaRecorder?.state === 'recording') {
    mediaRecorder.stop();
    stopRecordingBtn.disabled = true;
    startRecordingBtn.disabled = false;
    stopTracks();
  }
};

const discardRecording = () => {
  clearPreview();
  resetRecording();
  startRecordingBtn.disabled = !mediaStream;
  if (videoInput?.files?.length) {
    handleVideoSelection();
  } else {
    handlePreviewChange();
  }
};

enableCameraBtn?.addEventListener('click', enableCamera);
startRecordingBtn?.addEventListener('click', startRecording);
stopRecordingBtn?.addEventListener('click', stopRecording);
discardRecordingBtn?.addEventListener('click', discardRecording);
videoInput?.addEventListener('change', () => {
  handleVideoSelection();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!form.video.files.length && !recordedBlob) {
    setStatus('Please choose or record a video first.', 'error');
    return;
  }

  resultSection.hidden = true;
  setStatus('Uploading video and contacting AI…', 'info');
  submitBtn.disabled = true;

  const formData = new FormData();
  if (recordedBlob) {
    formData.append('video', recordedBlob, 'webcam-recording.webm');
  } else {
    formData.append('video', form.video.files[0]);
  }
  const promptValue =
    promptVisible && promptField?.value?.trim() ? promptField.value.trim() : '';
  formData.append('prompt', promptValue);

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    const payload = await response.json();
    if (!response.ok) {
      const message = payload?.error || 'AI request failed.';
      throw new Error(message);
    }

    resultText.textContent = payload.resultText;
    resultSection.hidden = false;
    setStatus('AI response ready.', 'success');
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Unexpected error occurred.', 'error');
  } finally {
    submitBtn.disabled = false;
  }
});
