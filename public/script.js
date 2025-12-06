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
const playersPanel = document.querySelector('.players-panel');
const landmarkCanvas = document.getElementById('landmark-canvas');
const blendShapeList = document.getElementById('blend-shape-list');
const gestureOutput = document.getElementById('gesture-output');
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
let lastFaceResult = null;
let lastHandResult = null;
let lastPoseResult = null;
let lastObjectResult = null;
let lastGestureResult = null;
let lastFaceDetectionResult = null;
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

const resetFaceOutputs = () => {
  if (landmarkCtx && landmarkCanvas) {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  }
  setBlendShapesMessage('Waiting for MediaPipe data…');
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
  if (!gestureEnabled) {
    updateGestureOutput('', false);
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
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: '#C0C0C070',
      lineWidth: 1
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
      color: '#FF3030'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {
      color: '#FF3030'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
      color: '#30FF30'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {
      color: '#30FF30'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
      color: '#E0E0E0'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: '#E0E0E0'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, {
      color: '#FF3030'
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, {
      color: '#30FF30'
    });
  });
  landmarkCtx.restore();
};

const drawHandLandmarks = (result) => {
  if (!landmarkCtx || !result) return;
  const hands = result.landmarks || [];
  const width = landmarkCanvas.width || 1;
  const height = landmarkCanvas.height || 1;

  landmarkCtx.save();
  landmarkCtx.lineCap = 'round';
  landmarkCtx.lineJoin = 'round';
  hands.forEach((landmarks) => {
    HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
      const start = landmarks[startIdx];
      const end = landmarks[endIdx];
      if (!start || !end) return;
      landmarkCtx.beginPath();
      landmarkCtx.moveTo(start.x * width, start.y * height);
      landmarkCtx.lineTo(end.x * width, end.y * height);
      landmarkCtx.strokeStyle = '#39ff14';
      landmarkCtx.lineWidth = 4;
      landmarkCtx.stroke();
    });

    landmarks.forEach((point) => {
      landmarkCtx.beginPath();
      landmarkCtx.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
      landmarkCtx.fillStyle = '#ff1a1a';
      landmarkCtx.fill();
    });
  });
  landmarkCtx.restore();
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

const drawPoseLandmarks = (result) => {
  if (!landmarkCtx || !drawingUtils || !result || !hasVisiblePoseLandmarks(result)) return;
  const poses = result.landmarks || [];
  landmarkCtx.save();
  poses.forEach((landmarks) => {
    drawingUtils.drawConnectors(landmarks, POSE_CONNECTIONS, {
      color: '#22c55e',
      lineWidth: 3
    });
    drawingUtils.drawLandmarks(landmarks, {
      color: '#15803d',
      radius: 3
    });
  });
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
    landmarkCtx.strokeStyle = '#f97316';
    landmarkCtx.lineWidth = 3;
    landmarkCtx.strokeRect(originX, originY, width, height);
    const label = detection.categories?.[0];
    if (label) {
      const text = `${label.categoryName || 'Object'} ${(label.score * 100).toFixed(1)}%`;
      const textWidth = landmarkCtx.measureText(text).width;
      landmarkCtx.fillStyle = '#f97316';
      landmarkCtx.fillRect(originX, Math.max(originY - 22, 0), textWidth + 10, 22);
      landmarkCtx.fillStyle = '#fff';
      landmarkCtx.font = '14px Inter, system-ui';
      landmarkCtx.fillText(text, originX + 5, Math.max(originY - 6, 13));
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
    landmarkCtx.strokeStyle = '#facc15';
    landmarkCtx.lineWidth = 2;
    landmarkCtx.strokeRect(originX, originY, width, height);
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

const updateGestureOutput = (text, show = true) => {
  if (!gestureOutput) return;
  if (!show || !text) {
    gestureOutput.hidden = true;
    gestureOutput.style.display = 'none';
    gestureOutput.textContent = '';
    return;
  }
  gestureOutput.hidden = false;
  gestureOutput.style.display = 'block';
  gestureOutput.textContent = text;
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
    updateGestureOutput('', false);
    return;
  }

  const startTimeMs = performance.now();
  const shouldDetect = lastVideoTime !== previewEl.currentTime;
  if (shouldDetect) {
    lastVideoTime = previewEl.currentTime;
    if (faceEnabled && faceLandmarker) {
      lastFaceResult = faceLandmarker.detectForVideo(previewEl, startTimeMs);
    }
    if (handEnabled && handLandmarker) {
      lastHandResult = handLandmarker.detectForVideo(previewEl, startTimeMs);
    }
        if (poseEnabled && poseLandmarker) {
          const poseResult = poseLandmarker.detectForVideo(previewEl, startTimeMs);
          lastPoseResult = hasVisiblePoseLandmarks(poseResult) ? poseResult : null;
        }
    if (objectEnabled && objectDetector) {
      lastObjectResult = objectDetector.detectForVideo(previewEl, startTimeMs);
    }
    if (gestureEnabled && gestureRecognizer) {
      lastGestureResult = gestureRecognizer.recognizeForVideo(previewEl, Date.now());
    }
    if (faceDetectionEnabled && faceDetector) {
      lastFaceDetectionResult = faceDetector.detectForVideo(previewEl, startTimeMs);
    }
  }

  if (faceEnabled && lastFaceResult) {
    drawFaceLandmarks(lastFaceResult);
    drawBlendShapesList(lastFaceResult.faceBlendshapes || []);
  } else if (faceEnabled) {
    setBlendShapesMessage('Detecting face landmarks…');
  } else {
    setBlendShapesMessage('Face landmarks disabled.');
  }

  if (handEnabled && lastHandResult) {
    drawHandLandmarks(lastHandResult);
  }

  if (poseEnabled && lastPoseResult) {
    drawPoseLandmarks(lastPoseResult);
  }

  if (objectEnabled && lastObjectResult) {
    drawObjectDetections(lastObjectResult);
  }

  if (gestureEnabled && lastGestureResult?.gestures?.length) {
    const bestGesture = lastGestureResult.gestures[0][0];
    const handedness = lastGestureResult.handednesses?.[0]?.[0]?.displayName || '';
    const text = `Gesture: ${bestGesture.categoryName} (${(bestGesture.score * 100).toFixed(
      1
    )}%)${handedness ? `\nHand: ${handedness}` : ''}`;
    updateGestureOutput(text, true);
  } else if (gestureEnabled) {
    updateGestureOutput('Detecting gestures…', true);
  } else {
    updateGestureOutput('', false);
  }

  if (faceDetectionEnabled && lastFaceDetectionResult) {
    drawFaceDetections(lastFaceDetectionResult);
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
updateGestureOutput('', false);

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
    lastObjectResult = null;
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
    updateGestureOutput('', false);
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
    lastFaceDetectionResult = null;
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
  setStatus('Uploading video and contacting Nonverbal AI.COM…', 'info');
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
      const message = payload?.error || 'Nonverbal AI.COM request failed.';
      throw new Error(message);
    }

    resultText.textContent = payload.resultText;
    resultSection.hidden = false;
    setStatus('Nonverbal AI.COM response ready.', 'success');
  } catch (error) {
    console.error(error);
    setStatus(error.message || 'Unexpected error occurred.', 'error');
  } finally {
    submitBtn.disabled = false;
  }
});
