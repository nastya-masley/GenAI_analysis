import * as THREE from 'three';

const SCAN_PAGE = '/globe.html';
const STORAGE_KEY = 'lidar_scans';
const LOADING_DURATION_MS = 2200;

const loadingCanvas = document.getElementById('loading-canvas');
const globeCanvas = document.getElementById('globe-canvas');

// ── Loading scene (same aesthetic as scan canvas) ─────────────────────────────
const loadRenderer = new THREE.WebGLRenderer({ canvas: loadingCanvas, antialias: true });
loadRenderer.setPixelRatio(Math.min(devicePixelRatio, 3));
loadRenderer.setClearColor(0x000508, 1);
loadRenderer.setSize(window.innerWidth, window.innerHeight);

const loadScene = new THREE.Scene();
const loadCamera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 200);
loadCamera.position.z = 3;

const loadGeo = new THREE.BufferGeometry();
const N = 3000;
const posArr = new Float32Array(N * 3);
const colArr = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  posArr[i * 3] = (Math.random() - 0.5) * 2;
  posArr[i * 3 + 1] = (Math.random() - 0.5) * 2;
  posArr[i * 3 + 2] = (Math.random() - 0.5) * 2;
  colArr[i * 3] = 0.05;
  colArr[i * 3 + 1] = 0.35 + Math.random() * 0.3;
  colArr[i * 3 + 2] = 0.65 + Math.random() * 0.3;
}
loadGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
loadGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
const loadMat = new THREE.PointsMaterial({
  size: 0.013,
  vertexColors: true,
  sizeAttenuation: true,
});
const loadCloud = new THREE.Points(loadGeo, loadMat);
loadScene.add(loadCloud);

function loadingLoop() {
  loadCloud.rotation.y += 0.003;
  loadRenderer.render(loadScene, loadCamera);
}

let loadingStart = performance.now();
function animateLoading() {
  if (document.body.classList.contains('main')) return;
  requestAnimationFrame(animateLoading);
  loadingLoop();
  const elapsed = performance.now() - loadingStart;
  if (elapsed >= LOADING_DURATION_MS) {
    document.body.classList.add('main');
    initGlobe();
  }
}

// ── Main globe scene ─────────────────────────────────────────────────────────
let globeScene, globeCamera, globeRenderer, globeGroup, globeMesh;
let targetRotY = 0, targetRotX = 0, currentRotX = 0, currentRotY = 0;
let dragging = false, prevMouse = { x: 0, y: 0 };

function initGlobe() {
  globeRenderer = new THREE.WebGLRenderer({ canvas: globeCanvas, antialias: true });
  globeRenderer.setPixelRatio(Math.min(devicePixelRatio, 3));
  globeRenderer.setClearColor(0x000508, 1);
  globeRenderer.setSize(window.innerWidth, window.innerHeight);

  globeScene = new THREE.Scene();
  globeCamera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
  globeCamera.position.z = 3.5;

  globeGroup = new THREE.Group();
  globeScene.add(globeGroup);

  const VIDEO_SRC = '/assets/MVI_3101.MP4';
  const VIDEO_COLS = 8;
  const VIDEO_ROWS = 5;
  const TILE_SEG_PHI = 12;
  const TILE_SEG_THETA = 8;
  const CLIP_DURATION = 10; // seconds — each pool video loops a 10 s window
  const N_POOL = 6;         // independent video elements for visual variety
  const videoSegments = [];

  // Pool of videos, each looping a different random 10 s window
  const videoPool = Array.from({ length: N_POOL }, () => {
    const vid = document.createElement('video');
    vid.src = VIDEO_SRC;
    vid.muted = true;
    vid.playsInline = true;
    vid.preload = 'auto';
    let clipStart = 0;
    vid.addEventListener('loadedmetadata', () => {
      clipStart = Math.random() * Math.max(0, vid.duration - CLIP_DURATION);
      vid.currentTime = clipStart;
      vid.play().catch(() => {});
    });
    vid.addEventListener('timeupdate', () => {
      if (vid.currentTime >= clipStart + CLIP_DURATION) vid.currentTime = clipStart;
    });
    vid.load();
    const tex = new THREE.VideoTexture(vid);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.format = THREE.RGBAFormat;
    return { vid, tex };
  });

  const phiStep = (Math.PI * 2) / VIDEO_COLS;
  const thetaStep = Math.PI / VIDEO_ROWS;
  let poolIdx = 0;
  for (let row = 0; row < VIDEO_ROWS; row++) {
    for (let col = 0; col < VIDEO_COLS; col++) {
      const { vid, tex } = videoPool[poolIdx % N_POOL];
      poolIdx++;
      const segmentGeom = new THREE.SphereGeometry(
        1,
        TILE_SEG_PHI, TILE_SEG_THETA,
        col * phiStep, phiStep,
        row * thetaStep, thetaStep
      );
      const segmentMat = new THREE.MeshBasicMaterial({
        map: tex,
        side: THREE.DoubleSide
      });
      const segmentMesh = new THREE.Mesh(segmentGeom, segmentMat);
      segmentMesh.userData = { video: vid };
      videoSegments.push(segmentMesh);
      globeGroup.add(segmentMesh);
    }
  }

  // Wireframe lines matching tile grid for crisp borders
  const globeGeom = new THREE.SphereGeometry(1.001, VIDEO_COLS * TILE_SEG_PHI, VIDEO_ROWS * TILE_SEG_THETA);
  const wireframeMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    wireframe: true,
    transparent: true,
    opacity: 0.25
  });
  const gridGeom = new THREE.SphereGeometry(1.001, VIDEO_COLS, VIDEO_ROWS);
  const gridMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    wireframe: true,
    transparent: true,
    opacity: 0.9
  });
  globeMesh = new THREE.Mesh(gridGeom, gridMat);
  globeGroup.add(globeMesh);

  // Stars
  const starPos = new Float32Array(1500 * 3);
  for (let i = 0; i < 1500; i++) {
    const r = 25 + Math.random() * 40;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    starPos[i * 3] = r * Math.sin(ph) * Math.cos(th);
    starPos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
    starPos[i * 3 + 2] = r * Math.cos(ph);
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  const stars = new THREE.Points(
    starGeo,
    new THREE.PointsMaterial({ color: 0xffffff, size: 0.2, transparent: true, opacity: 0.5 })
  );
  globeScene.add(stars);

  window.addEventListener('resize', onGlobeResize);
  let pointerIsDown = false;
  let pointerDownX = 0, pointerDownY = 0, pointerDownTime = 0;
  globeCanvas.addEventListener('pointerdown', (e) => {
    pointerIsDown = true;
    dragging = false;
    pointerDownX = e.clientX;
    pointerDownY = e.clientY;
    pointerDownTime = Date.now();
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  // pointerup catches releases even outside the canvas / window
  window.addEventListener('pointerup', () => { pointerIsDown = false; dragging = false; });
  window.addEventListener('pointermove', (e) => {
    if (!pointerIsDown) return;
    const moved = Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY);
    if (moved > 4) dragging = true;
    if (!dragging) return;
    targetRotY += (e.clientX - prevMouse.x) * 0.005;
    targetRotX += (e.clientY - prevMouse.y) * 0.005;
    targetRotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, targetRotX));
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  globeCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    // Allow zoom from very close (0.6) to far (10) for a closer look at scans
    globeCamera.position.z = Math.max(0.6, Math.min(10, globeCamera.position.z + e.deltaY * 0.002));
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyS' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      window.location.href = SCAN_PAGE;
    }
  });

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const overlay = document.getElementById('video-overlay');
  const overlayVideo = document.getElementById('video-overlay-video');
  const overlayClose = document.getElementById('video-overlay-close');

  function onGlobeClick(e) {
    // Reject long presses (> 200 ms) or if the pointer moved more than 6 px (drag)
    const held = Date.now() - pointerDownTime;
    const moved = Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY);
    if (held > 200 || moved > 6 || dragging) return;
    const rect = globeCanvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, globeCamera);
    const hits = raycaster.intersectObjects(videoSegments);
    if (hits.length > 0) {
      const mesh = hits[0].object;
      const vid = mesh.userData.video;
      if (vid) {
        sessionStorage.setItem('autoAnalyze', JSON.stringify({ src: VIDEO_SRC, globeSplit: true }));
        window.location.href = '/index.html';
      }
    }
  }

  // Use pointerup (fires before mouseup) so pointerDownTime is still valid when checked
  globeCanvas.addEventListener('pointerup', onGlobeClick);

  function closeOverlay() {
    overlayVideo.pause();
    overlay.classList.remove('visible');
  }
  overlayClose.addEventListener('click', closeOverlay);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeOverlay();
  });

  animateGlobe(videoPool);
}

function onGlobeResize() {
  globeRenderer.setSize(window.innerWidth, window.innerHeight);
  globeCamera.aspect = window.innerWidth / window.innerHeight;
  globeCamera.updateProjectionMatrix();
}

let _globeFrame = 0;
function animateGlobe(videoPool) {
  requestAnimationFrame(() => animateGlobe(videoPool));
  _globeFrame++;
  // Upload each pool texture every other frame to halve GPU upload cost
  if (_globeFrame % 2 === 0) {
    videoPool.forEach(({ vid, tex }) => {
      if (vid.readyState >= 2) tex.needsUpdate = true;
    });
  }
  currentRotX += (targetRotX - currentRotX) * 0.05;
  currentRotY += (targetRotY - currentRotY) * 0.05;
  globeGroup.rotation.x = currentRotX;
  globeGroup.rotation.y = currentRotY;
  globeGroup.rotation.y += 0.0003;
  globeRenderer.render(globeScene, globeCamera);
}

// Start loading animation, then transition to main
animateLoading();
