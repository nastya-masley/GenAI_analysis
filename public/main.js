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

  // White skeleton globe (wireframe only)
  const globeGeom = new THREE.SphereGeometry(1, 32, 24);
  const globeMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    wireframe: true,
    transparent: true,
    opacity: 0.85
  });
  globeMesh = new THREE.Mesh(globeGeom, globeMat);
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
  globeCanvas.addEventListener('mousedown', (e) => {
    dragging = true;
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', (e) => {
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

  animateGlobe();
}

function onGlobeResize() {
  globeRenderer.setSize(window.innerWidth, window.innerHeight);
  globeCamera.aspect = window.innerWidth / window.innerHeight;
  globeCamera.updateProjectionMatrix();
}

function animateGlobe() {
  requestAnimationFrame(animateGlobe);
  currentRotX += (targetRotX - currentRotX) * 0.05;
  currentRotY += (targetRotY - currentRotY) * 0.05;
  globeGroup.rotation.x = currentRotX;
  globeGroup.rotation.y = currentRotY;
  globeGroup.rotation.y += 0.0003;
  globeRenderer.render(globeScene, globeCamera);
}

// Start loading animation, then transition to main
animateLoading();
