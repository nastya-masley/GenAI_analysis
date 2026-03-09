import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MODEL_URL = '/assets/rectangle_objects_bust.obj';
const MTL_URL   = '/assets/rectangle_objects_bust.mtl';

// ── Per-polygon video assignment ───────────────────────────────────────────────
// Key = exact mesh name shown in the on-screen label when you click a polygon.
// Value = video URL to play in the overlay when that polygon is clicked.
// Polygons not listed here use the default video below.
const DEFAULT_VIDEO = '/assets/MVI_3101.MP4';
const VIDEO_MAP = {};

const TILE_VIDEOS = [
  '/assets/vid/MVI_3250.MP4',
  '/assets/vid/MVI_3251.MP4',
  '/assets/vid/MVI_3252.MP4',
  '/assets/vid/MVI_3253.MP4',
];

const canvas = document.getElementById('app-canvas');
const loadingOverlay = document.getElementById('loading-overlay');
const progressBar = document.getElementById('progress-bar');
const hint = document.getElementById('hint');

// ── Renderer ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000508, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();

// Stars
const starCount = 2000;
const starPos = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {
  const r = 60 + Math.random() * 80;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  starPos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
  starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
  starPos[i * 3 + 2] = r * Math.cos(phi);
}
const starGeo = new THREE.BufferGeometry();
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
const stars = new THREE.Points(
  starGeo,
  new THREE.PointsMaterial({ color: 0xffffff, size: 0.25, transparent: true, opacity: 0.55 })
);
scene.add(stars);

// ── Camera ────────────────────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 500);
camera.position.set(0, 0, 5);

// ── Lights ────────────────────────────────────────────────────────────────────
const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0x88ccff, 2.5);
keyLight.position.set(3, 5, 4);
keyLight.castShadow = true;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xffd0a0, 0.8);
fillLight.position.set(-4, 2, -3);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0x44aaff, 1.2);
rimLight.position.set(0, -3, -5);
scene.add(rimLight);

// ── Controls ──────────────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 0.5;
controls.maxDistance = 50;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;

// ── Video pool — each entry starts at a random timeline offset ─────────────────
const POOL_SIZE = 12;
const videoPool = [];

for (let i = 0; i < POOL_SIZE; i++) {
  const v = document.createElement('video');
  v.src = TILE_VIDEOS[Math.floor(Math.random() * TILE_VIDEOS.length)];
  v.loop = true;
  v.muted = true;
  v.playsInline = true;
  v.autoplay = true;
  v.preload = 'auto';
  v.crossOrigin = 'anonymous';

  const tex = new THREE.VideoTexture(v);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.rotation = -Math.PI / 2;
  tex.center.set(0.5, 0.5);

  // Once dimensions are known: compute cover-fill repeat for a 16:9 tile.
  // After 90° CW rotation, tile-horizontal maps to video-height axis and
  // tile-vertical maps to video-width axis.
  // ratio=1 → perfect match (e.g. portrait 9:16 video on 16:9 tile after rotation).
  // ratio<1 → video too tall  → shrink sy  (crop top/bottom of video).
  // ratio>1 → video too wide  → shrink sx  (crop sides of video).
  // center=(0.5,0.5) keeps the crop centered automatically; offset stays (0,0).
  v.addEventListener('loadedmetadata', () => {
    const ratio = (16 * v.videoWidth) / (9 * v.videoHeight);
    if (ratio <= 1) {
      tex.repeat.set(1, ratio);
    } else {
      tex.repeat.set(1 / ratio, 1);
    }
    tex.offset.set(0, 0);
    v.currentTime = Math.random() * v.duration;
  });

  // Play as soon as seeking is done (or immediately if no seek happened)
  const startPlay = () => v.play().catch(() => {});
  v.addEventListener('seeked', startPlay, { once: true });
  v.addEventListener('canplay', startPlay, { once: true });

  v.load();

  videoPool.push({ vid: v, tex });
}

let poolIdx = 0;
function makeVideoMaterial() {
  const { tex } = videoPool[poolIdx % POOL_SIZE];
  poolIdx++;
  return new THREE.MeshStandardMaterial({
    map: tex,
    emissiveMap: tex,
    emissive: new THREE.Color(0xffffff),
    emissiveIntensity: 0.4,
  });
}

// ── Load OBJ + MTL model ──────────────────────────────────────────────────────
const mtlLoader = new MTLLoader();

mtlLoader.load(MTL_URL, (materials) => {
  materials.preload();
  const objLoader = new OBJLoader();
  objLoader.setMaterials(materials);
  objLoader.load(
    MODEL_URL,
    (model) => {
      // Center and scale to fit a ~2-unit bounding sphere
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 1.4 / maxDim;

      model.scale.setScalar(scale);
      model.position.sub(center.multiplyScalar(scale));

      model.traverse((child) => {
        if (!child.isMesh) return;
        child.castShadow = true;
        child.receiveShadow = true;
        allMeshes.push(child);

        if (child.name !== 'Male') {
          child.material = makeVideoMaterial();
          videoMeshes.push(child);
        }
      });

      scene.add(model);

      // Adjust camera to face the model
      const scaledBox = new THREE.Box3().setFromObject(model);
      const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
      controls.target.copy(scaledCenter);
      camera.position.set(scaledCenter.x, scaledCenter.y, scaledCenter.z + 4);
      controls.update();

      // Dismiss loading screen
      loadingOverlay.classList.add('fade-out');
      setTimeout(() => { loadingOverlay.style.display = 'none'; }, 650);
      setTimeout(() => { hint.classList.add('visible'); }, 900);
    },
    (xhr) => {
      if (xhr.lengthComputable) {
        progressBar.style.width = ((xhr.loaded / xhr.total) * 100).toFixed(1) + '%';
      }
    },
    (err) => {
      console.error('Failed to load model:', err);
      document.querySelector('#loading-overlay h1').textContent = 'Load failed';
    }
  );
});

// ── Video overlay ─────────────────────────────────────────────────────────────
const overlay       = document.getElementById('video-overlay');
const overlayVideo  = document.getElementById('overlay-video');
const overlayClose  = document.getElementById('video-overlay-close');

function openOverlay(src = DEFAULT_VIDEO) {
  if (overlayVideo.src !== new URL(src, location.href).href) {
    overlayVideo.src = src;
  }
  overlay.classList.add('open');
  overlayVideo.currentTime = 0;
  overlayVideo.play().catch(() => {});
  controls.autoRotate = false;
}

function closeOverlay() {
  overlay.classList.remove('open');
  overlayVideo.pause();
  controls.autoRotate = true;
}

overlayClose.addEventListener('click', closeOverlay);
overlay.addEventListener('click', (e) => { if (e.target === overlay) closeOverlay(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeOverlay(); });

// ── Raycaster for mesh click ───────────────────────────────────────────────────
const raycaster  = new THREE.Raycaster();
const pointer    = new THREE.Vector2();
let videoMeshes  = [];   // meshes that received the video texture
let allMeshes    = [];   // every mesh in the scene (for identification)
let pointerDown  = { x: 0, y: 0, time: 0 };

// On-screen name label (shown on click for identification)
const nameLabel = document.createElement('div');
nameLabel.style.cssText = [
  'position:fixed', 'bottom:32px', 'left:50%', 'transform:translateX(-50%)',
  'z-index:20', 'background:rgba(0,0,0,0.7)', 'color:#0cf',
  'font:12px/1 "Courier New",monospace', 'padding:6px 14px',
  'border-radius:4px', 'pointer-events:none', 'opacity:0',
  'transition:opacity 0.2s ease', 'white-space:nowrap'
].join(';');
document.body.appendChild(nameLabel);

let labelTimer = null;
function showLabel(name) {
  clearTimeout(labelTimer);
  nameLabel.textContent = name;
  nameLabel.style.opacity = '1';
  labelTimer = setTimeout(() => { nameLabel.style.opacity = '0'; }, 2500);
}

canvas.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY, time: Date.now() };
});

canvas.addEventListener('pointerup', (e) => {
  const dx   = e.clientX - pointerDown.x;
  const dy   = e.clientY - pointerDown.y;
  const dist = Math.hypot(dx, dy);
  const held = Date.now() - pointerDown.time;
  if (dist > 6 || held > 250) return;   // was a drag, not a click

  const rect = canvas.getBoundingClientRect();
  pointer.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const hits = raycaster.intersectObjects(allMeshes, false);
  if (!hits.length) return;

  const mesh = hits[0].object;
  const name = mesh.name || '(unnamed)';

  // Always log to console for identification
  console.log('[polygon click]', name);
  showLabel(name);

  // Only open the player if this mesh has the video texture
  if (videoMeshes.includes(mesh)) {
    const src = VIDEO_MAP[name] ?? DEFAULT_VIDEO;
    openOverlay(src);
  }
});

// ── Resize ────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  stars.rotation.y += 0.00006;
  videoPool.forEach(({ tex }) => { tex.needsUpdate = true; });
  renderer.render(scene, camera);
}
animate();
