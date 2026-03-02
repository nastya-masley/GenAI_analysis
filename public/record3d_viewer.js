import * as THREE from 'three';

const WS_URL  = 'ws://localhost:8888';
const MAX_PTS = 400_000;

// ── DOM ───────────────────────────────────────────────────────────────────────
const canvas     = document.getElementById('scan-canvas');
const hud        = document.getElementById('hud');
const idleScreen = document.getElementById('idle-screen');

// ── Three.js ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 3));
renderer.setClearColor(0x000508, 1);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 200);
camera.position.z = 3;

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// Pre-allocated geometry buffers (never re-allocated)
const geo    = new THREE.BufferGeometry();
const posArr = new Float32Array(MAX_PTS * 3);
const colArr = new Float32Array(MAX_PTS * 3);
const posBuf = new THREE.BufferAttribute(posArr, 3);
const colBuf = new THREE.BufferAttribute(colArr, 3);
posBuf.setUsage(THREE.DynamicDrawUsage);
colBuf.setUsage(THREE.DynamicDrawUsage);
geo.setAttribute('position', posBuf);
geo.setAttribute('color',    colBuf);
geo.setDrawRange(0, 0);

const mat = new THREE.PointsMaterial({
  size: 0.006,
  vertexColors: true,
  sizeAttenuation: true,
});

const cloud = new THREE.Points(geo, mat);
scene.add(cloud);

// ── test cloud on startup ─────────────────────────────────────────────────────
(function seedTest() {
  const N = 3000;
  for (let i = 0; i < N; i++) {
    posArr[i*3]   = (Math.random()-0.5)*2;
    posArr[i*3+1] = (Math.random()-0.5)*2;
    posArr[i*3+2] = (Math.random()-0.5)*2;
    colArr[i*3]   = 0.05;
    colArr[i*3+1] = 0.35 + Math.random()*0.3;
    colArr[i*3+2] = 0.65 + Math.random()*0.3;
  }
  posBuf.needsUpdate = true;
  colBuf.needsUpdate = true;
  geo.setDrawRange(0, N);
})();

// ── render loop ───────────────────────────────────────────────────────────────
let autoRotate = true;

function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (autoRotate) cloud.rotation.y += 0.003;
  renderer.render(scene, camera);
}
renderLoop();

// ── drag / zoom ───────────────────────────────────────────────────────────────
let dragging = false, prev = {x:0,y:0};
let rotX = 0, rotY = 0, camZ = 3;

canvas.addEventListener('mousedown',  e => { dragging=true; prev={x:e.clientX,y:e.clientY}; autoRotate=false; });
window.addEventListener('mouseup',    () => { dragging=false; });
window.addEventListener('mousemove',  e => {
  if (!dragging) return;
  rotY += (e.clientX - prev.x) * 0.006;
  rotX += (e.clientY - prev.y) * 0.006;
  rotX  = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
  cloud.rotation.set(rotX, rotY, 0);
  prev = {x:e.clientX, y:e.clientY};
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  camZ = Math.max(0.3, Math.min(10, camZ + e.deltaY * 0.003));
  camera.position.z = camZ;
}, {passive:false});
canvas.addEventListener('touchstart', e => {
  if (e.touches.length===1){ dragging=true; prev={x:e.touches[0].clientX,y:e.touches[0].clientY}; autoRotate=false; }
},{passive:true});
canvas.addEventListener('touchmove', e => {
  if (!dragging||e.touches.length!==1) return;
  rotY += (e.touches[0].clientX-prev.x)*0.006;
  rotX += (e.touches[0].clientY-prev.y)*0.006;
  cloud.rotation.set(rotX,rotY,0);
  prev={x:e.touches[0].clientX,y:e.touches[0].clientY};
},{passive:true});
canvas.addEventListener('touchend', ()=>{ dragging=false; });

// ── normalisation cache (recomputed every 20 frames) ─────────────────────────
let normCx=0, normCy=0, normCz=0, normScale=1;
let normTimer = 0;

function recomputeNorm(xyzF32, n) {
  let cx=0, cy=0, cz=0;
  for (let i=0; i<n; i++) { cx+=xyzF32[i*3]; cy+=xyzF32[i*3+1]; cz+=xyzF32[i*3+2]; }
  cx/=n; cy/=n; cz/=n;
  let maxR=0.001;
  for (let i=0; i<n; i++) {
    const dx=xyzF32[i*3]-cx, dy=xyzF32[i*3+1]-cy, dz=xyzF32[i*3+2]-cz;
    const r=dx*dx+dy*dy+dz*dz;
    if (r>maxR) maxR=r;
  }
  normCx=cx; normCy=cy; normCz=cz; normScale=1.5/Math.sqrt(maxR);
}

// ── binary frame parser ───────────────────────────────────────────────────────
let frozen = false;

function parseBinaryFrame(buf) {
  if (frozen) return;

  const n = new DataView(buf).getUint32(0, true);
  if (n===0 || n>MAX_PTS) return;

  const xyzF32 = new Float32Array(buf, 4, n*3);
  const rgbU8  = new Uint8Array(buf, 4 + n*12, n*3);

  // recompute normalisation every 20 frames
  normTimer++;
  if (normTimer % 20 === 1) recomputeNorm(xyzF32, n);

  const cx=normCx, cy=normCy, cz=normCz, s=normScale;

  // transform all points in one tight loop
  for (let i=0; i<n; i++) {
    posArr[i*3]   =  (xyzF32[i*3]   - cx) * s;
    posArr[i*3+1] = -(xyzF32[i*3+1] - cy) * s;  // flip Y
    posArr[i*3+2] = -(xyzF32[i*3+2] - cz) * s;  // flip Z

    // tint real colours toward cyan-blue
    colArr[i*3]   = rgbU8[i*3]   * 0.00157 + 0.05;   // /255*0.4+0.05
    colArr[i*3+1] = rgbU8[i*3+1] * 0.00196 + 0.20;   // /255*0.5+0.20
    colArr[i*3+2] = rgbU8[i*3+2] * 0.00118 + 0.55;   // /255*0.3+0.55
  }

  posBuf.needsUpdate = true;
  colBuf.needsUpdate = true;
  geo.setDrawRange(0, n);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws = null;

function setStatus(msg, cls = '') {
  // UI removed; status not shown
}

function doCapture() {
  frozen = true;
  autoRotate = false;
  saveScanAndGoToMain();
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && hud.classList.contains('active') && !frozen) {
    e.preventDefault();
    doCapture();
  }
});

const MAIN_PAGE = '/main.html';
const STORAGE_KEY = 'lidar_scans';
const MAX_SCANS = 50;
const SAMPLE_POINTS = 60000;

function saveScanAndGoToMain() {
  const n = geo.drawRange.count;
  if (n <= 0) {
    setStatus('No scan data', 'error');
    return;
  }
  const positions = [];
  const colors = [];
  const step = Math.max(1, Math.floor(n / SAMPLE_POINTS));
  for (let i = 0; i < n; i += step) {
    positions.push(posArr[i * 3], posArr[i * 3 + 1], posArr[i * 3 + 2]);
    colors.push(
      Math.round(colArr[i * 3] * 255),
      Math.round(colArr[i * 3 + 1] * 255),
      Math.round(colArr[i * 3 + 2] * 255)
    );
  }
  const scan = {
    id: Date.now(),
    positions,
    colors,
    pointCount: n,
    timestamp: new Date().toISOString()
  };
  try {
    let list = [];
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) list = JSON.parse(raw);
    } catch (_) {}
    list.unshift(scan);
    list = list.slice(0, MAX_SCANS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch (err) {
    console.warn('Could not save scan', err);
  }
  window.location.href = MAIN_PAGE;
}

function doConnect() {
  if (ws) { ws.close(); ws = null; }
  setStatus('Connecting…');

  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => setStatus('Connected — waiting for scan data…', 'connected');

  ws.onmessage = ev => {
    if (ev.data instanceof ArrayBuffer) {
      parseBinaryFrame(ev.data);
    } else {
      try {
        const d = JSON.parse(ev.data);
        if (d.type === 'connected') {
          setStatus('Streaming', 'connected');
          showHud();
        }
      } catch { /* ignore */ }
    }
  };

  ws.onerror = () => {
    setStatus('Connection refused — start record3d server?', 'error');
  };

  ws.onclose = () => {
    if (hud.classList.contains('active')) hideHud();
    setStatus('Disconnected — reconnecting…', 'error');
    setTimeout(doConnect, 3000);
  };
}

function doDisconnect() {
  if (ws){ ws.close(); ws=null; }
  hideHud();
}

function showHud() {
  idleScreen.style.display = 'none';
  hud.classList.add('active');
  autoRotate = false;
  geo.setDrawRange(0, 0);
  normTimer = 0;
}

function hideHud() {
  hud.classList.remove('active');
  idleScreen.style.display = 'flex';
  frozen = false;
  autoRotate = true;
  geo.setDrawRange(0, 3000);
}

// Auto-connect on load (no connect modal)
doConnect();
