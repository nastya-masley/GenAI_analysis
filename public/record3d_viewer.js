import * as THREE from 'three';

const WS_URL  = 'ws://localhost:8888';
const MAX_PTS = 400_000;

// ── DOM ───────────────────────────────────────────────────────────────────────
const canvas        = document.getElementById('scan-canvas');
const scanInfo      = document.getElementById('scan-info');
const connectIdle   = document.getElementById('connect-idle');
const record3dModal = document.getElementById('record3d-modal');
const closeModalBtn = document.getElementById('close-record3d-modal');
const connectBtn    = document.getElementById('connect-record3d');
const statusEl      = document.getElementById('record3d-status');
const captureBtn    = document.getElementById('capture-btn');
const disconnectBtn = document.getElementById('disconnect-btn');
const hud           = document.getElementById('hud');
const idleScreen    = document.getElementById('idle-screen');
const scanHint      = document.getElementById('scan-hint');
const liveLabel     = document.getElementById('live-label');

// ── Three.js ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
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
  size: 0.013,
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
let fpsCount=0, lastFpsT=performance.now(), fps=0;

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

  // fps
  fpsCount++;
  const now=performance.now();
  if (now-lastFpsT >= 1000) {
    fps = Math.round(fpsCount*1000/(now-lastFpsT));
    fpsCount=0; lastFpsT=now;
  }

  scanInfo.innerHTML =
    `<span>${fps} fps</span><span>${n} pts</span>` +
    (frozen ? '<span>FROZEN</span>' : '');
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws = null;

function openModal()  { record3dModal.hidden=false; setStatus(''); }
function closeModal() { record3dModal.hidden=true; }
function setStatus(msg, cls='') {
  statusEl.textContent = msg;
  statusEl.className   = 'modal-status'+(cls?' '+cls:'');
}

connectIdle.addEventListener('click', openModal);
closeModalBtn.addEventListener('click', closeModal);
record3dModal.addEventListener('click', e=>{ if(e.target===record3dModal) closeModal(); });
connectBtn.addEventListener('click', doConnect);
disconnectBtn.addEventListener('click', doDisconnect);

captureBtn.addEventListener('click', () => {
  frozen = !frozen;
  autoRotate = false;
  captureBtn.textContent = frozen ? 'Resume' : 'Capture Scan';
  captureBtn.className   = frozen ? 'btn blue' : 'btn green';
  if (liveLabel) liveLabel.textContent = frozen ? 'Frozen' : 'Live';
});

function doConnect() {
  if (ws){ ws.close(); ws=null; }
  setStatus('Connecting to ws://localhost:8888 …');
  connectBtn.disabled = true;

  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => setStatus('Connected — waiting for scan data…', 'success');

  ws.onmessage = ev => {
    if (ev.data instanceof ArrayBuffer) {
      parseBinaryFrame(ev.data);
    } else {
      try {
        const d = JSON.parse(ev.data);
        if (d.type === 'connected') {
          setStatus('Streaming!', 'success');
          setTimeout(closeModal, 400);
          showHud();
        }
      } catch { /* ignore */ }
    }
  };

  ws.onerror = () => {
    setStatus('Connection refused — is record3d_server.py running?', 'error');
    connectBtn.disabled = false;
  };

  ws.onclose = () => {
    connectBtn.disabled = false;
    if (hud.classList.contains('active')) hideHud();
  };
}

function doDisconnect() {
  if (ws){ ws.close(); ws=null; }
  hideHud();
}

function showHud() {
  idleScreen.style.display  = 'none';
  connectIdle.style.display = 'none';
  hud.classList.add('active');
  if (scanHint) scanHint.style.display = 'block';
  autoRotate = false;
  geo.setDrawRange(0, 0);   // clear test cloud
  normTimer = 0;
}

function hideHud() {
  hud.classList.remove('active');
  idleScreen.style.display  = 'flex';
  connectIdle.style.display = 'block';
  frozen = false;
  captureBtn.textContent = 'Capture Scan';
  captureBtn.className   = 'btn green';
  autoRotate = true;
  geo.setDrawRange(0, 3000);  // restore test cloud
  scanInfo.innerHTML = '<span>Disconnected — connect Record3D to scan</span>';
}
