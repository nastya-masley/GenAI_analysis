import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class FaceGlobe {
  constructor() {
    this.canvas = document.getElementById('globe-canvas');
    this.loadingOverlay = document.getElementById('loading');
    this.loadingText = this.loadingOverlay.querySelector('.loading-text');
    this.resetBtn = document.getElementById('reset-btn');
    this.uploadInput = document.getElementById('model-upload');
    this.cameraSelectBtn = document.getElementById('camera-select-btn');
    this.cameraModal = document.getElementById('camera-modal');
    this.cameraList = document.getElementById('camera-list');
    this.closeCameraModalBtn = document.getElementById('close-camera-modal');
    
    this.record3dBtn = document.getElementById('record3d-btn');
    this.record3dModal = document.getElementById('record3d-modal');
    this.closeRecord3dModalBtn = document.getElementById('close-record3d-modal');
    this.connectRecord3dBtn = document.getElementById('connect-record3d');
    this.record3dStatus = document.getElementById('record3d-status');
    this.record3dIpInput = document.getElementById('record3d-ip');
    this.record3dPortInput = document.getElementById('record3d-port');
    this.wifiInput = document.getElementById('wifi-input');
    this.usbInfo = document.getElementById('usb-info');
    
    this.videoContainer = document.getElementById('video-container');
    this.videoCanvas = document.getElementById('video-canvas');
    this.videoCtx = this.videoCanvas ? this.videoCanvas.getContext('2d') : null;
    this.videoInfo = document.getElementById('video-info');
    this.frameCount = 0;
    this.lastFpsTime = 0;
    this.fps = 0;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.faceGroup = null;
    this.faceMesh = null;
    this.atmosphere = null;
    this.stars = null;
    this.pointCloud = null;

    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.currentRotationX = 0;
    this.currentRotationY = 0;
    this.targetZoom = 2.8;
    this.currentZoom = 2.8;
    this.autoRotate = true;
    this.autoRotateSpeed = 0.0015;

    this.faceLandmarker = null;
    this.objLoader = new OBJLoader();
    this.gltfLoader = new GLTFLoader();

    this.selectedCameraId = null;
    this.availableCameras = [];
    
    this.record3dStreaming = false;
    this.record3dWebSocket = null;
    this.liveIndicator = null;

    this.init();
  }

  async init() {
    this.setupThreeJS();
    this.createStars();
    this.setupLighting();
    this.setupEventListeners();
    
    this.loadingText.textContent = 'Loading face detector...';
    
    try {
      await this.initFaceDetection();
      this.loadingText.textContent = 'Starting camera...';
      await this.captureAndProcessFace();
    } catch (err) {
      console.error('Face capture failed:', err);
      this.loadingText.textContent = 'Camera error. Using default...';
      await new Promise(r => setTimeout(r, 1500));
      this.createDefaultFaceGlobe();
    }

    this.loadingOverlay.classList.add('hidden');
    this.animate();
  }

  setupThreeJS() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      50,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.z = this.currentZoom;

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000508, 1);

    this.faceGroup = new THREE.Group();
    this.scene.add(this.faceGroup);
  }

  async initFaceDetection() {
    const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm');
    const { FaceLandmarker, FilesetResolver } = vision;

    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: 'VIDEO',
      numFaces: 1
    });
  }

  showRecord3dModal() {
    this.record3dModal.hidden = false;
    this.record3dStatus.textContent = '';
    this.record3dStatus.className = 'record3d-status';
    
    const radioButtons = document.querySelectorAll('input[name="stream-type"]');
    radioButtons.forEach(radio => {
      radio.addEventListener('change', () => {
        if (radio.value === 'wifi') {
          this.wifiInput.style.display = 'flex';
          this.usbInfo.style.display = 'none';
        } else {
          this.wifiInput.style.display = 'none';
          this.usbInfo.style.display = 'block';
        }
      });
    });
  }

  hideRecord3dModal() {
    this.record3dModal.hidden = true;
  }

  async connectRecord3d() {
    const streamType = document.querySelector('input[name="stream-type"]:checked').value;
    
    this.record3dStatus.textContent = 'Connecting...';
    this.record3dStatus.className = 'record3d-status';

    try {
      if (streamType === 'wifi') {
        const ip = this.record3dIpInput.value.trim();
        const port = this.record3dPortInput.value.trim() || '8080';
        
        if (!ip) {
          this.record3dStatus.textContent = 'Please enter iPhone IP address';
          this.record3dStatus.className = 'record3d-status error';
          return;
        }
        
        await this.connectRecord3dWifi(ip, port);
      } else {
        await this.connectRecord3dUsb();
      }
    } catch (err) {
      console.error('Record3D connection failed:', err);
      this.record3dStatus.textContent = `Connection failed: ${err.message}`;
      this.record3dStatus.className = 'record3d-status error';
    }
  }

  async connectRecord3dWifi(ip, port) {
    this.record3dStatus.textContent = 'Connecting via WiFi...';
    
    const wsUrl = `ws://${ip}:${port}`;
    
    try {
      this.record3dWebSocket = new WebSocket(wsUrl);
      
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 5000);
        
        this.record3dWebSocket.onopen = () => {
          clearTimeout(timeout);
          resolve();
        };
        
        this.record3dWebSocket.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('WebSocket connection failed'));
        };
      });

      this.record3dStatus.textContent = 'Connected! Starting LiDAR stream...';
      this.record3dStatus.className = 'record3d-status success';
      
      await new Promise(r => setTimeout(r, 500));
      this.hideRecord3dModal();
      this.startRecord3dStream();

    } catch (err) {
      this.record3dStatus.textContent = `WiFi connection failed. Make sure Record3D is streaming. Try: http://${ip}:${port}`;
      this.record3dStatus.className = 'record3d-status error';
      throw err;
    }
  }

  async connectRecord3dUsb() {
    this.record3dStatus.textContent = 'Connecting to local server (localhost:8888)...';
    
    const wsUrl = 'ws://localhost:8888';
    
    try {
      this.record3dWebSocket = new WebSocket(wsUrl);
      
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout - is the Python server running?'));
        }, 5000);
        
        this.record3dWebSocket.onopen = () => {
          clearTimeout(timeout);
          resolve();
        };
        
        this.record3dWebSocket.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('Could not connect to localhost:8888'));
        };
      });

      this.record3dStatus.textContent = 'Connected to USB stream!';
      this.record3dStatus.className = 'record3d-status success';
      
      await new Promise(r => setTimeout(r, 500));
      this.hideRecord3dModal();
      this.startRecord3dStream();

    } catch (err) {
      this.record3dStatus.textContent = 'USB server not found. Run: python -m record3d --stream';
      this.record3dStatus.className = 'record3d-status error';
      throw err;
    }
  }

  startRecord3dStream() {
    this.stopRecord3dStream();
    this.clearFaceGroup();
    
    this.record3dStreaming = true;
    this.autoRotate = false;
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
    
    if (this.videoContainer) {
      this.videoContainer.hidden = false;
      this.canvas.style.display = 'none';
    }
    
    this.showLiveIndicator();
    
    this.record3dWebSocket.onmessage = (event) => {
      this.processRecord3dFrame(event.data);
    };
    
    this.record3dWebSocket.onclose = () => {
      console.log('Record3D stream closed');
      this.stopRecord3dStream();
    };
    
    this.record3dWebSocket.onerror = (err) => {
      console.error('Record3D stream error:', err);
      this.stopRecord3dStream();
    };
  }

  createPointCloud() {
    const numPoints = 256 * 192;
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.006,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      sizeAttenuation: true
    });

    this.pointCloud = new THREE.Points(geometry, material);
    this.faceGroup.add(this.pointCloud);
  }

  processRecord3dFrame(data) {
    try {
      if (typeof data === 'string') {
        const json = JSON.parse(data);
        
        if (json.type === 'connected') {
          console.log('Record3D:', json.message);
          return;
        }
        
        if (json.rgb && this.videoCtx) {
          this.displayVideoFrame(json);
        }
        
      } else if (data instanceof Blob) {
        data.arrayBuffer().then(buffer => {
          if (this.pointCloud) this.updatePointCloudFromBuffer(buffer);
        });
      } else if (data instanceof ArrayBuffer) {
        if (this.pointCloud) this.updatePointCloudFromBuffer(data);
      }
    } catch (err) {
      console.warn('Failed to process Record3D frame:', err);
    }
  }

  displayVideoFrame(json) {
    const img = new Image();
    img.onload = () => {
      if (this.videoCanvas.width !== json.rgbWidth || this.videoCanvas.height !== json.rgbHeight) {
        this.videoCanvas.width = json.rgbWidth;
        this.videoCanvas.height = json.rgbHeight;
      }
      
      this.videoCtx.drawImage(img, 0, 0);
      
      this.frameCount++;
      const now = performance.now();
      const elapsed = now - this.lastFpsTime;
      
      if (elapsed >= 1000) {
        this.fps = Math.round(this.frameCount * 1000 / elapsed);
        this.frameCount = 0;
        this.lastFpsTime = now;
      }
      
      if (this.videoInfo) {
        this.videoInfo.innerHTML = `
          <span>Resolution: ${json.rgbWidth} × ${json.rgbHeight}</span>
          <span>FPS: ${this.fps}</span>
          <span>Depth: ${json.depthWidth || '-'} × ${json.depthHeight || '-'}</span>
          <span>Points: ${json.points ? json.points.length : '-'}</span>
        `;
      }
    };
    img.src = 'data:image/jpeg;base64,' + json.rgb;
  }

  updatePointCloudFromBuffer(buffer) {
    const positions = this.pointCloud.geometry.attributes.position.array;
    const colors = this.pointCloud.geometry.attributes.color.array;
    
    const dataView = new DataView(buffer);
    const width = 256;
    const height = 192;
    
    let offset = 0;
    let idx = 0;
    
    const hasHeader = buffer.byteLength > width * height * 4 * 4;
    if (hasHeader) {
      offset = 16;
    }
    
    const bytesPerPoint = (buffer.byteLength - offset) / (width * height);
    const hasColor = bytesPerPoint >= 16;
    
    for (let y = 0; y < height && offset < buffer.byteLength - 12; y++) {
      for (let x = 0; x < width && offset < buffer.byteLength - 12; x++) {
        try {
          const px = dataView.getFloat32(offset, true);
          const py = dataView.getFloat32(offset + 4, true);
          const pz = dataView.getFloat32(offset + 8, true);
          
          if (Math.abs(px) < 10 && Math.abs(py) < 10 && Math.abs(pz) < 10 && pz !== 0) {
            positions[idx * 3] = px;
            positions[idx * 3 + 1] = -py;
            positions[idx * 3 + 2] = -pz + 1;
            
            if (hasColor && offset + 15 < buffer.byteLength) {
              const r = dataView.getUint8(offset + 12) / 255;
              const g = dataView.getUint8(offset + 13) / 255;
              const b = dataView.getUint8(offset + 14) / 255;
              colors[idx * 3] = r * 0.3 + 0.1;
              colors[idx * 3 + 1] = g * 0.3 + 0.4;
              colors[idx * 3 + 2] = b * 0.3 + 0.7;
            } else {
              const depth = Math.abs(pz);
              colors[idx * 3] = 0.1 + depth * 0.1;
              colors[idx * 3 + 1] = 0.3 + (1 - depth) * 0.4;
              colors[idx * 3 + 2] = 0.6 + (1 - depth) * 0.4;
            }
          } else {
            positions[idx * 3] = 0;
            positions[idx * 3 + 1] = 0;
            positions[idx * 3 + 2] = -100;
          }
          
          offset += bytesPerPoint >= 16 ? 16 : 12;
          idx++;
        } catch (e) {
          break;
        }
      }
    }
    
    this.pointCloud.geometry.attributes.position.needsUpdate = true;
    this.pointCloud.geometry.attributes.color.needsUpdate = true;
  }

  updatePointCloudFromJson(json) {
    const points = json.points || [];
    const jsonColors = json.colors || [];
    const depth = json.depth || [];
    const width = json.width || 256;
    const height = json.height || 192;
    
    if (points.length > 0) {
      const numPoints = points.length;
      const positions = new Float32Array(numPoints * 3);
      const colors = new Float32Array(numPoints * 3);
      
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;
      
      for (let i = 0; i < numPoints; i++) {
        const p = points[i];
        const x = p.x !== undefined ? p.x : p[0];
        const y = p.y !== undefined ? p.y : p[1];
        const z = p.z !== undefined ? p.z : p[2];
        
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
      }
      
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;
      const centerZ = (minZ + maxZ) / 2;
      const rangeMax = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
      const scale = rangeMax > 0 ? 2.0 / rangeMax : 1;
      
      for (let i = 0; i < numPoints; i++) {
        const p = points[i];
        const x = (p.x !== undefined ? p.x : p[0]) - centerX;
        const y = (p.y !== undefined ? p.y : p[1]) - centerY;
        const z = (p.z !== undefined ? p.z : p[2]) - centerZ;
        
        positions[i * 3] = x * scale;
        positions[i * 3 + 1] = -y * scale;
        positions[i * 3 + 2] = -z * scale;
        
        if (jsonColors[i]) {
          const c = jsonColors[i];
          const r = (c.r !== undefined ? c.r : c[0]) / 255;
          const g = (c.g !== undefined ? c.g : c[1]) / 255;
          const b = (c.b !== undefined ? c.b : c[2]) / 255;
          colors[i * 3] = r * 0.3 + 0.1;
          colors[i * 3 + 1] = g * 0.3 + 0.4;
          colors[i * 3 + 2] = b * 0.3 + 0.6;
        } else {
          const depthNorm = (z * scale + 1) / 2;
          colors[i * 3] = 0.1 + depthNorm * 0.15;
          colors[i * 3 + 1] = 0.35 + (1 - depthNorm) * 0.35;
          colors[i * 3 + 2] = 0.6 + (1 - depthNorm) * 0.35;
        }
      }
      
      this.pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      this.pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      this.pointCloud.geometry.attributes.position.needsUpdate = true;
      this.pointCloud.geometry.attributes.color.needsUpdate = true;
      
    } else if (depth.length > 0) {
      const positions = this.pointCloud.geometry.attributes.position.array;
      const colors = this.pointCloud.geometry.attributes.color.array;
      
      const fovH = 58 * Math.PI / 180;
      const fovV = 45 * Math.PI / 180;
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const d = depth[idx] || 0;
          
          if (d > 0.1 && d < 5) {
            const angleH = (x / width - 0.5) * fovH;
            const angleV = (y / height - 0.5) * fovV;
            
            positions[idx * 3] = Math.tan(angleH) * d;
            positions[idx * 3 + 1] = -Math.tan(angleV) * d;
            positions[idx * 3 + 2] = -d + 1;
            
            colors[idx * 3] = 0.1 + d * 0.05;
            colors[idx * 3 + 1] = 0.3 + (2 - d) * 0.2;
            colors[idx * 3 + 2] = 0.6 + (2 - d) * 0.2;
          } else {
            positions[idx * 3 + 2] = -100;
          }
        }
      }
      
      this.pointCloud.geometry.attributes.position.needsUpdate = true;
      this.pointCloud.geometry.attributes.color.needsUpdate = true;
    }
  }

  showLiveIndicator() {
    if (this.liveIndicator) return;
    
    this.liveIndicator = document.createElement('div');
    this.liveIndicator.className = 'record3d-live-indicator';
    this.liveIndicator.innerHTML = `
      <span class="live-dot"></span>
      <span>Record3D LIVE</span>
      <button id="stop-record3d">Stop</button>
    `;
    document.body.appendChild(this.liveIndicator);
    
    this.liveIndicator.querySelector('#stop-record3d').addEventListener('click', () => {
      this.stopRecord3dStream();
      this.createDefaultFaceGlobe();
    });
  }

  hideLiveIndicator() {
    if (this.liveIndicator) {
      this.liveIndicator.remove();
      this.liveIndicator = null;
    }
  }

  stopRecord3dStream() {
    this.record3dStreaming = false;
    
    if (this.record3dWebSocket) {
      this.record3dWebSocket.close();
      this.record3dWebSocket = null;
    }
    
    this.hideLiveIndicator();
    this.autoRotate = true;
    
    if (this.videoContainer) {
      this.videoContainer.hidden = true;
      this.canvas.style.display = 'block';
    }
  }

  async getAvailableCameras() {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true });
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.availableCameras = devices.filter(device => device.kind === 'videoinput');
      return this.availableCameras;
    } catch (err) {
      console.error('Failed to enumerate cameras:', err);
      return [];
    }
  }

  async showCameraModal() {
    this.cameraModal.hidden = false;
    this.cameraList.innerHTML = '<p class="loading-cameras">Detecting cameras...</p>';

    const cameras = await this.getAvailableCameras();

    if (cameras.length === 0) {
      this.cameraList.innerHTML = '<p class="loading-cameras">No cameras found.</p>';
      return;
    }

    this.cameraList.innerHTML = '';

    cameras.forEach((camera, index) => {
      const isIphone = camera.label.toLowerCase().includes('iphone') || 
                       camera.label.toLowerCase().includes('ios') ||
                       camera.label.toLowerCase().includes('apple');
      
      const option = document.createElement('div');
      option.className = `camera-option ${isIphone ? 'iphone' : ''} ${camera.deviceId === this.selectedCameraId ? 'selected' : ''}`;
      option.dataset.deviceId = camera.deviceId;
      
      let cameraName = camera.label || `Camera ${index + 1}`;
      let cameraType = 'Built-in';
      
      if (isIphone) {
        cameraType = 'iPhone USB';
      } else if (cameraName.toLowerCase().includes('facetime')) {
        cameraType = 'FaceTime';
      } else if (cameraName.toLowerCase().includes('external') || cameraName.toLowerCase().includes('usb')) {
        cameraType = 'External';
      }

      option.innerHTML = `
        <div class="camera-icon">${isIphone ? '📱' : '📷'}</div>
        <div class="camera-name">${cameraName}</div>
        <div class="camera-type">${cameraType}</div>
      `;

      option.addEventListener('click', () => {
        this.cameraList.querySelectorAll('.camera-option').forEach(opt => opt.classList.remove('selected'));
        option.classList.add('selected');
        this.selectedCameraId = camera.deviceId;
      });

      option.addEventListener('dblclick', () => {
        this.selectedCameraId = camera.deviceId;
        this.useSelectedCamera();
      });

      this.cameraList.appendChild(option);
    });

    const useBtn = document.createElement('button');
    useBtn.className = 'globe-btn use-camera-btn';
    useBtn.textContent = 'Use Selected Camera';
    useBtn.addEventListener('click', () => this.useSelectedCamera());
    
    this.cameraList.appendChild(useBtn);
  }

  hideCameraModal() {
    this.cameraModal.hidden = true;
  }

  async useSelectedCamera() {
    if (!this.selectedCameraId) {
      const firstSelected = this.cameraList.querySelector('.camera-option.selected');
      if (firstSelected) {
        this.selectedCameraId = firstSelected.dataset.deviceId;
      } else {
        alert('Please select a camera first');
        return;
      }
    }

    this.hideCameraModal();
    this.stopRecord3dStream();
    this.clearFaceGroup();

    this.loadingOverlay.classList.remove('hidden');
    this.loadingText.textContent = 'Connecting to camera...';

    try {
      await this.captureAndProcessFace(this.selectedCameraId);
    } catch (err) {
      console.error('Camera capture failed:', err);
      this.loadingText.textContent = 'Failed. Using default...';
      await new Promise(r => setTimeout(r, 1500));
      this.createDefaultFaceGlobe();
    }

    this.loadingOverlay.classList.add('hidden');
  }

  async captureAndProcessFace(deviceId = null) {
    const constraints = {
      video: { 
        width: { ideal: 1920 }, 
        height: { ideal: 1080 }, 
        frameRate: { ideal: 30 }
      }
    };

    if (deviceId) {
      constraints.video.deviceId = { exact: deviceId };
    } else {
      constraints.video.facingMode = 'user';
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);

    const video = document.createElement('video');
    video.srcObject = stream;
    video.setAttribute('playsinline', '');
    await video.play();

    this.loadingText.textContent = 'Position your face in frame...';
    await new Promise(r => setTimeout(r, 1500));

    let bestLandmarks = null;
    let bestConfidence = 0;
    const maxAttempts = 45;

    for (let i = 0; i < maxAttempts; i++) {
      const startTimeMs = performance.now();
      const results = this.faceLandmarker.detectForVideo(video, startTimeMs);

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        
        let minZ = Infinity, maxZ = -Infinity;
        landmarks.forEach(lm => {
          minZ = Math.min(minZ, lm.z);
          maxZ = Math.max(maxZ, lm.z);
        });
        const zRange = maxZ - minZ;

        if (zRange > bestConfidence) {
          bestConfidence = zRange;
          bestLandmarks = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
        }
      }

      await new Promise(r => setTimeout(r, 80));
      const progress = Math.round((i / maxAttempts) * 100);
      this.loadingText.textContent = `Scanning face... ${progress}%`;
    }

    stream.getTracks().forEach(track => track.stop());

    if (bestLandmarks) {
      this.loadingText.textContent = 'Building 3D model...';
      await new Promise(r => setTimeout(r, 300));
      this.createPreciseFaceMesh(bestLandmarks);
    } else {
      throw new Error('No face detected');
    }
  }

  async loadUploadedModel(file) {
    this.stopRecord3dStream();
    this.clearFaceGroup();
    
    this.loadingOverlay.classList.remove('hidden');
    this.loadingText.textContent = 'Loading 3D scan...';

    const ext = file.name.split('.').pop().toLowerCase();
    const url = URL.createObjectURL(file);

    try {
      let geometry;

      if (ext === 'obj') {
        const obj = await new Promise((resolve, reject) => {
          this.objLoader.load(url, resolve, undefined, reject);
        });
        obj.traverse((child) => {
          if (child.isMesh && !geometry) {
            geometry = child.geometry.clone();
          }
        });
      } else if (ext === 'glb' || ext === 'gltf') {
        const gltf = await new Promise((resolve, reject) => {
          this.gltfLoader.load(url, resolve, undefined, reject);
        });
        gltf.scene.traverse((child) => {
          if (child.isMesh && !geometry) {
            geometry = child.geometry.clone();
          }
        });
      }

      URL.revokeObjectURL(url);

      if (!geometry) {
        throw new Error('No mesh found in file');
      }

      geometry.computeBoundingBox();
      const bbox = geometry.boundingBox;
      const center = new THREE.Vector3();
      bbox.getCenter(center);
      
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2.0 / maxDim;

      geometry.translate(-center.x, -center.y, -center.z);
      geometry.scale(scale, scale, scale);
      geometry.computeVertexNormals();

      const material = this.createGlobeMaterial();
      this.faceMesh = new THREE.Mesh(geometry, material);
      this.faceGroup.add(this.faceMesh);

      this.createAtmosphere(geometry);
      this.createWireframe(geometry);

      this.targetRotationX = 0;
      this.targetRotationY = 0;
      this.targetZoom = 2.8;
      this.autoRotate = true;

      this.loadingOverlay.classList.add('hidden');

    } catch (err) {
      console.error('Failed to load model:', err);
      URL.revokeObjectURL(url);
      this.loadingText.textContent = 'Failed to load. Using default...';
      await new Promise(r => setTimeout(r, 1500));
      this.createDefaultFaceGlobe();
      this.loadingOverlay.classList.add('hidden');
    }
  }

  clearFaceGroup() {
    while (this.faceGroup.children.length > 0) {
      const child = this.faceGroup.children[0];
      this.faceGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach(m => m.dispose());
        } else {
          child.material.dispose();
        }
      }
    }
    this.faceMesh = null;
    this.atmosphere = null;
    this.pointCloud = null;
  }

  createPreciseFaceMesh(landmarks) {
    const numLandmarks = landmarks.length;
    
    let sumX = 0, sumY = 0, sumZ = 0;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    landmarks.forEach(lm => {
      sumX += lm.x; sumY += lm.y; sumZ += lm.z;
      minX = Math.min(minX, lm.x); maxX = Math.max(maxX, lm.x);
      minY = Math.min(minY, lm.y); maxY = Math.max(maxY, lm.y);
    });

    const centerX = sumX / numLandmarks;
    const centerY = sumY / numLandmarks;
    const centerZ = sumZ / numLandmarks;

    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    
    const scale = 2.0 / Math.max(rangeX, rangeY);
    const zScale = scale * 2.5;

    const positions = new Float32Array(numLandmarks * 3);

    for (let i = 0; i < numLandmarks; i++) {
      const lm = landmarks[i];
      positions[i * 3] = (lm.x - centerX) * scale;
      positions[i * 3 + 1] = -(lm.y - centerY) * scale;
      positions[i * 3 + 2] = -(lm.z - centerZ) * zScale;
    }

    const triangulation = this.getMediaPipeFaceTriangulation();
    const indices = [];

    for (let i = 0; i < triangulation.length; i += 3) {
      const a = triangulation[i];
      const b = triangulation[i + 1];
      const c = triangulation[i + 2];
      
      if (a < numLandmarks && b < numLandmarks && c < numLandmarks) {
        indices.push(a, b, c);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = this.createGlobeMaterial();
    this.faceMesh = new THREE.Mesh(geometry, material);
    this.faceGroup.add(this.faceMesh);

    this.createAtmosphere(geometry);
    this.createWireframe(geometry);
  }

  createWireframe(geometry) {
    const wireGeo = new THREE.WireframeGeometry(geometry);
    const wireMat = new THREE.LineBasicMaterial({ 
      color: 0x1a4a7a,
      transparent: true,
      opacity: 0.12
    });
    const wireframe = new THREE.LineSegments(wireGeo, wireMat);
    this.faceGroup.add(wireframe);
  }

  createDefaultFaceGlobe() {
    const geometry = new THREE.SphereGeometry(0.9, 64, 64);
    const posAttr = geometry.attributes.position;
    
    for (let i = 0; i < posAttr.count; i++) {
      let x = posAttr.getX(i);
      let y = posAttr.getY(i);
      let z = posAttr.getZ(i);
      
      const origX = x, origY = y;
      
      x *= 0.78;
      y *= 1.05;
      
      if (z > 0) {
        if (Math.abs(origX) < 0.12 && origY > -0.15 && origY < 0.25) {
          z += 0.18 * (1 - Math.abs(origY) * 2);
        }
        
        if (Math.abs(origX) < 0.2 && origY > -0.35 && origY < -0.05) {
          const noseDist = Math.sqrt(origX * origX + Math.pow(origY + 0.2, 2));
          z += Math.max(0, 0.25 - noseDist * 1.5);
        }
        
        const eyeDistL = Math.sqrt(Math.pow(origX + 0.28, 2) + Math.pow(origY - 0.18, 2));
        const eyeDistR = Math.sqrt(Math.pow(origX - 0.28, 2) + Math.pow(origY - 0.18, 2));
        if (eyeDistL < 0.12) z -= 0.06 * (1 - eyeDistL / 0.12);
        if (eyeDistR < 0.12) z -= 0.06 * (1 - eyeDistR / 0.12);
        
        if (Math.abs(origX) < 0.22 && origY > -0.55 && origY < -0.35) z -= 0.04;
        
        const browDistL = Math.sqrt(Math.pow(origX + 0.28, 2) + Math.pow(origY - 0.32, 2));
        const browDistR = Math.sqrt(Math.pow(origX - 0.28, 2) + Math.pow(origY - 0.32, 2));
        if (browDistL < 0.15) z += 0.04 * (1 - browDistL / 0.15);
        if (browDistR < 0.15) z += 0.04 * (1 - browDistR / 0.15);
        
        const cheekDistL = Math.sqrt(Math.pow(origX + 0.4, 2) + Math.pow(origY - 0.0, 2));
        const cheekDistR = Math.sqrt(Math.pow(origX - 0.4, 2) + Math.pow(origY - 0.0, 2));
        if (cheekDistL < 0.2) z += 0.06 * (1 - cheekDistL / 0.2);
        if (cheekDistR < 0.2) z += 0.06 * (1 - cheekDistR / 0.2);
      }
      
      posAttr.setXYZ(i, x, y, z);
    }
    
    geometry.computeVertexNormals();

    const material = this.createGlobeMaterial();
    this.faceMesh = new THREE.Mesh(geometry, material);
    this.faceGroup.add(this.faceMesh);

    this.createAtmosphere(geometry);
  }

  createGlobeMaterial() {
    const vertexShader = `
      varying vec3 vNormal;
      varying vec3 vPosition;
      varying vec3 vViewPosition;
      
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vPosition = position;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewPosition = -mvPosition.xyz;
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      uniform float time;
      varying vec3 vNormal;
      varying vec3 vPosition;
      varying vec3 vViewPosition;

      float noise(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
      }

      void main() {
        vec3 pos = vPosition;
        
        float theta = atan(pos.y, pos.x);
        float phi = atan(length(pos.xy), pos.z);
        
        float gridLat = abs(sin(phi * 10.0));
        float gridLon = abs(sin(theta * 16.0));
        
        float sphereGrid = max(
          smoothstep(0.92, 1.0, gridLat),
          smoothstep(0.92, 1.0, gridLon)
        );
        
        float gridX = abs(sin(pos.x * 12.0));
        float gridY = abs(sin(pos.y * 12.0));
        float gridZ = abs(sin(pos.z * 10.0));
        
        float cartGrid = max(
          max(smoothstep(0.93, 1.0, gridX), smoothstep(0.93, 1.0, gridY)),
          smoothstep(0.93, 1.0, gridZ)
        );
        
        float grid = max(sphereGrid, cartGrid);

        float depthFactor = smoothstep(-0.5, 0.5, pos.z);
        
        vec3 deepColor = vec3(0.005, 0.015, 0.04);
        vec3 midColor = vec3(0.02, 0.06, 0.12);
        vec3 surfaceColor = vec3(0.04, 0.1, 0.18);
        
        vec3 baseColor = mix(deepColor, mix(midColor, surfaceColor, depthFactor), 0.7);
        
        vec3 gridColor = vec3(0.12, 0.35, 0.65);
        vec3 highlightColor = vec3(0.25, 0.55, 0.95);
        vec3 glowColor = vec3(0.15, 0.45, 0.85);

        vec3 finalColor = mix(baseColor, gridColor, grid * 0.9);

        vec3 viewDir = normalize(vViewPosition);
        float rim = 1.0 - max(0.0, dot(vNormal, viewDir));
        rim = pow(rim, 2.2);
        finalColor += glowColor * rim * 0.55;

        float fresnel = pow(1.0 - max(0.0, dot(vNormal, viewDir)), 4.0);
        finalColor += highlightColor * fresnel * 0.2;

        float pulse = sin(time * 1.8 + pos.y * 4.0) * 0.5 + 0.5;
        float hotspot = noise(pos.xy * 12.0 + time * 0.06);
        if (hotspot > 0.965) {
          finalColor += highlightColor * pulse * 0.5;
        }

        float scan = sin(pos.y * 25.0 + time * 1.5) * 0.5 + 0.5;
        scan = smoothstep(0.85, 1.0, scan);
        finalColor += glowColor * scan * 0.12;

        gl_FragColor = vec4(finalColor, 1.0);
      }
    `;

    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: { time: { value: 0 } },
      side: THREE.DoubleSide
    });
  }

  createAtmosphere(baseGeometry) {
    baseGeometry.computeBoundingSphere();
    const radius = baseGeometry.boundingSphere.radius * 1.18;
    
    const atmosphereGeom = new THREE.SphereGeometry(radius, 48, 48);

    const material = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          gl_FragColor = vec4(0.1, 0.4, 0.9, intensity * 0.45);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true
    });

    this.atmosphere = new THREE.Mesh(atmosphereGeom, material);
    this.faceGroup.add(this.atmosphere);
  }

  getMediaPipeFaceTriangulation() {
    return [127,34,139,11,0,37,232,231,120,72,37,39,128,121,47,232,121,128,104,69,67,175,171,148,118,50,101,73,39,40,9,151,108,48,115,131,194,204,211,74,40,185,80,42,183,40,92,186,230,229,118,202,212,214,83,18,17,76,61,146,160,29,30,56,157,173,106,204,194,135,214,192,203,165,98,21,71,68,51,45,4,144,24,23,77,146,91,205,50,187,201,200,18,91,106,182,90,91,181,85,84,17,206,203,36,148,171,140,92,40,39,193,189,244,159,158,28,247,246,161,236,3,196,54,68,104,193,168,8,117,228,31,189,193,55,98,97,99,126,47,100,166,79,218,155,154,26,209,49,131,135,136,150,47,126,217,223,52,53,45,51,134,211,170,140,67,69,108,43,106,91,230,119,120,226,130,247,63,53,52,238,20,242,46,70,156,78,62,96,46,53,63,143,34,227,123,117,111,44,125,19,236,134,51,216,206,205,154,153,22,39,37,167,200,201,208,36,142,100,57,212,202,20,60,99,28,158,157,35,226,113,160,159,27,204,202,210,113,225,46,43,202,204,62,78,191,132,129,142,245,98,99,33,7,163,4,45,220,61,62,191,80,81,42,180,179,41,93,234,216,24,110,228,25,130,226,23,24,229,198,236,196,173,157,172,42,81,38,171,175,232,233,232,175,156,70,63,192,214,212,83,201,18,239,238,241,17,84,201,73,72,39,216,212,57,214,135,169,210,202,214,169,135,170,220,45,134,219,220,134,218,219,134,217,218,134,47,217,134,126,47,134,217,126,134,100,142,36,93,216,205,192,212,57,10,338,297,297,338,332,332,338,284,284,338,251,251,338,389,389,338,356,356,338,454,454,338,323,323,338,361,361,338,288,288,338,397,397,338,365,365,338,379,379,338,378,378,338,400,400,338,377,377,338,152,152,338,148,148,338,176,176,338,149,149,338,150,150,338,136,136,338,172,172,338,58,58,338,132,132,338,93,93,338,234,234,338,127,127,338,162,162,338,21,21,338,54,54,338,103,103,338,67,67,338,109,109,338,10];
  }

  createStars() {
    const positions = new Float32Array(2000 * 3);
    for (let i = 0; i < 2000; i++) {
      const r = 30 + Math.random() * 60;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.stars = new THREE.Points(geo, new THREE.PointsMaterial({
      color: 0xffffff, size: 0.35, transparent: true, opacity: 0.6
    }));
    this.scene.add(this.stars);
  }

  setupLighting() {
    this.scene.add(new THREE.AmbientLight(0x223344, 0.4));
    const l1 = new THREE.DirectionalLight(0xffffff, 0.6);
    l1.position.set(3, 2, 5);
    this.scene.add(l1);
    const l2 = new THREE.DirectionalLight(0x4488ff, 0.25);
    l2.position.set(-3, -1, 2);
    this.scene.add(l2);
  }

  setupEventListeners() {
    window.addEventListener('resize', () => this.onResize());
    this.resetBtn.addEventListener('click', () => this.recaptureFace());
    
    if (this.uploadInput) {
      this.uploadInput.addEventListener('change', (e) => {
        if (e.target.files[0]) this.loadUploadedModel(e.target.files[0]);
      });
    }
    
    if (this.cameraSelectBtn) {
      this.cameraSelectBtn.addEventListener('click', () => this.showCameraModal());
    }
    if (this.closeCameraModalBtn) {
      this.closeCameraModalBtn.addEventListener('click', () => this.hideCameraModal());
    }
    if (this.cameraModal) {
      this.cameraModal.addEventListener('click', (e) => {
        if (e.target === this.cameraModal) this.hideCameraModal();
      });
    }
    
    if (this.record3dBtn) {
      this.record3dBtn.addEventListener('click', () => this.showRecord3dModal());
    }
    if (this.closeRecord3dModalBtn) {
      this.closeRecord3dModalBtn.addEventListener('click', () => this.hideRecord3dModal());
    }
    if (this.connectRecord3dBtn) {
      this.connectRecord3dBtn.addEventListener('click', () => this.connectRecord3d());
    }
    if (this.record3dModal) {
      this.record3dModal.addEventListener('click', (e) => {
        if (e.target === this.record3dModal) this.hideRecord3dModal();
      });
    }

    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };

    this.canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
      this.autoRotate = false;
    });
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      this.targetRotationY += (e.clientX - prevMouse.x) * 0.005;
      this.targetRotationX += (e.clientY - prevMouse.y) * 0.005;
      this.targetRotationX = Math.max(-Math.PI/2, Math.min(Math.PI/2, this.targetRotationX));
      prevMouse = { x: e.clientX, y: e.clientY };
    });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.targetZoom = Math.max(1.2, Math.min(6, this.targetZoom + e.deltaY * 0.002));
    }, { passive: false });

    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.autoRotate = false;
      }
    });
    this.canvas.addEventListener('touchmove', (e) => {
      if (!isDragging || e.touches.length !== 1) return;
      this.targetRotationY += (e.touches[0].clientX - prevMouse.x) * 0.005;
      this.targetRotationX += (e.touches[0].clientY - prevMouse.y) * 0.005;
      this.targetRotationX = Math.max(-Math.PI/2, Math.min(Math.PI/2, this.targetRotationX));
      prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    });
    this.canvas.addEventListener('touchend', () => isDragging = false);
  }

  async recaptureFace() {
    this.stopRecord3dStream();
    this.clearFaceGroup();
    this.loadingOverlay.classList.remove('hidden');
    this.loadingText.textContent = 'Starting camera...';
    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.targetZoom = 2.8;
    this.autoRotate = true;

    try {
      await this.captureAndProcessFace(this.selectedCameraId);
    } catch (err) {
      this.loadingText.textContent = 'Failed. Using default...';
      await new Promise(r => setTimeout(r, 1000));
      this.createDefaultFaceGlobe();
    }
    this.loadingOverlay.classList.add('hidden');
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.faceMesh?.material?.uniforms) {
      this.faceMesh.material.uniforms.time.value = performance.now() * 0.001;
    }

    if (this.autoRotate && !this.record3dStreaming) {
      this.targetRotationY += this.autoRotateSpeed;
    }

    this.currentRotationX += (this.targetRotationX - this.currentRotationX) * 0.05;
    this.currentRotationY += (this.targetRotationY - this.currentRotationY) * 0.05;
    this.currentZoom += (this.targetZoom - this.currentZoom) * 0.05;

    this.faceGroup.rotation.x = this.currentRotationX;
    this.faceGroup.rotation.y = this.currentRotationY;
    if (this.stars) this.stars.rotation.y += 0.00008;
    this.camera.position.z = this.currentZoom;

    this.renderer.render(this.scene, this.camera);
  }
}

window.addEventListener('DOMContentLoaded', () => new FaceGlobe());
