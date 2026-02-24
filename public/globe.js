import * as THREE from 'three';

class FaceAssembly {
  constructor() {
    this.canvas = document.getElementById('globe-canvas');
    this.loadingOverlay = document.getElementById('loading');
    this.resetBtn = document.getElementById('reset-btn');

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.particles = null;
    this.stars = null;

    this.particleCount = 8000;
    this.originalPositions = [];
    this.scatteredPositions = [];
    this.currentPositions = [];
    this.assemblyProgress = 0;
    this.isAssembled = false;
    this.animationPhase = 'scattering';

    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.currentRotationX = 0;
    this.currentRotationY = 0;
    this.targetZoom = 5;
    this.currentZoom = 5;
    this.autoRotate = true;
    this.autoRotateSpeed = 0.002;

    this.init();
  }

  async init() {
    this.setupThreeJS();
    this.generateFacePoints();
    this.createParticles();
    this.createStars();
    this.setupLighting();
    this.setupEventListeners();
    this.animate();

    setTimeout(() => {
      this.loadingOverlay.classList.add('hidden');
      this.startAssemblyAnimation();
    }, 800);
  }

  setupThreeJS() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      45,
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
    this.renderer.setClearColor(0x000000, 1);
  }

  generateFacePoints() {
    const facePoints = [];
    
    const addEllipsoidPoints = (cx, cy, cz, rx, ry, rz, count, zOffset = 0) => {
      for (let i = 0; i < count; i++) {
        const u = Math.random() * Math.PI * 2;
        const v = Math.random() * Math.PI;
        const x = cx + rx * Math.sin(v) * Math.cos(u);
        const y = cy + ry * Math.sin(v) * Math.sin(u);
        let z = cz + rz * Math.cos(v);
        if (z < cz) z = cz + (cz - z) * 0.3;
        z += zOffset;
        facePoints.push({ x, y, z });
      }
    };

    addEllipsoidPoints(0, 0, 0, 0.8, 1.0, 0.6, 3000);

    const addSpherePoints = (cx, cy, cz, r, count) => {
      for (let i = 0; i < count; i++) {
        const u = Math.random() * Math.PI * 2;
        const v = Math.random() * Math.PI;
        const x = cx + r * Math.sin(v) * Math.cos(u);
        const y = cy + r * Math.sin(v) * Math.sin(u);
        const z = cz + r * Math.cos(v) * 0.5 + r * 0.5;
        facePoints.push({ x, y, z });
      }
    };

    addSpherePoints(-0.28, 0.25, 0.35, 0.12, 400);
    addSpherePoints(0.28, 0.25, 0.35, 0.12, 400);

    const addIrisPoints = (cx, cy, cz, r, count) => {
      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * r;
        facePoints.push({
          x: cx + Math.cos(angle) * radius,
          y: cy + Math.sin(angle) * radius,
          z: cz + 0.05
        });
      }
    };

    addIrisPoints(-0.28, 0.25, 0.5, 0.06, 200);
    addIrisPoints(0.28, 0.25, 0.5, 0.06, 200);

    for (let i = 0; i < 600; i++) {
      const t = (i / 600) * Math.PI * 2;
      const noseX = Math.sin(t) * 0.08;
      const noseY = -0.1 + Math.cos(t) * 0.25;
      const noseZ = 0.5 + Math.sin(t * 0.5) * 0.15 + Math.random() * 0.05;
      facePoints.push({ x: noseX, y: noseY, z: noseZ });
    }

    for (let i = 0; i < 300; i++) {
      const t = Math.random();
      const bridgeY = 0.1 + t * 0.3;
      const bridgeZ = 0.45 + t * 0.1;
      facePoints.push({
        x: (Math.random() - 0.5) * 0.06,
        y: bridgeY,
        z: bridgeZ + Math.random() * 0.05
      });
    }

    for (let i = 0; i < 500; i++) {
      const t = (i / 500) * Math.PI;
      const lipWidth = 0.25;
      const x = Math.cos(t) * lipWidth;
      const y = -0.35 + Math.sin(t) * 0.03;
      const z = 0.4 + Math.sin(t) * 0.05 + Math.random() * 0.02;
      facePoints.push({ x, y, z });
    }

    for (let i = 0; i < 400; i++) {
      const t = (i / 400) * Math.PI;
      const x = Math.cos(t) * 0.22;
      const y = -0.4 + Math.sin(t) * -0.025;
      const z = 0.38 + Math.sin(t) * 0.03 + Math.random() * 0.02;
      facePoints.push({ x, y, z });
    }

    const addEyebrowPoints = (startX, y, count, direction) => {
      for (let i = 0; i < count; i++) {
        const t = i / count;
        const x = startX + direction * t * 0.25;
        const eyebrowY = y + Math.sin(t * Math.PI) * 0.03;
        const z = 0.42 + Math.random() * 0.03;
        facePoints.push({ x, y: eyebrowY, z });
      }
    };

    addEyebrowPoints(-0.38, 0.45, 300, 1);
    addEyebrowPoints(0.38, 0.45, 300, -1);

    for (let i = 0; i < 400; i++) {
      const side = i < 200 ? -1 : 1;
      const t = Math.random();
      const earX = side * (0.78 + Math.random() * 0.08);
      const earY = 0.1 + (Math.random() - 0.5) * 0.4;
      const earZ = -0.1 + Math.random() * 0.15;
      facePoints.push({ x: earX, y: earY, z: earZ });
    }

    for (let i = 0; i < 800; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 0.75 + Math.random() * 0.15;
      const y = 0.6 + Math.random() * 0.5;
      const x = Math.cos(angle) * radius * (1 - (y - 0.6) * 0.5);
      const z = Math.sin(angle) * radius * 0.6 * (1 - (y - 0.6) * 0.3);
      facePoints.push({ x, y, z });
    }

    for (let i = 0; i < 300; i++) {
      const angle = Math.random() * Math.PI;
      const radius = 0.6 + Math.random() * 0.2;
      const y = -0.7 - Math.random() * 0.3;
      const x = Math.cos(angle) * radius * 0.7;
      const z = Math.sin(angle) * radius * 0.5 - 0.1;
      facePoints.push({ x, y, z });
    }

    while (facePoints.length < this.particleCount) {
      const existing = facePoints[Math.floor(Math.random() * facePoints.length)];
      facePoints.push({
        x: existing.x + (Math.random() - 0.5) * 0.05,
        y: existing.y + (Math.random() - 0.5) * 0.05,
        z: existing.z + (Math.random() - 0.5) * 0.05
      });
    }

    this.originalPositions = facePoints.slice(0, this.particleCount);

    this.scatteredPositions = this.originalPositions.map(() => {
      const radius = 3 + Math.random() * 5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      return {
        x: radius * Math.sin(phi) * Math.cos(theta),
        y: radius * Math.sin(phi) * Math.sin(theta),
        z: radius * Math.cos(phi)
      };
    });

    this.currentPositions = this.scatteredPositions.map(p => ({ ...p }));
  }

  createParticles() {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.particleCount * 3);
    const colors = new Float32Array(this.particleCount * 3);
    const sizes = new Float32Array(this.particleCount);

    for (let i = 0; i < this.particleCount; i++) {
      const pos = this.currentPositions[i];
      positions[i * 3] = pos.x;
      positions[i * 3 + 1] = pos.y;
      positions[i * 3 + 2] = pos.z;

      const brightness = 0.5 + Math.random() * 0.5;
      colors[i * 3] = brightness * 0.6;
      colors[i * 3 + 1] = brightness * 0.8;
      colors[i * 3 + 2] = brightness;

      sizes[i] = 0.015 + Math.random() * 0.015;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const vertexShader = `
      attribute float size;
      varying vec3 vColor;
      
      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      varying vec3 vColor;
      
      void main() {
        float dist = length(gl_PointCoord - vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha);
      }
    `;

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      vertexColors: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.particles = new THREE.Points(geometry, material);
    this.scene.add(this.particles);
  }

  createStars() {
    const starsGeometry = new THREE.BufferGeometry();
    const starCount = 2000;
    const positions = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const radius = 30 + Math.random() * 70;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
    }

    starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const starsMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.3,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true
    });

    this.stars = new THREE.Points(starsGeometry, starsMaterial);
    this.scene.add(this.stars);
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    this.scene.add(ambientLight);
  }

  setupEventListeners() {
    window.addEventListener('resize', () => this.onResize());

    this.resetBtn.addEventListener('click', () => this.resetView());

    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    this.canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
      this.autoRotate = false;
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;

      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;

      this.targetRotationY += deltaX * 0.005;
      this.targetRotationX += deltaY * 0.005;
      this.targetRotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.targetRotationX));

      previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.targetZoom += e.deltaY * 0.002;
      this.targetZoom = Math.max(2, Math.min(10, this.targetZoom));
    }, { passive: false });

    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.autoRotate = false;
      }
    });

    this.canvas.addEventListener('touchmove', (e) => {
      if (!isDragging || e.touches.length !== 1) return;

      const deltaX = e.touches[0].clientX - previousMousePosition.x;
      const deltaY = e.touches[0].clientY - previousMousePosition.y;

      this.targetRotationY += deltaX * 0.005;
      this.targetRotationX += deltaY * 0.005;
      this.targetRotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.targetRotationX));

      previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    });

    this.canvas.addEventListener('touchend', () => {
      isDragging = false;
    });
  }

  startAssemblyAnimation() {
    this.animationPhase = 'assembling';
    this.assemblyProgress = 0;
  }

  resetView() {
    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.targetZoom = 5;
    this.autoRotate = true;
    
    this.animationPhase = 'scattering';
    this.assemblyProgress = 1;
    
    setTimeout(() => {
      this.startAssemblyAnimation();
    }, 1500);
  }

  updateParticlePositions() {
    if (this.animationPhase === 'assembling' && this.assemblyProgress < 1) {
      this.assemblyProgress += 0.003;
      this.assemblyProgress = Math.min(1, this.assemblyProgress);
    } else if (this.animationPhase === 'scattering' && this.assemblyProgress > 0) {
      this.assemblyProgress -= 0.008;
      this.assemblyProgress = Math.max(0, this.assemblyProgress);
    }

    const easeProgress = this.easeInOutCubic(this.assemblyProgress);
    const positions = this.particles.geometry.attributes.position.array;
    const time = performance.now() * 0.001;

    for (let i = 0; i < this.particleCount; i++) {
      const scattered = this.scatteredPositions[i];
      const original = this.originalPositions[i];
      
      const delay = (i / this.particleCount) * 0.3;
      const individualProgress = Math.max(0, Math.min(1, (easeProgress - delay) / (1 - delay)));
      
      const floatX = Math.sin(time * 0.5 + i * 0.1) * 0.01 * (1 - individualProgress);
      const floatY = Math.cos(time * 0.3 + i * 0.15) * 0.01 * (1 - individualProgress);
      const floatZ = Math.sin(time * 0.4 + i * 0.12) * 0.01 * (1 - individualProgress);

      positions[i * 3] = scattered.x + (original.x - scattered.x) * individualProgress + floatX;
      positions[i * 3 + 1] = scattered.y + (original.y - scattered.y) * individualProgress + floatY;
      positions[i * 3 + 2] = scattered.z + (original.z - scattered.z) * individualProgress + floatZ;
    }

    this.particles.geometry.attributes.position.needsUpdate = true;
  }

  easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  onResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    this.updateParticlePositions();

    if (this.autoRotate) {
      this.targetRotationY += this.autoRotateSpeed;
    }

    const lerpFactor = 0.05;
    this.currentRotationX += (this.targetRotationX - this.currentRotationX) * lerpFactor;
    this.currentRotationY += (this.targetRotationY - this.currentRotationY) * lerpFactor;
    this.currentZoom += (this.targetZoom - this.currentZoom) * lerpFactor;

    if (this.particles) {
      this.particles.rotation.x = this.currentRotationX;
      this.particles.rotation.y = this.currentRotationY;
    }

    if (this.stars) {
      this.stars.rotation.y += 0.0001;
    }

    this.camera.position.z = this.currentZoom;

    this.renderer.render(this.scene, this.camera);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new FaceAssembly();
});
