import * as THREE from 'three';

class GlobeGallery {
  constructor() {
    this.canvas = document.getElementById('globe-canvas');
    this.loadingOverlay = document.getElementById('loading');
    this.resetBtn = document.getElementById('reset-btn');

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.globe = null;
    this.atmosphere = null;
    this.stars = null;

    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.currentRotationX = 0;
    this.currentRotationY = 0;
    this.targetZoom = 4;
    this.currentZoom = 4;
    this.autoRotate = true;
    this.autoRotateSpeed = 0.001;

    this.init();
  }

  async init() {
    this.setupThreeJS();
    this.createGlobe();
    this.createAtmosphere();
    this.createStars();
    this.setupLighting();
    this.setupEventListeners();
    this.animate();

    setTimeout(() => {
      this.loadingOverlay.classList.add('hidden');
    }, 1000);
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

  createGlobe() {
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    
    const vertexShader = `
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float time;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vPosition;

      float noise(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
      }

      void main() {
        float lat = vUv.y * 3.14159;
        float lon = vUv.x * 6.28318;
        
        float gridLat = abs(sin(lat * 18.0));
        float gridLon = abs(sin(lon * 36.0));
        
        float grid = max(
          smoothstep(0.95, 1.0, gridLat),
          smoothstep(0.95, 1.0, gridLon)
        );

        float continentNoise = noise(vUv * 8.0 + time * 0.05);
        float continent = smoothstep(0.45, 0.55, continentNoise);

        vec3 oceanColor = vec3(0.02, 0.05, 0.1);
        vec3 landColor = vec3(0.1, 0.15, 0.2);
        vec3 gridColor = vec3(0.2, 0.4, 0.6);

        vec3 baseColor = mix(oceanColor, landColor, continent);
        vec3 finalColor = mix(baseColor, gridColor, grid * 0.6);

        float rim = 1.0 - max(0.0, dot(vNormal, vec3(0.0, 0.0, 1.0)));
        rim = pow(rim, 3.0);
        finalColor += vec3(0.1, 0.3, 0.5) * rim;

        float pulse = sin(time * 2.0) * 0.5 + 0.5;
        float hotspot = noise(vUv * 20.0 + time * 0.1);
        if (hotspot > 0.97) {
          finalColor += vec3(0.3, 0.5, 0.8) * pulse;
        }

        gl_FragColor = vec4(finalColor, 1.0);
      }
    `;

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0 }
      }
    });

    this.globe = new THREE.Mesh(geometry, material);
    this.scene.add(this.globe);
  }

  createAtmosphere() {
    const geometry = new THREE.SphereGeometry(1.15, 64, 64);
    
    const vertexShader = `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      varying vec3 vNormal;
      void main() {
        float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
        vec3 atmosphereColor = vec3(0.1, 0.4, 0.8);
        gl_FragColor = vec4(atmosphereColor, intensity * 0.5);
      }
    `;

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true
    });

    this.atmosphere = new THREE.Mesh(geometry, material);
    this.scene.add(this.atmosphere);
  }

  createStars() {
    const starsGeometry = new THREE.BufferGeometry();
    const starCount = 3000;
    const positions = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);

    for (let i = 0; i < starCount; i++) {
      const radius = 50 + Math.random() * 100;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
      
      sizes[i] = Math.random() * 2;
    }

    starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starsGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const starsMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.5,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true
    });

    this.stars = new THREE.Points(starsGeometry, starsMaterial);
    this.scene.add(this.stars);
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0x333333, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 3, 5);
    this.scene.add(directionalLight);
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
      this.targetZoom = Math.max(2, Math.min(8, this.targetZoom));
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

  resetView() {
    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.targetZoom = 4;
    this.autoRotate = true;
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

    if (this.globe.material.uniforms) {
      this.globe.material.uniforms.time.value = performance.now() * 0.001;
    }

    if (this.autoRotate) {
      this.targetRotationY += this.autoRotateSpeed;
    }

    const lerpFactor = 0.05;
    this.currentRotationX += (this.targetRotationX - this.currentRotationX) * lerpFactor;
    this.currentRotationY += (this.targetRotationY - this.currentRotationY) * lerpFactor;
    this.currentZoom += (this.targetZoom - this.currentZoom) * lerpFactor;

    this.globe.rotation.x = this.currentRotationX;
    this.globe.rotation.y = this.currentRotationY;

    if (this.atmosphere) {
      this.atmosphere.rotation.x = this.currentRotationX;
      this.atmosphere.rotation.y = this.currentRotationY;
    }

    if (this.stars) {
      this.stars.rotation.y += 0.0001;
    }

    this.camera.position.z = this.currentZoom;

    this.renderer.render(this.scene, this.camera);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new GlobeGallery();
});
