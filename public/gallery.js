import * as THREE from 'three';

// ============ STATE MANAGEMENT ============
const GalleryState = {
  GLOBE_OUTSIDE: 'GLOBE_OUTSIDE', // Camera outside, viewing globe
  GLOBE_ENTER: 'GLOBE_ENTER',     // Camera enters the globe
  SPHERE_INSIDE: 'SPHERE_INSIDE', // Camera inside sphere, busts around
  MORPHING: 'MORPHING',           // Sphere transforms into tunnel
  TUNNEL: 'TUNNEL',
  DETAIL: 'DETAIL'
};

let currentState = GalleryState.GLOBE_OUTSIDE;
let scrollProgress = 0;
let maxScroll = 8000; // Virtual scroll height - increased for slower animation
let virtualScroll = 0;
let targetScroll = 0;
let selectedBust = null;
let bustsData = [];
let bustMeshes = [];
let isDetailView = false;

// Random offsets for bust animations
const bustRandomOffsets = [];
const bustRandomScales = [];
const bustScalePhases = [];
let animationTime = 0;

// ============ THREE.JS SETUP ============
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.Fog(0x000000, 25, 80); // Adjusted for globe + tunnel views

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 25); // Start outside the globe

const renderer = new THREE.WebGLRenderer({ 
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance'
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(5, 10, 7);
scene.add(directionalLight);

const pointLight = new THREE.PointLight(0xffffff, 0.5);
pointLight.position.set(0, 0, 0);
scene.add(pointLight);

// Raycaster for click detection
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// ============ GEOMETRY AND MATERIALS ============
// Flat geometry for tunnel view
const flatBustGeometry = new THREE.PlaneGeometry(1.2, 1.5);

// Curved geometry for globe view - bends to follow sphere surface
function createCurvedPlaneGeometry(width, height, curveAmount, segments = 16) {
  const geometry = new THREE.PlaneGeometry(width, height, segments, segments);
  const positions = geometry.attributes.position;
  
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i);
    const y = positions.getY(i);
    
    // Curve the plane - push vertices back based on distance from center
    const distFromCenter = Math.sqrt(x * x + y * y);
    const maxDist = Math.sqrt((width/2) * (width/2) + (height/2) * (height/2));
    const normalizedDist = distFromCenter / maxDist;
    
    // Create spherical curve
    const z = -curveAmount * (1 - Math.cos(normalizedDist * Math.PI * 0.5));
    positions.setZ(i, z);
  }
  
  geometry.computeVertexNormals();
  return geometry;
}

const curvedBustGeometry = createCurvedPlaneGeometry(1.2, 1.5, 0.4, 16);
const textureLoader = new THREE.TextureLoader();

// Store original positions for morphing
const spherePositions = [];
const tunnelPositions = [];

// ============ POSITION CALCULATIONS ============

// Fibonacci sphere distribution for globe view - positioned for external viewing
function calculateSpherePositions(count, radius = 8) {
  const positions = [];
  const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle
  
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2; // y goes from 1 to -1
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = phi * i;
    
    positions.push(new THREE.Vector3(
      Math.cos(theta) * radiusAtY * radius,
      y * radius,
      Math.sin(theta) * radiusAtY * radius
    ));
  }
  return positions;
}

// Cylindrical tunnel distribution
function calculateTunnelPositions(count, tunnelRadius = 8, tunnelLength = 80) {
  const positions = [];
  const ringsCount = Math.ceil(count / 12); // 12 busts per ring
  const ringSpacing = tunnelLength / ringsCount;
  
  let index = 0;
  for (let ring = 0; ring < ringsCount && index < count; ring++) {
    const bustsInRing = Math.min(12, count - index);
    const angleStep = (Math.PI * 2) / bustsInRing;
    
    for (let i = 0; i < bustsInRing && index < count; i++) {
      const angle = i * angleStep + (ring % 2) * (angleStep / 2); // Offset alternate rings
      const z = -ring * ringSpacing - 15; // Start ahead of camera
      
      positions.push(new THREE.Vector3(
        Math.cos(angle) * tunnelRadius,
        Math.sin(angle) * tunnelRadius,
        z
      ));
      index++;
    }
  }
  return positions;
}

// ============ BUST CREATION ============

async function createBustMesh(bustData, index) {
  return new Promise((resolve) => {
    textureLoader.load(
      bustData.image_path,
      (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        
        const material = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 1
        });
        
        // Use curved geometry for globe effect
        const mesh = new THREE.Mesh(curvedBustGeometry, material);
        mesh.userData = {
          bustId: bustData.id,
          description: bustData.description,
          imagePath: bustData.image_path,
          index: index,
          curvedGeometry: curvedBustGeometry,
          flatGeometry: flatBustGeometry
        };
        
        resolve(mesh);
      },
      undefined,
      (error) => {
        console.error('Error loading texture:', error);
        // Create placeholder mesh
        const material = new THREE.MeshBasicMaterial({
          color: 0x333333,
          side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(curvedBustGeometry, material);
        mesh.userData = {
          bustId: bustData.id,
          description: bustData.description,
          imagePath: bustData.image_path,
          index: index,
          curvedGeometry: curvedBustGeometry,
          flatGeometry: flatBustGeometry
        };
        resolve(mesh);
      }
    );
  });
}

// ============ LOADING INDICATOR ============

function showLoadingIndicator() {
  const loader = document.createElement('div');
  loader.id = 'gallery-loader';
  loader.innerHTML = `
    <div class="loader-content">
      <div class="loader-spinner"></div>
      <p class="loader-text">Loading gallery...</p>
    </div>
  `;
  document.body.appendChild(loader);
}

function updateLoadingProgress(current, total) {
  const text = document.querySelector('#gallery-loader .loader-text');
  if (text) {
    text.textContent = `Loading busts... ${current}/${total}`;
  }
}

function hideLoadingIndicator() {
  const loader = document.getElementById('gallery-loader');
  if (loader) {
    loader.classList.add('fade-out');
    setTimeout(() => loader.remove(), 500);
  }
}

// ============ GALLERY INITIALIZATION ============

async function initGallery() {
  showLoadingIndicator();
  
  // Fetch busts from API
  try {
    const response = await fetch('/api/busts');
    bustsData = await response.json();
  } catch (error) {
    console.error('Failed to fetch busts:', error);
    hideLoadingIndicator();
    return;
  }

  if (bustsData.length === 0) {
    console.log('No busts in database');
    hideLoadingIndicator();
    return;
  }

  // Calculate positions
  const count = bustsData.length;
  spherePositions.length = 0;
  tunnelPositions.length = 0;
  spherePositions.push(...calculateSpherePositions(count));
  spherePositions.push(...calculateTunnelPositions(count));
  tunnelPositions.push(...calculateTunnelPositions(count));

  // Create bust meshes in batches for better loading feedback
  const batchSize = 20;
  for (let i = 0; i < bustsData.length; i += batchSize) {
    const batch = bustsData.slice(i, Math.min(i + batchSize, bustsData.length));
    const meshPromises = batch.map((bust, batchIndex) => 
      createBustMesh(bust, i + batchIndex)
    );
    const batchMeshes = await Promise.all(meshPromises);
    bustMeshes.push(...batchMeshes);
    updateLoadingProgress(bustMeshes.length, bustsData.length);
    
    // Small delay to allow UI to update
    await new Promise(resolve => setTimeout(resolve, 10));
  }

  // Recalculate positions with correct count
  spherePositions.length = 0;
  tunnelPositions.length = 0;
  spherePositions.push(...calculateSpherePositions(bustMeshes.length));
  tunnelPositions.push(...calculateTunnelPositions(bustMeshes.length));

  // Initialize random offsets and scales for each bust
  bustRandomOffsets.length = 0;
  bustRandomScales.length = 0;
  bustScalePhases.length = 0;
  
  bustMeshes.forEach((mesh, index) => {
    // Random horizontal offset for globe phase (range: -3 to 3)
    bustRandomOffsets.push({
      x: (Math.random() - 0.5) * 6,
      y: (Math.random() - 0.5) * 2,
      z: (Math.random() - 0.5) * 3
    });
    
    // Random base scale (0.8 to 1.3) and animation phase
    bustRandomScales.push(0.8 + Math.random() * 0.5);
    bustScalePhases.push(Math.random() * Math.PI * 2);
  });

  // Add meshes to scene and set initial positions with fade-in effect
  bustMeshes.forEach((mesh, index) => {
    const pos = spherePositions[index].clone();
    mesh.position.copy(pos);
    
    // Face outward (camera starts outside the globe)
    const outward = pos.clone().normalize();
    mesh.lookAt(pos.x + outward.x, pos.y + outward.y, pos.z + outward.z);
    
    mesh.material.opacity = 0;
    scene.add(mesh);
  });

  // Fade in all busts
  let fadeProgress = 0;
  const fadeIn = () => {
    fadeProgress += 0.02;
    bustMeshes.forEach((mesh, index) => {
      const delay = index * 0.002;
      const opacity = Math.min(1, Math.max(0, (fadeProgress - delay) * 2));
      mesh.material.opacity = opacity;
    });
    
    if (fadeProgress < 1.5) {
      requestAnimationFrame(fadeIn);
    }
  };
  fadeIn();

  hideLoadingIndicator();
  console.log(`Gallery initialized with ${bustMeshes.length} busts`);
}

// ============ ANIMATION AND RENDERING ============

function lerp(start, end, t) {
  return start + (end - start) * t;
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function updateGallery() {
  // Smooth scroll interpolation
  virtualScroll = lerp(virtualScroll, targetScroll, 0.06);
  scrollProgress = Math.max(0, Math.min(1, virtualScroll / maxScroll));
  
  // Update animation time for random scale effects
  animationTime += 0.016;
  
  if (isDetailView) {
    return;
  }

  // Animation phases:
  // 0-20%: Outside globe, viewing it rotate
  // 20-35%: Camera enters the globe
  // 35-55%: Inside sphere, busts around
  // 55-70%: Morphing to tunnel
  // 70-100%: Tunnel flythrough
  
  if (scrollProgress < 0.2) {
    currentState = GalleryState.GLOBE_OUTSIDE;
    updateGlobeOutside(scrollProgress / 0.2);
  } else if (scrollProgress < 0.35) {
    currentState = GalleryState.GLOBE_ENTER;
    const enterProgress = (scrollProgress - 0.2) / 0.15;
    updateGlobeEnter(enterProgress);
  } else if (scrollProgress < 0.55) {
    currentState = GalleryState.SPHERE_INSIDE;
    const insideProgress = (scrollProgress - 0.35) / 0.2;
    updateSphereInside(insideProgress);
  } else if (scrollProgress < 0.7) {
    currentState = GalleryState.MORPHING;
    const morphProgress = (scrollProgress - 0.55) / 0.15;
    updateMorphing(morphProgress);
  } else {
    currentState = GalleryState.TUNNEL;
    const flyProgress = (scrollProgress - 0.7) / 0.3;
    updateFlythrough(flyProgress);
  }
  
  // Update add bust button visibility - only show in tunnel mode
  const addBustBtn = document.getElementById('add-bust-btn');
  if (addBustBtn) {
    const inTunnelMode = scrollProgress >= 0.7;
    addBustBtn.style.opacity = inTunnelMode ? '1' : '0';
    addBustBtn.style.pointerEvents = inTunnelMode ? 'auto' : 'none';
  }
}

// Stage 1: Outside view - Orbital rings structure (like atom/gyroscope)
function updateGlobeOutside(progress) {
  // Camera outside looking at structure
  camera.position.set(0, 0, 28);
  camera.lookAt(0, 0, 0);
  
  const orbitRadius = 10;
  const numRings = 5; // Number of orbital rings
  const bustsPerRing = Math.ceil(bustMeshes.length / numRings);
  
  bustMeshes.forEach((mesh, index) => {
    // Use flat geometry for cleaner look
    if (mesh.geometry !== flatBustGeometry) {
      mesh.geometry = flatBustGeometry;
    }
    
    // Determine which ring this bust belongs to
    const ringIndex = Math.floor(index / bustsPerRing);
    const posInRing = index % bustsPerRing;
    const totalInThisRing = Math.min(bustsPerRing, bustMeshes.length - ringIndex * bustsPerRing);
    
    // Each ring has different tilt angle
    const ringTiltX = (ringIndex / numRings) * Math.PI * 0.8 - Math.PI * 0.4;
    const ringTiltZ = (ringIndex / numRings) * Math.PI * 0.5 - Math.PI * 0.25;
    
    // Position around the ring
    const angleInRing = (posInRing / totalInThisRing) * Math.PI * 2;
    
    // Each ring rotates at different speed based on scroll
    const ringSpeed = 1 + ringIndex * 0.3;
    const rotationOffset = progress * Math.PI * 2 * ringSpeed + ringIndex * Math.PI * 0.4;
    const finalAngle = angleInRing + rotationOffset;
    
    // Calculate position on ring
    let x = Math.cos(finalAngle) * orbitRadius;
    let y = Math.sin(finalAngle) * orbitRadius;
    let z = 0;
    
    // Apply ring tilt
    const pos = new THREE.Vector3(x, y, z);
    pos.applyAxisAngle(new THREE.Vector3(1, 0, 0), ringTiltX);
    pos.applyAxisAngle(new THREE.Vector3(0, 0, 1), ringTiltZ);
    
    // Add subtle floating motion
    const floatPhase = bustScalePhases[index] || index;
    pos.y += Math.sin(animationTime * 0.3 + floatPhase) * 0.3;
    
    mesh.position.copy(pos);
    
    // Face camera
    mesh.lookAt(camera.position);
    
    // Breathing effect
    if (bustScalePhases[index] !== undefined) {
      const phase = bustScalePhases[index];
      const breathingEffect = Math.sin(animationTime * 0.5 + phase) * 0.08;
      mesh.scale.setScalar(0.9 + breathingEffect);
    }
    
    mesh.material.opacity = 1;
  });
}

// Stage 2: Camera enters - transition from orbital rings to sphere
function updateGlobeEnter(progress) {
  const easedProgress = easeInOutCubic(progress);
  
  // Camera moves from outside (28) to inside (0)
  const cameraZ = lerp(28, 0, easedProgress);
  camera.position.set(0, 0, cameraZ);
  camera.lookAt(0, 0, -1);
  
  const orbitRadius = 10;
  const numRings = 5;
  const bustsPerRing = Math.ceil(bustMeshes.length / numRings);
  
  // Rotation for sphere positions
  const sphereRotation = Math.PI + progress * Math.PI * 0.5;
  
  bustMeshes.forEach((mesh, index) => {
    // Calculate orbital ring position (start position)
    const ringIndex = Math.floor(index / bustsPerRing);
    const posInRing = index % bustsPerRing;
    const totalInThisRing = Math.min(bustsPerRing, bustMeshes.length - ringIndex * bustsPerRing);
    
    const ringTiltX = (ringIndex / numRings) * Math.PI * 0.8 - Math.PI * 0.4;
    const ringTiltZ = (ringIndex / numRings) * Math.PI * 0.5 - Math.PI * 0.25;
    const angleInRing = (posInRing / totalInThisRing) * Math.PI * 2;
    const ringSpeed = 1 + ringIndex * 0.3;
    const rotationOffset = Math.PI * 2 * ringSpeed + ringIndex * Math.PI * 0.4; // End rotation from phase 1
    const finalAngle = angleInRing + rotationOffset + progress * Math.PI * 0.5;
    
    let orbX = Math.cos(finalAngle) * orbitRadius;
    let orbY = Math.sin(finalAngle) * orbitRadius;
    let orbZ = 0;
    
    const orbitalPos = new THREE.Vector3(orbX, orbY, orbZ);
    orbitalPos.applyAxisAngle(new THREE.Vector3(1, 0, 0), ringTiltX);
    orbitalPos.applyAxisAngle(new THREE.Vector3(0, 0, 1), ringTiltZ);
    
    // Calculate sphere position (end position)
    const spherePos = spherePositions[index].clone();
    spherePos.applyAxisAngle(new THREE.Vector3(0, 1, 0), sphereRotation);
    
    // Add random offset to sphere position
    if (bustRandomOffsets[index]) {
      const offset = bustRandomOffsets[index];
      spherePos.x += offset.x * easedProgress;
      spherePos.y += offset.y * easedProgress;
      spherePos.z += offset.z * easedProgress;
    }
    
    // Interpolate between orbital and sphere positions
    mesh.position.lerpVectors(orbitalPos, spherePos, easedProgress);
    
    // Face camera
    mesh.lookAt(camera.position);
    
    // Scale transition
    if (bustRandomScales[index] !== undefined && bustScalePhases[index] !== undefined) {
      const randomBaseScale = bustRandomScales[index];
      const phase = bustScalePhases[index];
      const breathingEffect = Math.sin(animationTime * 0.5 + phase) * 0.1;
      const baseScale = lerp(0.9, randomBaseScale, easedProgress);
      mesh.scale.setScalar(baseScale + breathingEffect);
    }
    
    mesh.material.opacity = 1;
  });
}

// Stage 3: Inside sphere, busts rotating around
function updateSphereInside(progress) {
  // Camera at center
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -1);
  
  // Continue rotation
  const baseRotation = Math.PI + Math.PI * 0.5;
  const rotation = baseRotation + progress * Math.PI;
  
  bustMeshes.forEach((mesh, index) => {
    const originalPos = spherePositions[index].clone();
    const rotatedPos = originalPos.clone();
    rotatedPos.applyAxisAngle(new THREE.Vector3(0, 1, 0), rotation);
    
    // Add random offset
    if (bustRandomOffsets[index]) {
      const offset = bustRandomOffsets[index];
      rotatedPos.x += offset.x;
      rotatedPos.y += offset.y;
      rotatedPos.z += offset.z;
    }
    
    mesh.position.copy(rotatedPos);
    mesh.lookAt(0, 0, 0); // Face camera at center
    
    // Random scale animation
    if (bustRandomScales[index] !== undefined && bustScalePhases[index] !== undefined) {
      const baseScale = bustRandomScales[index];
      const phase = bustScalePhases[index];
      const scaleAnimation = Math.sin(animationTime * 0.5 + phase) * 0.15;
      mesh.scale.setScalar(baseScale + scaleAnimation);
    }
    
    mesh.material.opacity = 1;
  });
}

function updateMorphing(progress) {
  const easedProgress = easeInOutCubic(progress);
  
  // Camera moves from center to tunnel entrance
  const cameraZ = lerp(0, 5, easedProgress);
  camera.position.set(0, 0, cameraZ);
  camera.lookAt(0, 0, -10);
  
  // Rotation at end of sphere inside phase
  const sphereRotation = Math.PI + Math.PI * 0.5 + Math.PI; // = 2.5 * PI
  
  bustMeshes.forEach((mesh, index) => {
    // Switch to flat geometry when transitioning to tunnel
    if (easedProgress > 0.5 && mesh.geometry !== flatBustGeometry) {
      mesh.geometry = flatBustGeometry;
    }
    
    // Get sphere position with rotation applied
    const spherePos = spherePositions[index].clone();
    spherePos.applyAxisAngle(new THREE.Vector3(0, 1, 0), sphereRotation);
    
    // Add random offset (fading out)
    if (bustRandomOffsets[index]) {
      const offset = bustRandomOffsets[index];
      spherePos.x += offset.x * (1 - easedProgress);
      spherePos.y += offset.y * (1 - easedProgress);
      spherePos.z += offset.z * (1 - easedProgress);
    }
    
    const tunnelPos = tunnelPositions[index];
    
    // Smoothly interpolate position
    mesh.position.lerpVectors(spherePos, tunnelPos, easedProgress);
    
    // Face camera
    mesh.lookAt(camera.position);
    
    // Scale transitions to 1
    if (bustRandomScales[index] !== undefined) {
      const baseScale = bustRandomScales[index];
      mesh.scale.setScalar(lerp(baseScale, 1, easedProgress));
    } else {
      mesh.scale.setScalar(1);
    }
    
    mesh.material.opacity = 1;
  });
}

function updateFlythrough(progress) {
  const tunnelLength = 80;
  const cameraZ = 5 - progress * (tunnelLength - 10);
  
  camera.position.set(0, 0, cameraZ);
  camera.lookAt(0, 0, cameraZ - 10);
  
  bustMeshes.forEach((mesh) => {
    mesh.lookAt(camera.position);
    mesh.scale.setScalar(1);
    
    // Fade out busts behind camera
    const distanceBehind = mesh.position.z - camera.position.z;
    if (distanceBehind > 2) {
      mesh.material.opacity = Math.max(0, 1 - (distanceBehind - 2) / 5);
    } else {
      mesh.material.opacity = 1;
    }
  });
}

// ============ DETAIL VIEW ============

function showDetailView(bustData) {
  isDetailView = true;
  selectedBust = bustData;
  
  // Create overlay
  const overlay = document.createElement('div');
  overlay.id = 'bust-detail-overlay';
  overlay.innerHTML = `
    <div class="detail-content">
      <button class="detail-close" id="detail-close-btn">&times;</button>
      <div class="detail-image-container">
        <img src="${bustData.imagePath}" alt="Bust sculpture" class="detail-image" />
      </div>
      <div class="detail-info">
        <h2>Bust Sculpture</h2>
        <p class="detail-description">${bustData.description}</p>
        <button class="detail-back-btn" id="detail-back-btn">Back to Gallery</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(overlay);
  
  // Animate in
  requestAnimationFrame(() => {
    overlay.classList.add('visible');
  });
  
  // Event listeners
  document.getElementById('detail-close-btn').addEventListener('click', hideDetailView);
  document.getElementById('detail-back-btn').addEventListener('click', hideDetailView);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) hideDetailView();
  });
}

function hideDetailView() {
  const overlay = document.getElementById('bust-detail-overlay');
  if (overlay) {
    overlay.classList.remove('visible');
    setTimeout(() => {
      overlay.remove();
      isDetailView = false;
      selectedBust = null;
    }, 300);
  }
}

// ============ EVENT HANDLERS ============

function onScroll(event) {
  if (isDetailView) {
    event.preventDefault();
    return;
  }
  
  // Normalize wheel delta across browsers - reduced sensitivity for slower scroll
  const delta = event.deltaY || event.detail || -event.wheelDelta;
  const scrollSpeed = 0.35; // Slower scroll animation
  targetScroll = Math.max(0, Math.min(maxScroll, targetScroll + delta * scrollSpeed));
}

function onClick(event) {
  if (isDetailView) return;
  
  // Calculate mouse position in normalized device coordinates
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  
  raycaster.setFromCamera(mouse, camera);
  
  const intersects = raycaster.intersectObjects(bustMeshes);
  
  if (intersects.length > 0) {
    const clickedMesh = intersects[0].object;
    showDetailView(clickedMesh.userData);
  }
}

function onMouseMove(event) {
  if (isDetailView) return;
  
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  
  raycaster.setFromCamera(mouse, camera);
  
  const intersects = raycaster.intersectObjects(bustMeshes);
  
  // Reset all hovers
  bustMeshes.forEach(mesh => {
    if (mesh.material.emissive) {
      mesh.material.emissive.setHex(0x000000);
    }
    mesh.scale.setScalar(1);
  });
  
  // Highlight hovered bust
  if (intersects.length > 0) {
    const hoveredMesh = intersects[0].object;
    hoveredMesh.scale.setScalar(1.1);
    renderer.domElement.style.cursor = 'pointer';
  } else {
    renderer.domElement.style.cursor = 'default';
  }
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// ============ ANIMATION LOOP ============

function animate() {
  requestAnimationFrame(animate);
  
  updateGallery();
  renderer.render(scene, camera);
}

// ============ PUBLIC API ============

export async function init(container) {
  container.appendChild(renderer.domElement);
  
  // Event listeners
  window.addEventListener('wheel', onScroll, { passive: false });
  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('click', onClick);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  
  // Initialize gallery
  await initGallery();
  
  // Start animation loop
  animate();
  
  return {
    addBust,
    getState: () => currentState,
    getScrollProgress: () => scrollProgress
  };
}

export async function addBust(bustData) {
  // Add new bust to the scene
  const index = bustMeshes.length;
  const mesh = await createBustMesh(bustData, index);
  
  // Recalculate positions
  bustsData.push(bustData);
  const count = bustsData.length;
  spherePositions.length = 0;
  tunnelPositions.length = 0;
  spherePositions.push(...calculateSpherePositions(count));
  tunnelPositions.push(...calculateTunnelPositions(count));
  
  // Add random offset and scale for the new bust
  bustRandomOffsets.push({
    x: (Math.random() - 0.5) * 6,
    y: (Math.random() - 0.5) * 2,
    z: (Math.random() - 0.5) * 3
  });
  bustRandomScales.push(0.8 + Math.random() * 0.5);
  bustScalePhases.push(Math.random() * Math.PI * 2);
  
  // Set position based on current state
  if (currentState === GalleryState.GLOBE_OUTSIDE || currentState === GalleryState.GLOBE_ENTER || 
      currentState === GalleryState.SPHERE_INSIDE || currentState === GalleryState.MORPHING) {
    mesh.position.copy(spherePositions[index]);
    const outward = mesh.position.clone().normalize();
    mesh.lookAt(mesh.position.x + outward.x, mesh.position.y + outward.y, mesh.position.z + outward.z);
  } else {
    mesh.position.copy(tunnelPositions[index]);
    mesh.lookAt(camera.position);
  }
  
  bustMeshes.push(mesh);
  scene.add(mesh);
  
  console.log(`Added new bust: ${bustData.id}`);
}

export function cleanup() {
  window.removeEventListener('wheel', onScroll);
  window.removeEventListener('resize', onResize);
  renderer.domElement.removeEventListener('click', onClick);
  renderer.domElement.removeEventListener('mousemove', onMouseMove);
  
  bustMeshes.forEach(mesh => {
    mesh.geometry.dispose();
    mesh.material.dispose();
    if (mesh.material.map) mesh.material.map.dispose();
  });
  
  renderer.dispose();
}
