import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const canvas = document.getElementById('bust-canvas');
const toggle = document.getElementById('menu-toggle');

if (canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
  camera.position.set(0, 0, 5.5);

  // Animation targets
  let targetCameraZ = 5.5;
  const INITIAL_Z = 5.5;
  const ACTIVATED_Z = 7.5;

  let bust = null;

  const loader = new OBJLoader();
  loader.load('/assets/models/Bust/mesh_head.obj', (obj) => {
    const wireframeMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.6
    });

    obj.traverse((child) => {
      if (child.isMesh) {
        child.material = wireframeMat;
      }
    });

    const box = new THREE.Box3().setFromObject(obj);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 1.65 / maxDim;
    obj.scale.setScalar(scale);
    obj.position.sub(center.multiplyScalar(scale));

    scene.add(obj);
    bust = obj;
  });

  function resize() {
    const container = canvas.parentElement;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w > 0 && h > 0) {
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }

  resize();
  window.addEventListener('resize', resize);

  const observer = new ResizeObserver(() => resize());
  observer.observe(canvas.parentElement);

  function animate() {
    requestAnimationFrame(animate);
    camera.position.z += (targetCameraZ - camera.position.z) * 0.04;
    if (bust) {
      bust.rotation.y += 0.005;
    }
    renderer.render(scene, camera);
  }

  animate();

  // Expose control
  window.bust3d = {
    activate() { targetCameraZ = ACTIVATED_Z; },
    deactivate() { targetCameraZ = INITIAL_Z; }
  };
}
