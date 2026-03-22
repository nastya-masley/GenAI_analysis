import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const canvas = document.getElementById('bust-canvas');
if (canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
  camera.position.set(0, 0, 5.5);

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

    // Center and scale the model
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
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  resize();
  window.addEventListener('resize', resize);

  function animate() {
    requestAnimationFrame(animate);
    if (bust) {
      bust.rotation.y += 0.005;
    }
    renderer.render(scene, camera);
  }

  animate();
}
