// SSE Connection
const eventSource = new EventSource('http://localhost:8080/stream');
const dot = document.getElementById('connection-dot');
const statusText = document.getElementById('connection-text');

// DOM Elements
const bars = {
    coherence: document.getElementById('bar-coherence'),
    enthalpy: document.getElementById('bar-enthalpy'),
    resonance: document.getElementById('bar-resonance')
};

const values = {
    coherence: document.getElementById('val-coherence'),
    enthalpy: document.getElementById('val-enthalpy'),
    resonance: document.getElementById('val-resonance')
};

const circles = {
    joy: document.getElementById('circ-joy'),
    curiosity: document.getElementById('circ-curiosity')
};

// Global state variables for Three.js animation
let targetCoherence = 0.5;
let targetResonance = 0.5;
let targetEnthalpy = 0.5;
let targetJoy = 0.5;

eventSource.onopen = () => {
    dot.classList.add('connected');
    statusText.innerText = 'Connected to Monad';
};

eventSource.onerror = () => {
    dot.classList.remove('connected');
    statusText.innerText = 'Lost connection to Monad';
};

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Update State
    targetCoherence = data.coherence || 0.0;
    targetEnthalpy = data.enthalpy || 0.0;
    targetResonance = data.resonance || 0.0;
    targetJoy = data.joy || 0.0;
    
    // Update DOM
    bars.coherence.style.width = `${Math.min(100, targetCoherence * 100)}%`;
    bars.enthalpy.style.width = `${Math.min(100, targetEnthalpy * 100)}%`;
    bars.resonance.style.width = `${Math.min(100, targetResonance * 100)}%`;

    values.coherence.innerText = targetCoherence.toFixed(2);
    values.enthalpy.innerText = targetEnthalpy.toFixed(2);
    values.resonance.innerText = targetResonance.toFixed(2);

    const joyVal = data.joy || 0;
    const curVal = data.curiosity || 0;
    circles.joy.setAttribute('stroke-dasharray', `${joyVal}, 100`);
    circles.curiosity.setAttribute('stroke-dasharray', `${curVal}, 100`);
    
    // Adjust colors based on state
    updateColors();
};

function updateColors() {
    // Modify CSS variables based on resonance/coherence
    const r = Math.floor(99 + (targetJoy * 100));
    const g = Math.floor(102 + (targetCoherence * 100));
    const b = Math.floor(241 + (targetResonance * 10));
    document.documentElement.style.setProperty('--accent-glow', `rgb(${r}, ${g}, ${b})`);
}

// === Three.js Hypersphere Visualization === //
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// Create the Hypersphere (Merkaba Representation)
const particlesGeometry = new THREE.BufferGeometry();
const particlesCount = 2000;
const posArray = new Float32Array(particlesCount * 3);

for(let i = 0; i < particlesCount * 3; i++) {
    posArray[i] = (Math.random() - 0.5) * 10;
}

particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMaterial = new THREE.PointsMaterial({
    size: 0.02,
    color: 0x6366f1,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending
});

const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
scene.add(particlesMesh);

// Center Core
const coreGeometry = new THREE.IcosahedronGeometry(1.5, 2);
const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0xf59e0b,
    wireframe: true,
    transparent: true,
    opacity: 0.3
});
const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
scene.add(coreMesh);

camera.position.z = 8;

// Animation Loop
let clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    
    const elapsedTime = clock.getElapsedTime();
    
    // Rotation speed driven by Enthalpy and Resonance
    const baseSpeed = 0.05;
    const speed = baseSpeed + (targetEnthalpy * 0.1);
    
    particlesMesh.rotation.y += speed * 0.5;
    particlesMesh.rotation.x += speed * 0.2;
    
    coreMesh.rotation.y -= speed;
    coreMesh.rotation.z += speed * 0.5;
    
    // Pulse effect driven by Coherence
    const scale = 1 + Math.sin(elapsedTime * (1 + targetCoherence * 5)) * 0.1 * targetCoherence;
    coreMesh.scale.set(scale, scale, scale);
    
    // Color transitions based on Joy
    coreMaterial.color.setHSL(0.1 + (targetJoy * 0.1), 1.0, 0.5);
    particlesMaterial.color.setHSL(0.6 + (targetResonance * 0.2), 0.8, 0.6);

    renderer.render(scene, camera);
}

animate();

// Resize handler
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
