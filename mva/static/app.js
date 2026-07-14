// Three.js Scene Setup (Phase 4: Continuous Topological Manifold)
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050505, 0.02);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('canvas-container').appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.set(0, 15, 20);
camera.lookAt(0, 0, 0);
controls.update();

// The Manifold (Continuous Mesh representing Spacetime/Causal Field)
const gridSegments = 64;
const planeGeo = new THREE.PlaneGeometry(30, 30, gridSegments, gridSegments);
planeGeo.rotateX(-Math.PI / 2); // Lay flat on XZ plane

// Use a Wireframe/Mesh combination for a digital fabric look
const planeMat = new THREE.MeshBasicMaterial({ 
    color: 0x38bdf8, 
    wireframe: true, 
    transparent: true, 
    opacity: 0.3 
});
const manifoldMesh = new THREE.Mesh(planeGeo, planeMat);
scene.add(manifoldMesh);

let time = 0;
let pointsData = [];
let isAutoObserve = true; // Auto physics engine loop

async function submitWords() {
    const text = document.getElementById('wordInput').value;
    if (!text) return;
    
    // Server injection
    await fetch('/api/init_field', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });
    
    // Trigger visual ripple at random location on the manifold
    createRipple((Math.random()-0.5)*15, (Math.random()-0.5)*15, 5.0);
    document.getElementById('wordInput').value = '';
}

async function submitImage() {
    const fileInput = document.getElementById('imageInput');
    if (fileInput.files.length === 0) return;
    
    const file = fileInput.files[0];
    const reader = new FileReader();
    reader.onload = async function(event) {
        await fetch('/api/init_image_field', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: event.target.result })
        });
        
        // Massive ripple for image data (cross-modal disruption)
        createRipple(0, 0, 10.0);
    };
    reader.readAsDataURL(file);
}

// Ripple effect on the manifold
let ripples = [];
function createRipple(x, z, strength) {
    ripples.push({ x, z, strength, age: 0 });
}

async function evaluateState() {
    try {
        const response = await fetch('/api/evaluate_state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points_data: pointsData, time_t: time, quaternion: [0,0,0,1] })
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            const tension = result.variance; // now represents topological surface tension
            updateManifoldColor(tension);
        }
    } catch(err) {}
}

async function autoObserveStep() {
    try {
        const response = await fetch('/api/auto_observe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points_data: pointsData, time_t: time })
        });
        const result = await response.json();
        
        if (result.status === 'success') {
            const tension = result.variance;
            updateManifoldColor(tension);
        }
    } catch(err) {}
}

function updateManifoldColor(tension) {
    // High tension = Friction (Red), Low Tension = Equilibrium (Green), Neutral = Blue
    if (tension > 5.0) {
        planeMat.color.setHex(0xf43f5e);
    } else if (tension < 0.5) {
        planeMat.color.setHex(0x10b981);
    } else {
        planeMat.color.setHex(0x38bdf8);
    }
}

// Animation loop (Client-side fluid simulation for visuals)
function animate() {
    requestAnimationFrame(animate);
    time += 0.05;
    
    // Update Manifold Vertices (Simulate Surface Waves)
    const positions = planeGeo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);
        
        // Base ambient wave (vacuum fluctuation)
        let y = Math.sin(x * 0.5 + time) * Math.cos(z * 0.5 + time) * 0.5;
        
        // Add active ripples from injected data
        for (let r = 0; r < ripples.length; r++) {
            const rip = ripples[r];
            const dist = Math.sqrt((x - rip.x)**2 + (z - rip.z)**2);
            // Damped wave equation visualization
            const wave = Math.sin(dist * 2.0 - rip.age * 5.0) * Math.exp(-dist * 0.2 - rip.age * 0.1);
            y += wave * rip.strength;
        }
        
        positions.setY(i, y);
    }
    
    // Age and remove old ripples
    for (let r = ripples.length - 1; r >= 0; r--) {
        ripples[r].age += 0.05;
        if (ripples[r].age > 20) ripples.splice(r, 1);
    }
    
    planeGeo.attributes.position.needsUpdate = true;
    
    controls.update();
    renderer.render(scene, camera);
}

// Periodic server sync for physical tension
setInterval(() => {
    if (isAutoObserve) autoObserveStep();
    else evaluateState();
}, 500);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}, false);

// Make things globally accessible for manifestation UI
window.scene = scene;
window.createRipple = createRipple;

animate();
