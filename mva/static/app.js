// Three.js Scene Setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('canvas-container').appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.set(5, 5, 5);
controls.update();

// Add Grid and Axes
const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
scene.add(gridHelper);
const axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// Global state
let pointsData = [];
let pointObjects = [];
let time = 0;
let isAnimating = true;

// Material for points
const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
const sphereGeometry = new THREE.SphereGeometry(0.1, 16, 16);

// Function to handle input
async function submitWords() {
    const text = document.getElementById('wordInput').value;
    if (!text) return;

    const response = await fetch('/api/init_field', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });
    const result = await response.json();

    if (result.status === 'success') {
        initializeField(result.data);
    }
}

function initializeField(data) {
    // Clear previous objects
    pointObjects.forEach(obj => scene.remove(obj));
    pointObjects = [];
    pointsData = data;
    time = 0;

    data.forEach(item => {
        const mesh = new THREE.Mesh(sphereGeometry, pointMaterial);
        // Set initial position
        mesh.position.set(item.position[0], item.position[2], item.position[1]); // Swap Y/Z for Three.js
        scene.add(mesh);

        // Add text label (simple sprite for MVA)
        const canvas = document.createElement('canvas');
        canvas.width = 64; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.font = '32px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(item.token, 32, 40);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.set(0, 0.3, 0); // slightly above the sphere
        sprite.scale.set(0.5, 0.5, 0.5);
        mesh.add(sprite);

        pointObjects.push({
            mesh: mesh,
            basePos: [...item.position],
            vel: [...item.velocity],
            phase: item.phase
        });
    });
}

async function autoAlign() {
    if (pointsData.length === 0) return;

    document.getElementById('equation-display').innerText = "수식: 초차원 공명 탐색 중...";

    const response = await fetch('/api/auto_align', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points_data: pointsData, time_t: time })
    });
    const result = await response.json();

    if (result.status === 'success') {
        syncSlidersToQuat(result.quaternion);

        document.getElementById('equation-display').innerText = `수식(역기록): ${result.formula}`;
        console.log("Resonance Variance:", result.variance);
    }
}



// Joystick state variables
let userQuat = new THREE.Quaternion(0, 0, 0, 1);
let tickRate = 1.0;
let isManualTime = false;
let lastEvaluateTime = 0;
const EVALUATE_INTERVAL = 200; // debounce requests to backend

// UI bindings
const qxSlider = document.getElementById('qxSlider');
const qySlider = document.getElementById('qySlider');
const qzSlider = document.getElementById('qzSlider');
const qwSlider = document.getElementById('qwSlider');
const qxVal = document.getElementById('qxVal');
const qyVal = document.getElementById('qyVal');
const qzVal = document.getElementById('qzVal');
const qwVal = document.getElementById('qwVal');

const tickRateSlider = document.getElementById('tickRateSlider');
const tickRateVal = document.getElementById('tickRateVal');
const manualTimeSlider = document.getElementById('manualTimeSlider');
const manualTimeVal = document.getElementById('manualTimeVal');
const equationDisplay = document.getElementById('equation-display');

function updateQuatFromUI() {
    let qx = parseFloat(qxSlider.value);
    let qy = parseFloat(qySlider.value);
    let qz = parseFloat(qzSlider.value);
    let qw = parseFloat(qwSlider.value);

    qxVal.innerText = qx.toFixed(2);
    qyVal.innerText = qy.toFixed(2);
    qzVal.innerText = qz.toFixed(2);
    qwVal.innerText = qw.toFixed(2);

    userQuat.set(qx, qy, qz, qw).normalize();

    // Update camera based on joystick
    camera.quaternion.copy(userQuat);
    const distance = 10;
    const offset = new THREE.Vector3(0, 0, distance);
    offset.applyQuaternion(userQuat);
    camera.position.copy(offset);
    camera.lookAt(0, 0, 0);
    controls.update();
}

qxSlider.addEventListener('input', updateQuatFromUI);
qySlider.addEventListener('input', updateQuatFromUI);
qzSlider.addEventListener('input', updateQuatFromUI);
qwSlider.addEventListener('input', updateQuatFromUI);

tickRateSlider.addEventListener('input', (e) => {
    tickRate = parseFloat(e.target.value);
    tickRateVal.innerText = tickRate.toFixed(1);
    isManualTime = false; // Using tick rate restores auto time flow
});

manualTimeSlider.addEventListener('input', (e) => {
    time = parseFloat(e.target.value);
    manualTimeVal.innerText = time.toFixed(1);
    isManualTime = true; // Stop auto time flow
});

// Sync UI sliders with auto align quaternion result
function syncSlidersToQuat(quatArray) {
    qxSlider.value = quatArray[0];
    qySlider.value = quatArray[1];
    qzSlider.value = quatArray[2];
    qwSlider.value = quatArray[3];
    updateQuatFromUI();
}

async function evaluateState() {
    if (pointsData.length === 0) return;

    const now = Date.now();
    if (now - lastEvaluateTime < EVALUATE_INTERVAL) return;
    lastEvaluateTime = now;

    const currentQuatArray = [userQuat.x, userQuat.y, userQuat.z, userQuat.w];

    try {
        const response = await fetch('/api/evaluate_state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                points_data: pointsData,
                time_t: time,
                quaternion: currentQuatArray
            })
        });
        const result = await response.json();

        if (result.status === 'success') {
            if (result.is_resonant) {
                equationDisplay.innerText = `수식(역기록): ${result.formula}`;
                // Trigger line clear effect (flash background)
                document.body.classList.remove('spike-effect');
                void document.body.offsetWidth; // trigger reflow
                document.body.classList.add('spike-effect');
            } else {
                equationDisplay.innerText = `수식: 위상 정렬 대기 중... (분산: ${result.variance.toFixed(2)})`;
            }
        }
    } catch(err) {
        console.error("Evaluate error:", err);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    if (isAnimating && pointObjects.length > 0) {
        if (!isManualTime) {
            time += 0.05 * tickRate;
            manualTimeSlider.value = time % 100; // loop visual
            manualTimeVal.innerText = time.toFixed(1);
        }

        // Simple spiral trajectory update based on velocity and phase
        pointObjects.forEach((obj) => {
            const newX = obj.basePos[0] + Math.sin(time + obj.phase) * 0.5 + obj.vel[0] * time * 0.1;
            const newY = obj.basePos[1] + Math.cos(time + obj.phase) * 0.5 + obj.vel[1] * time * 0.1;
            const newZ = obj.basePos[2] + obj.vel[2] * time;

            // Map to Three.js coordinates (X, Z, Y)
            obj.mesh.position.set(newX, newZ, newY);
        });

        // Continuously evaluate state to see if user found resonance
        evaluateState();
    }

    // Only update controls if not actively overridden by sliders (handled by OrbitControls naturally)
    controls.update();
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', onWindowResize, false);
function onWindowResize(){
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

animate();
