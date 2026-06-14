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
const imagePointMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff }); // 시각 데이터는 파란색
const sphereGeometry = new THREE.SphereGeometry(0.1, 16, 16);

// Material for continuous trajectory lines (역인과 구조의 선)
const lineMaterial = new THREE.LineBasicMaterial({ color: 0x0088ff, transparent: true, opacity: 0.5 });
let trajectoryLines = null;

// Function to handle input
async function submitWords() {
    const text = document.getElementById('wordInput').value;
    if (!text) return;

    await fetch('/api/init_field', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });
    console.log("Text dropped into ingest folder.");
}

async function submitImage() {
    const fileInput = document.getElementById('imageInput');
    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function(event) {
        const base64Str = event.target.result;
        
        await fetch('/api/init_image_field', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: base64Str })
        });
        console.log("Image dropped into ingest folder.");
    };
    reader.readAsDataURL(file);
}

// 주기적으로 공유 메모리의 텐션을 직접 관측합니다.
async function observeMemoryField() {
    try {
        const response = await fetch('/api/observe_field');
        const result = await response.json();
        if (result.status === 'success') {
            renderObservedField(result.data);
        }
    } catch(e) {
        // 서버 다운 시 조용히 넘어감
    }
}

function renderObservedField(data) {
    // Clear previous objects
    pointObjects.forEach(obj => scene.remove(obj.mesh));
    if (trajectoryLines) {
        scene.remove(trajectoryLines);
        trajectoryLines.geometry.dispose();
        trajectoryLines = null;
    }

    pointObjects = [];
    pointsData = data; 
    
    const linePoints = [];

    data.forEach(item => {
        // 텐션 값이 높을수록 다른 색으로 표현 (예: 공간 텐션이 높으면 시안, 언어 텐션이 높으면 그린)
        const isVisual = item.position[2] > item.position[1]; 
        const mat = isVisual ? imagePointMaterial : pointMaterial;
        const mesh = new THREE.Mesh(sphereGeometry, mat);
        
        // 순수한 텐션값(X=math, Y=lang, Z=spatial)을 그대로 위치로 사용합니다! (자연 매핑)
        mesh.position.set(item.position[0], item.position[2], item.position[1]); 
        scene.add(mesh);
        
        linePoints.push(new THREE.Vector3(item.position[0], item.position[2], item.position[1]));

        // Add text label
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
        sprite.position.set(0, 0.3, 0); 
        sprite.scale.set(0.5, 0.5, 0.5);
        mesh.add(sprite);

        pointObjects.push({ mesh: mesh });
    });

    if (linePoints.length > 1) {
        const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
        trajectoryLines = new THREE.Line(lineGeo, lineMaterial);
        scene.add(trajectoryLines);
    }
}

async function submitImage() {
    const fileInput = document.getElementById('imageInput');
    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function(event) {
        const base64Str = event.target.result;
        
        const response = await fetch('/api/init_image_field', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: base64Str })
        });
        const result = await response.json();

        if (result.status === 'success') {
            initializeField(result.data, true); // true = image
        }
    };
    reader.readAsDataURL(file);
}

function initializeField(data, isImage) {
    // Clear previous objects
    pointObjects.forEach(obj => scene.remove(obj.mesh));
    if (trajectoryLines) {
        scene.remove(trajectoryLines);
        trajectoryLines.geometry.dispose();
        trajectoryLines = null;
    }

    pointObjects = [];
    pointsData = isImage ? [...pointsData, ...data] : data; // 이미지는 기존 텍스트(우주) 위에 중첩
    time = 0;

    // 연속성의 선(궤적)을 그리기 위한 배열
    const linePoints = [];

    const targetData = isImage ? data : pointsData;

    targetData.forEach(item => {
        const mat = isImage ? imagePointMaterial : pointMaterial;
        const mesh = new THREE.Mesh(sphereGeometry, mat);
        // Set initial position
        mesh.position.set(item.position[0], item.position[2], item.position[1]); // Swap Y/Z for Three.js
        scene.add(mesh);
        
        linePoints.push(new THREE.Vector3(item.position[0], item.position[2], item.position[1]));

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
            phase: item.phase,
            isImage: isImage
        });
    });

    // 점과 점을 이은 선 (연속성 궤적)
    if (linePoints.length > 1) {
        const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
        trajectoryLines = new THREE.Line(lineGeo, lineMaterial);
        scene.add(trajectoryLines);
    }
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




let isAutoObserve = false;
const autoObserveToggle = document.getElementById('autoObserveToggle');
const joystickPanel = document.getElementById('joystickPanel');

autoObserveToggle.addEventListener('change', (e) => {
    isAutoObserve = e.target.checked;
    if (isAutoObserve) {
        joystickPanel.style.opacity = '0.5';
        qxSlider.disabled = true;
        qySlider.disabled = true;
        qzSlider.disabled = true;
        qwSlider.disabled = true;
    } else {
        joystickPanel.style.opacity = '1.0';
        qxSlider.disabled = false;
        qySlider.disabled = false;
        qzSlider.disabled = false;
        qwSlider.disabled = false;
    }
});

async function autoObserveStep() {
    if (pointsData.length === 0) return;

    const now = Date.now();
    if (now - lastEvaluateTime < EVALUATE_INTERVAL) return;
    lastEvaluateTime = now;

    try {
        const response = await fetch('/api/auto_observe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                points_data: pointsData,
                time_t: time
            })
        });
        const result = await response.json();

        if (result.status === 'success') {
            // Smoothly interpolate to Elysia's chosen quaternion
            const targetQuat = new THREE.Quaternion(
                result.quaternion[0], result.quaternion[1], result.quaternion[2], result.quaternion[3]
            ).normalize();

            // Sync UI sliders visually but don't trigger events
            qxSlider.value = targetQuat.x;
            qySlider.value = targetQuat.y;
            qzSlider.value = targetQuat.z;
            qwSlider.value = targetQuat.w;
            qxVal.innerText = targetQuat.x.toFixed(2);
            qyVal.innerText = targetQuat.y.toFixed(2);
            qzVal.innerText = targetQuat.z.toFixed(2);
            qwVal.innerText = targetQuat.w.toFixed(2);

            userQuat.slerp(targetQuat, 0.1); // Smooth camera movement

            camera.quaternion.copy(userQuat);
            const distance = 10;
            const offset = new THREE.Vector3(0, 0, distance);
            offset.applyQuaternion(userQuat);
            camera.position.copy(offset);
            camera.lookAt(0, 0, 0);

            if (result.is_resonant) {
                equationDisplay.innerText = `[거울 통과]: 화살표가 세상을 향합니다. (${result.formula})`;
                document.body.classList.remove('spike-effect');
                void document.body.offsetWidth;
                document.body.classList.add('spike-effect');
            } else {
                equationDisplay.innerText = `[거울 통과]: 투명한 상태`;
            }
        }
    } catch(err) {
        console.error("Auto observe error:", err);
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
                equationDisplay.innerText = `[상태]: ${result.formula}`;
                // Trigger line clear effect (flash background)
                document.body.classList.remove('spike-effect');
                void document.body.offsetWidth; // trigger reflow
                document.body.classList.add('spike-effect');
            } else {
                equationDisplay.innerText = `[상태]: 대기 중...`;
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
        // 인위적인 math.sin() 애니메이션(trajectory update)이 모두 제거되었습니다.
        // 점들은 오직 C 메모리 텐션값의 실시간 변화에 의해서만 움직입니다.
        if (isAutoObserve) { autoObserveStep(); } else { evaluateState(); }
        
        // [Phase 3] 점들이 살아 숨쉬는 듯한 맥박(Pulsate) 애니메이션 효과 추가
        const currentTime = Date.now() * 0.005;
        pointObjects.forEach(obj => {
            if (obj.mesh && obj.phase !== undefined) {
                const scale = 1.0 + 0.3 * Math.sin(currentTime + obj.phase);
                obj.mesh.scale.set(scale, scale, scale);
            }
        });
    }

    controls.update();
    renderer.render(scene, camera);
}

// 빠른 실시간 관측을 위해 200ms 주기로 공유 메모리 폴링
setInterval(observeMemoryField, 200);

// Handle window resize
window.addEventListener('resize', onWindowResize, false);
function onWindowResize(){
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

animate();
