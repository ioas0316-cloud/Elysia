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
        // 카메라 시점 이동 (관측점 이동)
        // MVA에서는 카메라의 위치/회전을 쿼터니언에 맞춰 부드럽게(혹은 즉시) 이동시킵니다.
        const targetQuat = new THREE.Quaternion(
            result.quaternion[0],
            result.quaternion[1],
            result.quaternion[2],
            result.quaternion[3]
        );

        // 카메라를 타겟 쿼터니언으로 회전
        camera.quaternion.copy(targetQuat);
        // 원점에서 일정 거리 떨어진 곳에 카메라 위치
        const distance = 10;
        const offset = new THREE.Vector3(0, 0, distance);
        offset.applyQuaternion(targetQuat);
        camera.position.copy(offset);

        // LookAt 원점
        camera.lookAt(0, 0, 0);
        controls.update();

        document.getElementById('equation-display').innerText = `수식(역기록): ${result.formula}`;
        console.log("Resonance Variance:", result.variance);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    if (isAnimating && pointObjects.length > 0) {
        time += 0.05;
        // Simple spiral trajectory update based on velocity and phase
        pointObjects.forEach((obj) => {
            // Very basic movement for MVA visualizing "Velocity/Wave"
            // x = base_x + vx * t + sin(phase + t)
            const newX = obj.basePos[0] + Math.sin(time + obj.phase) * 0.5 + obj.vel[0] * time * 0.1;
            const newY = obj.basePos[1] + Math.cos(time + obj.phase) * 0.5 + obj.vel[1] * time * 0.1;
            const newZ = obj.basePos[2] + obj.vel[2] * time; // Z moves according to jongsung tension

            // Map to Three.js coordinates (X, Z, Y)
            obj.mesh.position.set(newX, newZ, newY);
        });
    }

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
