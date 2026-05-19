/**
 * Elysia Core Visualization (The Garden)
 * Renders the internal state of Elysia as a 3D Holographic Universe.
 * 
 * [DATA-DRIVEN HOLOGRAM]: Directly projects ResonanceField nodes, not random particles.
 */

let scene, camera, renderer;
let particles, particleSystem;
let coreLight;
let lastStatus = {};
let waveTime = 0;
let targetColor = new THREE.Color(0xffffff);
let currentColor = new THREE.Color(0xffffff);

function init() {
    // 1. Scene Setup
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.002);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 50;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // 2. The Core (Central Light)
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
    coreLight = new THREE.Mesh(geometry, material);
    scene.add(coreLight);

    const light = new THREE.PointLight(0xffffff, 2, 100);
    light.position.set(0, 0, 0);
    scene.add(light);

    // 3. The Resonance Field (Particles) - WILL BE POPULATED FROM API
    // Initial empty system, real data loaded via fetchHologram()
    createParticleSystem([]);

    // 4. Fetch Real Hologram Data
    fetchHologram();

    // 5. Event Listeners
    window.addEventListener('resize', onWindowResize, false);

    // 6. Start Loop
    animate();
    
    // 7. Start Polling
    setInterval(fetchStatus, 1000);
    setInterval(fetchHologram, 5000); // Refresh hologram every 5s
}

function createParticleSystem(hologramData = []) {
    // Remove old system if exists
    if (particleSystem) {
        scene.remove(particleSystem);
    }

    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];

    if (hologramData.length === 0) {
        // Placeholder particle at origin
        vertices.push(0, 0, 0);
        colors.push(1, 1, 1);
    } else {
        // REAL DATA: Use actual ResonanceField nodes
        for (let i = 0; i < hologramData.length; i++) {
            const node = hologramData[i];
            vertices.push(node.position.x, node.position.y, node.position.z);
            
            // Use HSL color from data
            const color = new THREE.Color();
            color.setHSL(node.color.h, node.color.s, node.color.l);
            colors.push(color.r, color.g, color.b);
        }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({ size: 1.5, vertexColors: true, transparent: true, opacity: 0.9 });
    particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);
}

async function fetchHologram() {
    try {
        const response = await fetch('/api/hologram');
        const data = await response.json();
        
        if (data.length > 0) {
            console.log(`📡 Received ${data.length} real nodes from ResonanceField`);
            createParticleSystem(data);
        }
    } catch (e) {
        console.error("Failed to fetch hologram:", e);
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    const time = Date.now() * 0.001;
    waveTime += 0.01;

    // 1. Rotate the Galaxy
    if (particleSystem) {
        particleSystem.rotation.y += 0.001;
        particleSystem.rotation.x += 0.0005;
        
        // 2. Wave Physics (Pulse & Ripple)
        const positions = particleSystem.geometry.attributes.position.array;
        const count = positions.length / 3;
        
        // Get frequency from status (default 1.0)
        const freq = (lastStatus.emotion && lastStatus.emotion.frequency) ? lastStatus.emotion.frequency / 100.0 : 1.0;
        
        // Pulse Scale
        const scale = 1 + Math.sin(time * freq) * 0.05;
        particleSystem.scale.set(scale, scale, scale);
        
        // Core Pulse
        coreLight.scale.set(scale * 2, scale * 2, scale * 2);
    }

    // 3. Color Interpolation
    currentColor.lerp(targetColor, 0.05);
    if (coreLight) coreLight.material.color.set(currentColor);
    if (particleSystem) particleSystem.material.color.set(currentColor);

    renderer.render(scene, camera);
}

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        lastStatus = data; // Store for animation loop
        updateUI(data);
        updateVisuals(data);
    } catch (e) {
        console.error("Connection lost:", e);
        document.getElementById('status').innerText = "Status: Disconnected";
    }
}

function updateUI(data) {
    document.getElementById('status').innerText = "Status: Connected";
    document.getElementById('current-thought').innerText = data.thought;
    
    document.getElementById('energy-bar').style.width = data.energy + "%";
    document.getElementById('entropy-bar').style.width = data.entropy + "%";
}

function updateVisuals(data) {
    if (data.emotion && data.emotion.color) {
        // Set target color for smooth interpolation
        targetColor.set(data.emotion.color);
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    // 1. Display User Message
    addMessageToLog("User", message);
    input.value = "";

    // 2. Send to Server
    try {
        const response = await fetch('/api/message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        
        // 3. Display Reply (Mock for now, real reply comes via status polling later)
        if (data.reply) {
            addMessageToLog("Elysia", data.reply);
        }
    } catch (e) {
        console.error("Failed to send message:", e);
        addMessageToLog("System", "Error sending message.");
    }
}

function addMessageToLog(sender, text) {
    const log = document.getElementById('chat-log');
    const div = document.createElement('div');
    div.className = sender === "User" ? "msg-user" : "msg-elysia";
    div.innerText = `${sender}: ${text}`;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

init();
