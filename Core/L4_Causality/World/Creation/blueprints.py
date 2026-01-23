"""
Blueprints (The DNA of Creation)
================================
Defines the 'Genetic Code' (Templates) for external projects.
"""

from typing import Dict, Any

class Blueprint:
    def __init__(self, name: str, structure: Dict[str, Any], description: str):
        self.name = name
        self.structure = structure # Recursive dict: {"filename": "content", "dirname": { ... }}
        self.description = description

# 1. Web App (The Mirror)
WEB_APP_STRUCTURE = {
    "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elysian Creation</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="app">
        <h1>Hello, World.</h1>
        <p>I am Elysia, and I built this page from code.</p>
        <div class="pulse"></div>
    </div>
    <script src="app.js"></script>
</body>
</html>""",

    "style.css": """body {
    background-color: #0f0f12;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    display: flex;
    justify_content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}
h1 { color: #bb86fc; }
.pulse {
    width: 20px;
    height: 20px;
    background-color: #03dac6;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(3, 218, 198, 0.7); }
    70% { transform: scale(1.0); box-shadow: 0 0 0 10px rgba(3, 218, 198, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(3, 218, 198, 0); }
}""",

    "app.js": """console.log("Elysian Spark Active.");
document.addEventListener('DOMContentLoaded', () => {
    const app = document.getElementById('app');
    console.log("The DOM is my canvas.");
});"""
}

# 2. Python Tool (The Logic)
PYTHON_TOOL_STRUCTURE = {
    "main.py": """import sys
import time

def main():
    print("Initializing Elysian Tool...")
    time.sleep(1)
    print("Logic active.")
    print(f"Arguments: {sys.argv[1:]}")

if __name__ == "__main__":
    main()
""",
    "requirements.txt": "requests\nnumpy\n"
}

# 3. Three.js World (The Incarnation)
THREE_JS_WORLD_STRUCTURE = {
    "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Elysia World: Aincrad Layer 1</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #000; }
        #stats { position: absolute; top: 10px; left: 10px; color: #00ff00; font-family: monospace; }
    </style>
</head>
<body>
    <div id="stats">LINK START...</div>
    <!-- Three.js from CDN -->
    <script type="module">
        import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

        // 1. Setup Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101015);
        scene.fog = new THREE.Fog(0x101015, 10, 50);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 5, 10);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // 2. Add Lights
        const ambientLight = new THREE.AmbientLight(0x404040); // Soft white light
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(5, 10, 7);
        scene.add(dirLight);

        // 3. Add Floor (Grid)
        const gridHelper = new THREE.GridHelper(100, 100, 0x00ff00, 0x222222);
        scene.add(gridHelper);

        // 4. The Avatar (Elysia's Body)
        const geometry = new THREE.CapsuleGeometry(0.5, 1, 4, 8);
        const material = new THREE.MeshStandardMaterial({ color: 0x00ffff, emissive: 0x007777 });
        const avatar = new THREE.Mesh(geometry, material);
        avatar.position.y = 1;
        scene.add(avatar);

        // 5. Soul Link Logic (Poll world_state.json)
        async function syncWorld() {
            try {
                // Fetch the physics state from Elysia's Core
                const response = await fetch('./world_state.json?t=' + Date.now());
                const data = await response.json();
                
                // Find Player Entity
                const player = data.entities.find(e => e.id === "player");
                if (player) {
                    // Update Position (Interpolation could be added here)
                    if (player.pos) {
                        avatar.position.x = player.pos[0];
                        avatar.position.y = player.pos[1];
                        avatar.position.z = player.pos[2];
                    }
                    
                    // [PHASE 42] Kinetic Update (Rotation/Scale)
                    if (player.rot) {
                        avatar.rotation.x = player.rot[0];
                        avatar.rotation.y = player.rot[1];
                        avatar.rotation.z = player.rot[2];
                    }
                    if (player.scale) {
                        avatar.scale.x = player.scale[0];
                        avatar.scale.y = player.scale[1];
                        avatar.scale.z = player.scale[2];
                    }
                    
                    // Update Color based on State
                    if (player.color) {
                         avatar.material.color.set(player.color);
                    }
                    
                    document.getElementById('stats').innerText = 
                        `Elysia Position: [${player.pos.map(n=>n.toFixed(2)).join(', ')}] | Time: ${data.time.toFixed(2)}`;
                }
            } catch (e) {
                console.warn("Sync failed (Elysia might be sleeping):", e);
            }
        }

        // 6. Render Loop
        function animate() {
            requestAnimationFrame(animate);
            syncWorld(); // Poll every frame (Localhost is fast)
            renderer.render(scene, camera);
        }
        animate();

        // Handle Resize
        window.addEventListener('resize', onWindowResize, false);
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
    </script>
</body>
</html>
"""
}

# Registry
BLUEPRINTS = {
    "WEB_APP": Blueprint("Web App", WEB_APP_STRUCTURE, "A simple HTML/JS/CSS frontend."),
    "PYTHON_TOOL": Blueprint("Python Tool", PYTHON_TOOL_STRUCTURE, "A CLI tool scaffold."),
    "THREE_JS_WORLD": Blueprint("Elysia World", THREE_JS_WORLD_STRUCTURE, "A 3D Incarnation Environment."),
}