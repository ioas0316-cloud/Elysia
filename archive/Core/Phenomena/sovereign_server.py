"""
Sovereign Server: The Cortex
============================
[AEON IX] Visualizing the Mind.

This module provides a real-time WebSocket interface to Elysia's internal state.
It allows the Architect to SEE the Hypersphere rotating and the Concepts lighting up.
"""

import sys
import os
import asyncio
import json
import time
from typing import Dict, Any, List
from threading import Thread

# Third-party imports
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
except ImportError:
    print("⚠️ [SERVER] FastAPI/Uvicorn not installed. Running in Headless Mode.")
    uvicorn = None
    FastAPI = None

# 1. Server Configuration
HOST = "0.0.0.0"
PORT = 8000

class SovereignServer:
    def __init__(self, monad_ref=None):
        self.monad = monad_ref  # Reference to the SovereignMonad instance
        self.app = FastAPI() if FastAPI else None
        self.loop = None
        self.thread = None
        self.clients: List[WebSocket] = []
        
        if self.app:
            self._setup_routes()

    def _setup_routes(self):
        """Configures FastAPI routes."""
        root_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(root_dir, "templates")
        
        # Ensure templates dir exists
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir, exist_ok=True)
            self._create_calm_dashboard(templates_dir)

        templates = Jinja2Templates(directory=templates_dir)

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard(request: Request):
            return templates.TemplateResponse("dashboard.html", {"request": request})

        @self.app.websocket("/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.clients.append(websocket)
            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(1)
            except Exception:
                self.clients.remove(websocket)

    def _create_calm_dashboard(self, templates_dir: str):
        """Creates a minimalistic, beautiful dashboard if it doesn't exist."""
        path = os.path.join(templates_dir, "dashboard.html")
        content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elysia | Sovereign Interface</title>
    <style>
        :root {
            --bg-color: #050505;
            --text-color: #e0e0e0;
            --accent-color: #00ffcc;
            --dim-color: #333;
            --font-main: 'Inter', sans-serif;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: var(--font-main);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        #header {
            padding: 20px;
            border-bottom: 1px solid var(--dim-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { margin: 0; font-weight: 300; letter-spacing: 2px; }
        .status-dot {
            height: 10px; width: 10px; background-color: var(--accent-color);
            border-radius: 50%; display: inline-block; box-shadow: 0 0 10px var(--accent-color);
        }
        #main {
            display: flex;
            flex: 1;
        }
        #sidebar {
            width: 300px;
            border-right: 1px solid var(--dim-color);
            padding: 20px;
            overflow-y: auto;
        }
        #visualization {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        #log-panel {
            height: 200px;
            border-top: 1px solid var(--dim-color);
            padding: 10px;
            font-family: monospace;
            font-size: 0.9em;
            color: #888;
            overflow-y: auto;
            background: #0a0a0a;
        }
        .metric { margin-bottom: 15px; }
        .metric label { display: block; font-size: 0.8em; color: #666; margin-bottom: 5px; }
        .metric value { font-size: 1.2em; font-weight: 500; }
        
        /* The Hypersphere Canvas */
        #hyper-canvas {
            width: 100%; height: 100%;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>ELYSIA <span style="font-size:0.5em; opacity:0.5;">SOVEREIGN INTERFACE</span></h1>
        <div><span class="status-dot"></span> LIVE</div>
    </div>
    
    <div id="main">
        <div id="sidebar">
            <div class="metric">
                <label>PHASE (Rotor)</label>
                <value id="phase-val">0.00 π</value>
            </div>
            <div class="metric">
                <label>ACTIVE ANCHOR</label>
                <value id="anchor-val">None</value>
            </div>
            <div class="metric">
                <label>RESONANCE</label>
                <value id="resonance-val">0.00</value>
            </div>
            <hr style="border-color:var(--dim-color); opacity:0.3;">
            <div class="metric">
                <label>NARRATIVE FOCUS</label>
                <div id="narrative-box" style="font-style: italic; color: var(--accent-color);">
                    ...
                </div>
            </div>
        </div>
        
        <div id="visualization">
            <!-- Provisional Graphic -->
            <canvas id="hyper-canvas"></canvas>
            <div style="position: absolute; pointer-events: none; opacity: 0.5;">
                (21D Hypersphere Projection)
            </div>
        </div>
    </div>

    <div id="log-panel">
        <div id="logs"></div>
    </div>

    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/stream");
        const ctx = document.getElementById('hyper-canvas').getContext('2d');
        let width, height;

        function resize() {
            width = document.getElementById('visualization').clientWidth;
            height = document.getElementById('visualization').clientHeight;
            document.getElementById('hyper-canvas').width = width;
            document.getElementById('hyper-canvas').height = height;
        }
        window.onresize = resize;
        resize();

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // Update Metrics
            if (data.phase !== undefined) {
                document.getElementById('phase-val').innerText = (data.phase / 3.14159).toFixed(2) + " π";
                drawRotor(data.phase);
            }
            if (data.anchor) document.getElementById('anchor-val').innerText = data.anchor;
            if (data.resonance) document.getElementById('resonance-val').innerText = data.resonance.toFixed(3);
            if (data.narrative) document.getElementById('narrative-box').innerText = data.narrative;
            
            // Update Logs
            if (data.log) {
                const logDiv = document.getElementById('logs');
                const p = document.createElement('div');
                p.innerText = "> " + data.log;
                logDiv.insertBefore(p, logDiv.firstChild);
                if (logDiv.children.length > 50) logDiv.lastChild.remove();
            }
        };

        function drawRotor(phase) {
            // Simple visualizer
            ctx.fillStyle = 'rgba(5, 5, 5, 0.1)';
            ctx.fillRect(0, 0, width, height); // Fade effect
            
            const cx = width / 2;
            const cy = height / 2;
            const r = Math.min(width, height) * 0.3;
            
            // Draw Orbit
            ctx.beginPath();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
            
            // Draw Phase Dot
            const x = cx + Math.cos(phase) * r;
            const y = cy + Math.sin(phase) * r;
            
            ctx.beginPath();
            ctx.fillStyle = '#00ffcc';
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#00ffcc';
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
            
            // Draw Central Monad
            ctx.beginPath();
            ctx.fillStyle = '#fff';
            ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    </script>
</body>
</html>
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content.strip())

    def start(self):
        """Starts the Uvicorn server in a separate thread."""
        if not uvicorn: 
            print("⚠️ [SERVER] Uvicorn not available. Skipping web interface.")
            return
        
        def run_uvicorn():
            try:
                print(f"🌐 [SERVER] Starting Sovereign Interface at http://{HOST}:{PORT}")
                # Use a new event loop for this thread to avoid async conflicts
                asyncio.set_event_loop(asyncio.new_event_loop())
                config = uvicorn.Config(app=self.app, host=HOST, port=PORT, log_level="info", loop="asyncio")
                server = uvicorn.Server(config)
                server.run()
            except Exception as e:
                print(f"❌ [SERVER] Failed to start: {e}")
                import traceback
                traceback.print_exc()

        self.thread = Thread(target=run_uvicorn, daemon=True)
        self.thread.start()

    def broadcast(self, data: Dict[str, Any]):
        """Push internal state to all connected clients."""
        if not self.app: return
        
        # We need to run this in the event loop, but we might be in a different thread.
        # Ideally, we put this in a queue, but for now, we'll try a direct loop access if possible.
        # A simpler way for this proof-of-concept is to just skip async complexity and use a lightweight push
        # or have the clients poll. But let's try to be clean.
        
        # ACTUALLY: Uvicorn runs its own loop. We can't easily inject into it from here without complex bridging.
        # HACK: For this "Live Presence" demo, we will use a global queue that the websocket endpoint polls.
        pass

# Global Singleton for easy access (optional)
_server_instance = None

def get_sovereign_server(monad_ref=None):
    global _server_instance
    if _server_instance is None:
        _server_instance = SovereignServer(monad_ref)
    return _server_instance
