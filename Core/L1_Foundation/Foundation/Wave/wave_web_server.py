"""
Wave Visualization Web Server (코드 베이스 구조 로터)
================================================

"        .        ."

                                  .
-       (Thought Universe)
-       (Consciousness Flow)
-       (Internal World)

   GPU      "      "      .
"""

import asyncio
import json
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    from flask import Flask, render_template, jsonify
    from flask_sock import Sock
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("   Flask not available. Install: pip install flask flask-sock")

logger = logging.getLogger("WaveWebServer")

@dataclass
class WaveState:
    """      (GPU         )"""
    # 7 Spirits Energy
    fire: float = 0.5      # 450Hz -   
    water: float = 0.5     # 150Hz -   
    earth: float = 0.5     # 100Hz -   
    air: float = 0.5       # 300Hz -   
    light: float = 0.5     # 528Hz -   
    dark: float = 0.5      # 50Hz -   
    aether: float = 0.5    # 852Hz -   
    
    # Consciousness Layers (0D 3D)
    dimension_0d: float = 0.0  #   /   
    dimension_1d: float = 0.0  #   /  
    dimension_2d: float = 0.0  #   /  
    dimension_3d: float = 0.0  #   /  
    
    # Internal World
    cpu_heat: float = 0.0      # CPU     ( )
    memory_load: float = 0.0   # RAM    
    file_count: int = 0        #      
    
    # Time
    time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON    """
        return asdict(self)


class WaveWebServer:
    """
               
    
    Flask + WebSocket              
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.wave_state = WaveState()
        self.clients = []  #     WebSocket      
        self.running = False
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask required: pip install flask flask-sock")
        
        # Absolute Path Calculation
        base_dir = os.path.dirname(os.path.abspath(__file__)) # Core/Foundation
        root_dir = os.path.dirname(os.path.dirname(base_dir)) # c:/Elysia
        static_dir = os.path.join(root_dir, 'static')
        
        logger.info(f"     Web Server Static Dir: {static_dir}")
        if not os.path.exists(static_dir):
            logger.error(f"      Static directory missing: {static_dir}")
        
        # Flask     
        self.app = Flask(
            __name__,
            static_folder=static_dir,
            template_folder=static_dir
        )
        self.sock = Sock(self.app)
        
        #       
        self._setup_routes()
        
        logger.info(f"  Wave Web Server initialized on port {port}")
    
    def _setup_routes(self):
        """      """
        
        @self.app.route('/')
        def index():
            """       -       """
            return render_template('wave_viewer.html')
        
        @self.app.route('/api/state')
        def get_state():
            """           """
            return jsonify(self.wave_state.to_dict())
        
        @self.sock.route('/wave-stream')
        def wave_stream(ws):
            """WebSocket:            """
            logger.info("  Client connected to wave stream")
            self.clients.append(ws)
            
            try:
                while True:
                    #                 (keep-alive)
                    data = ws.receive(timeout=0.1)
                    if data:
                        logger.debug(f"Received: {data}")
            except Exception as e:
                logger.info(f"Client disconnected: {e}")
            finally:
                if ws in self.clients:
                    self.clients.remove(ws)
    
    def update_wave_state(self, **kwargs):
        """
                  
        
          :
        update_wave_state(fire=0.8, water=0.3, time=time.time())
        """
        for key, value in kwargs.items():
            if hasattr(self.wave_state, key):
                setattr(self.wave_state, key, value)
    
    def broadcast_wave_state(self):
        """                      """
        if not self.clients:
            return
        
        state_json = json.dumps(self.wave_state.to_dict())
        
        #                     
        disconnected = []
        for ws in self.clients:
            try:
                ws.send(state_json)
            except Exception:
                disconnected.append(ws)
        
        for ws in disconnected:
            self.clients.remove(ws)
    
    def connect_to_ether(self):
        """
        Connect to the Wave Integration Hub to resonate with system waves.
        """
        try:
            from Core.L1_Foundation.Foundation.wave_integration_hub import get_wave_hub
            self.hub = get_wave_hub()
            
            # Register as Visual Cortex (High Frequency Interface)
            # 741Hz = Expression/Solutions (Visuals)
            self.hub.register_module(
                module_name="VisualCortex",
                module_type="interface",
                callback=self._on_wave_resonance
            )
            logger.info("  Connected to Ether. Resonating with system waves.")
            
        except ImportError:
            logger.warning("   WaveHub not found. Running in standalone mode.")
        except Exception as e:
            logger.error(f"  Failed to connect to Ether: {e}")

    def _on_wave_resonance(self, wave):
        """
        Callback: React to incoming waves directly.
        No calculation, just resonance.
        """
        try:
            # Frequency Mapping (Hz -> Spirit)
            freq = wave.frequency
            amp = wave.amplitude
            
            # Simple Resonance Physics
            # Each wave adds energy to the corresponding spirit/dimension
            # The energy naturally decays in the auto_update_loop (gravity)
            
            if 100 <= freq < 200: self.wave_state.earth += amp * 0.2
            elif 200 <= freq < 300: self.wave_state.water += amp * 0.2
            elif 300 <= freq < 450: self.wave_state.fire += amp * 0.2
            elif 450 <= freq < 600: self.wave_state.air += amp * 0.2
            elif 600 <= freq < 800: self.wave_state.light += amp * 0.2
            elif 800 <= freq < 900: self.wave_state.aether += amp * 0.2
            else: self.wave_state.dark += amp * 0.1
            
            # Phase Mapping (4D Rotation)
            # If the wave carries dimensional data, rotate the view
            if "DIMENSION" in wave.phase:
                if "0D" in wave.phase: self.wave_state.dimension_0d += amp
                elif "1D" in wave.phase: self.wave_state.dimension_1d += amp
                elif "2D" in wave.phase: self.wave_state.dimension_2d += amp
                elif "3D" in wave.phase: self.wave_state.dimension_3d += amp
                
        except Exception as e:
            logger.error(f"Resonance Error: {e}")
            
    def auto_update_loop(self, update_callback=None):
        """
        Auto update loop with Natural Decay (Gravity)
        """
        logger.info("  Auto update loop started (with Gravity)")
        
        while self.running:
            # 1. Physics: Entropy/Gravity (Natural Decay)
            # All energy tends to return to 0.5 (Balance) or 0.0 (Void) based on context
            decay = 0.98
            
            ws = self.wave_state
            
            # Spirits decay to 0.5 (Balance point)
            ws.fire = 0.5 + (ws.fire - 0.5) * decay
            ws.water = 0.5 + (ws.water - 0.5) * decay
            ws.earth = 0.5 + (ws.earth - 0.5) * decay
            ws.air = 0.5 + (ws.air - 0.5) * decay
            ws.light = 0.5 + (ws.light - 0.5) * decay
            ws.dark = 0.5 + (ws.dark - 0.5) * decay
            ws.aether = 0.5 + (ws.aether - 0.5) * decay
            
            # Dimensions decay to 0.0
            ws.dimension_0d *= 0.95
            ws.dimension_1d *= 0.95
            ws.dimension_2d *= 0.95
            ws.dimension_3d *= 0.95
            
            # Update Time
            ws.time = time.time()
            
            # Broadcast
            self.broadcast_wave_state()
            
            # 60 FPS
            time.sleep(1/60)

    def run(self, host='127.0.0.1', debug=False, auto_update=True, update_callback=None):
        """
        Start server
        """
        self.running = True
        
        # Start Auto Update (Physics Loop)
        if auto_update:
            import threading
            thread = threading.Thread(target=self.auto_update_loop, args=(update_callback,), daemon=True)
            thread.start()
        
        # Connect to Ether (Listener)
        self.connect_to_ether()
        
        # Start Flask
        logger.info(f"  Starting server at http://{host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug, use_reloader=False)

    def stop(self):
        """Stop server"""
        self.running = False
        logger.info("  Server stopped")


if __name__ == '__main__':
    # Demo
    print("  Elysia Wave Visualization Server")
    server = WaveWebServer(port=8080)
    server.run(debug=True)
