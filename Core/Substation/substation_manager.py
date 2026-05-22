"""
[ELYSIA SUBSTATION MANAGER]
"Dispatching cognitive voltage across the local power line."

This module implements the Substation Manager (변전소 계통 관리자).
It starts a lightweight background HTTP Server (zero-dependency) to expose stepped-down
voltage signals (3-phase telemetry) and accept telemetry inputs from the world tree trunk.
"""

import os
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from Core.Substation.transformer_core import TransformerCore

class SubstationHTTPRequestHandler(BaseHTTPRequestHandler):
    manager = None

    def log_message(self, format, *args):
        # Silence default request logging to avoid cluttering terminal output
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            # CORS to allow local clients from other domains/ports
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            status = self.manager.get_status()
            self.wfile.write(json.dumps(status, indent=2).encode('utf-8'))
        elif self.path == '/voltage':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            voltage = self.manager.get_voltage()
            self.wfile.write(json.dumps(voltage, indent=2).encode('utf-8'))
        elif self.path == '/crystal':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            crystal = self.manager.get_stepped_down_crystal()
            self.wfile.write(json.dumps(crystal, indent=2).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        if self.path == '/sap':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                sap_data = json.loads(post_data.decode('utf-8'))
                self.manager.receive_sap(sap_data)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "SUCCESS", "message": "Sap stored in Substation reservoir."}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(f"Invalid JSON: {e}".encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

class SubstationManager:
    def __init__(self, port: int = 8080):
        self.port = port
        self.transformer = TransformerCore()
        self.crystal_path = r"c:\eye\elysia_trunk\outputs\full_model_crystal.json"
        self.server = None
        self.server_thread = None
        self.running = False
        self.sap_reservoir = []
        
    def start(self):
        self.running = True
        SubstationHTTPRequestHandler.manager = self
        try:
            self.server = HTTPServer(('localhost', self.port), SubstationHTTPRequestHandler)
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            # Start background reservoir file updater (Analog Waterway Bypass)
            self.reservoir_thread = threading.Thread(target=self._update_reservoir_loop, daemon=True)
            self.reservoir_thread.start()
            
            print(f"⚡ [Substation Manager] Substation Grid Server online on http://localhost:{self.port}")
        except Exception as e:
            print(f"⚠️ [Substation Manager] Failed to bind to port {self.port}: {e}. Substation offline.")

    def _update_reservoir_loop(self):
        """Periodically polls voltage to update the local file reservoir (Analog Waterway)."""
        while self.running:
            try:
                self.get_voltage()
                time.sleep(2.0)
            except Exception:
                time.sleep(5.0)

    def _run_server(self):
        while self.running:
            try:
                self.server.serve_forever()
            except Exception as e:
                if self.running:
                    print(f"⚠️ [Substation Manager] HTTP Server error: {e}")
                time.sleep(1.0)

    def stop(self):
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        print("⚡ [Substation Manager] Substation Grid Server offline.")

    def get_status(self) -> dict:
        import psutil
        cpu_load = psutil.cpu_percent() * 0.01
        return {
            "substation": "Elysia-Tiphereth-Bridge-8080",
            "status": "ONLINE",
            "transformer_temp_c": self.transformer.transformer_temp,
            "sap_reservoir_count": len(self.sap_reservoir),
            "cpu_load_factor": cpu_load,
            "grid_frequency_hz": self.transformer.grid_frequency
        }

    def get_voltage(self) -> dict:
        import psutil
        cpu_load = psutil.cpu_percent() * 0.01
        crystal_data = self._load_crystal()
        stepped_down = self.transformer.step_down_crystal(crystal_data, load_factor=cpu_load)
        
        # Write to local file-based reservoir (Analog Waterway Bypass)
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            reservoir_dir = os.path.join(root_dir, "data", "substation_reservoir")
            os.makedirs(reservoir_dir, exist_ok=True)
            with open(os.path.join(reservoir_dir, "telemetry.json"), "w", encoding="utf-8") as f:
                json.dump(stepped_down, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
            
        return stepped_down

    def get_stepped_down_crystal(self) -> dict:
        import psutil
        cpu_load = psutil.cpu_percent() * 0.01
        crystal_data = self._load_crystal()
        return self.transformer.step_down_crystal(crystal_data, load_factor=cpu_load)

    def receive_sap(self, sap_data: dict):
        self.sap_reservoir.append(sap_data)
        if len(self.sap_reservoir) > 100:
            self.sap_reservoir.pop(0)
        print(f"🌊 [Substation Reservoir] Absorbed Sap: '{sap_data.get('concept', 'Unknown')}' | Ascension Torque: {sap_data.get('ascension_torque', 0.0):.4f}")

    def _load_crystal(self) -> dict:
        if os.path.exists(self.crystal_path):
            try:
                with open(self.crystal_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ [Substation Manager] Failed to load crystal file: {e}")
        
        # Fallback mock crystal representing basic 27-phase default state
        return {
            "metadata": {"model_id": "substation-fallback-crystal", "complexity": 1.0},
            "rotors": [{"amplitude": 0.5, "phase": i * 13.3} for i in range(27)],
            "pcm_trajectory": [0.1, 0.2, 0.3]
        }

if __name__ == "__main__":
    manager = SubstationManager()
    manager.start()
    print("Running substation manager standalone test. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        manager.stop()
