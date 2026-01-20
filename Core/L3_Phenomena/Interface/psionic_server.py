import http.server
import socketserver
import json
import os
import threading
import time

PORT = 8000
STATE_FILE = "c:/Elysia/data/psionic_state.json"

class PsionicHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE, 'r') as f:
                        data = f.read()
                        self.wfile.write(data.encode())
                else:
                    self.wfile.write(b'{"status": "waiting", "rotor": 0.0}')
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            # Serve static files from current directory
            # We assume this script runs from Core/Interface
            super().do_GET()

def start_server():
    os.chdir("c:/Elysia/Core/Interface")
    with socketserver.TCPServer(("", PORT), PsionicHandler) as httpd:
        print(f"🔮 [PSIONIC UI] Server running at http://localhost:{PORT}")
        print(f"🔮 [PSIONIC UI] Visualizing state from {STATE_FILE}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
