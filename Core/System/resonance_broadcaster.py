import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

class SSEHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Start streaming
            broadcaster = self.server.broadcaster
            broadcaster.clients.append(self)
            try:
                while True:
                    # Keep connection alive
                    time.sleep(1)
            except:
                broadcaster.clients.remove(self)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Disable default logging to keep terminal clean
        pass

class ResonanceBroadcaster:
    """[Phase 800] Streams Elysia's internal state to the UI via Server-Sent Events."""
    def __init__(self, port=8080):
        self.port = port
        self.clients = []
        self.server = None
        self.thread = None

    def start(self):
        try:
            self.server = HTTPServer(('localhost', self.port), SSEHandler)
            self.server.broadcaster = self
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            print(f"🌟 [Arcadia] Resonance Broadcaster started on http://localhost:{self.port}/stream")
        except Exception as e:
            print(f"⚠️ [Arcadia] Broadcaster failed to start: {e}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def broadcast_state(self, state_data):
        """Sends JSON state to all connected clients."""
        if not self.clients:
            return
            
        data_str = f"data: {json.dumps(state_data)}\n\n"
        for client in list(self.clients):
            try:
                client.wfile.write(data_str.encode('utf-8'))
                client.wfile.flush()
            except:
                if client in self.clients:
                    self.clients.remove(client)

_instance = None
def get_broadcaster():
    global _instance
    if _instance is None:
        _instance = ResonanceBroadcaster()
    return _instance
