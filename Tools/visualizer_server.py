import http.server
import socketserver
import json
import threading
import os
import logging
from typing import Any

logger = logging.getLogger("Visualizer")

class VisualizerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, world=None, **kwargs):
        self.world = world
        # Set directory to Tools/web for static files
        directory = os.path.join(os.path.dirname(__file__), "web")
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        if self.path == '/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if self.world:
                state = self.world.get_state_json()
                self.wfile.write(json.dumps(state).encode())
            else:
                self.wfile.write(json.dumps({}).encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        # Suppress default logging to keep console clean
        pass

class VisualizerServer:
    def __init__(self, world: Any, port: int = 8000):
        self.world = world
        self.port = port
        self.httpd = None
        self.thread = None

    def start(self):
        def handler_factory(*args, **kwargs):
            return VisualizerHandler(*args, world=self.world, **kwargs)

        self.httpd = socketserver.TCPServer(("", self.port), handler_factory)
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True # Kill when main thread dies
        self.thread.start()
        logger.info(f"🔮 The Mirror is active at http://localhost:{self.port}")

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
