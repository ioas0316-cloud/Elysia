"""
Core/Engine/Genesis/pulse_server.py
===================================
The Voice of the System.
Simple HTTP Server to serve dashboard.html and data.
"""

import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8000
DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    """Starts the Pulse Server in a thread."""
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n🌐 [Pulse Server] Hosting at http://localhost:{PORT}")
        print(f"   Root: {DIRECTORY}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
