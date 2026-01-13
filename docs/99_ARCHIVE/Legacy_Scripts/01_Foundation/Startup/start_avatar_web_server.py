#!/usr/bin/env python3
"""
Elysia Avatar Web Server
========================

Combined HTTP + WebSocket server for Elysia Avatar visualization.
- HTTP server on port 8080 for serving avatar.html and static files
- WebSocket server on port 8765 for real-time avatar control

Usage:
    python start_avatar_web_server.py
    python start_avatar_web_server.py --http-port 8080 --ws-port 8765
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AvatarWebServer")


class AvatarHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for serving avatar files"""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve from (repo root)
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)
    
    def do_GET(self):
        """Handle GET requests with VRM validation"""
        # Special handling for VRM files
        if self.path.endswith('.vrm'):
            vrm_path = REPO_ROOT / self.path.lstrip('/')
            
            # Check if VRM file exists
            if not vrm_path.exists():
                logger.error(f"VRM file not found: {vrm_path}")
                self.send_error(404, f"VRM file not found: {self.path}")
                return
            
            # Check if file is readable
            try:
                with open(vrm_path, 'rb') as f:
                    # Try to read first few bytes to verify it's accessible
                    f.read(10)
            except Exception as e:
                logger.error(f"VRM file not readable: {vrm_path}, error: {e}")
                self.send_error(500, f"VRM file not readable: {e}")
                return
            
            # Log successful VRM request
            file_size = vrm_path.stat().st_size / (1024 * 1024)
            logger.info(f"Serving VRM file: {vrm_path.name} ({file_size:.2f} MB)")
        
        # Continue with normal GET handling
        super().do_GET()
    
    def end_headers(self):
        # Add CORS headers for cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Add MIME type for VRM files
        if self.path.endswith('.vrm'):
            self.send_header('Content-Type', 'model/gltf-binary')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Custom logging
        logger.info(f"HTTP: {format % args}")


def start_http_server(port=8080):
    """Start HTTP server in a separate thread"""
    server = HTTPServer(('0.0.0.0', port), AvatarHTTPRequestHandler)
    logger.info(f"🌐 HTTP Server started on http://localhost:{port}")
    logger.info(f"📂 Serving files from: {REPO_ROOT}")
    logger.info(f"🎭 Avatar page: http://localhost:{port}/Core/Creativity/web/avatar.html")
    server.serve_forever()


async def start_websocket_server(host='0.0.0.0', port=8765):
    """Start WebSocket server for avatar control"""
    from Core.Interaction.Interface.avatar_server import AvatarWebSocketServer
    
    server = AvatarWebSocketServer(host, port)
    await server.start()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Elysia Avatar Web Server")
    parser.add_argument('--http-port', type=int, default=8080,
                        help='HTTP server port (default: 8080)')
    parser.add_argument('--ws-port', type=int, default=8765,
                        help='WebSocket server port (default: 8765)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🎭 Elysia Avatar Web Server")
    logger.info("=" * 60)
    
    # Start HTTP server in a separate thread
    http_thread = Thread(target=start_http_server, args=(args.http_port,), daemon=True)
    http_thread.start()
    
    # Start WebSocket server in main thread
    try:
        asyncio.run(start_websocket_server(args.host, args.ws_port))
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down servers...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
