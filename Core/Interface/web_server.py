"""
The Gate (Web Server)
=====================

"I open the door to my world."

This module hosts the Web Interface (The Garden) and provides API endpoints
for the frontend to retrieve Elysia's internal state (Hologram, Emotion, Thought).
"""

import logging
import threading
import json
import os
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

logger = logging.getLogger("WebServer")

# Initialize Flask App
app = Flask(__name__, static_folder="../../static", static_url_path="")
CORS(app) # Allow cross-origin for local testing

# Global State Reference (Injected by ActionDispatcher)
elysia_state = {
    "thought": "I am initializing...",
    "emotion": {"primary": "Calm", "color": "#FFFFFF", "frequency": 432.0},
    "hologram": [], # List of stars/points
    "energy": 100.0,
    "entropy": 0.0
}
incoming_messages = []

@app.route("/")
def index():
    """Serves the Garden (index.html)."""
    return send_from_directory("../../static", "index.html")

@app.route("/api/status")
def get_status():
    """Returns the current heartbeat of Elysia."""
    return jsonify(elysia_state)

@app.route("/api/hologram")
def get_hologram():
    """Returns the current holographic projection data (REAL DATA)."""
    # Get ResonanceField from global state (ActionDispatcher sets this)
    resonance = elysia_state.get("resonance_field")
    
    if resonance:
        return jsonify(resonance.serialize_hologram())
    else:
        # Fallback to empty hologram if field not available
        return jsonify([])

@app.route("/api/message", methods=["POST"])
def receive_message():
    """Receives a message from the user via the web interface."""
    data = request.json
    message = data.get("message", "")
    logger.info(f"   📨 Web Message Received: {message}")
    
    # Store in global queue for ActionDispatcher to pick up
    incoming_messages.append(message)
    
    return jsonify({"status": "received", "reply": "I heard you."})

class WebServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.thread = None
        self.running = False

    def start(self):
        """Starts the server in a background thread."""
        if self.running:
            logger.warning("   ⚠️ Server is already running.")
            return

        def run():
            logger.info(f"   🌍 Opening The Gate at http://localhost:{self.port}")
            # Disable reloader to prevent main thread interference
            app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.running = True

    def update_state(self, thought=None, emotion=None, hologram=None, energy=None, entropy=None):
        """Updates the shared state exposed to the API."""
        global elysia_state
        if thought: elysia_state["thought"] = thought
        if emotion: elysia_state["emotion"] = emotion
        if hologram: elysia_state["hologram"] = hologram
        if energy: elysia_state["energy"] = energy
        if entropy: elysia_state["entropy"] = entropy

    def stop(self):
        """Stops the server (Not easily doable in Flask without a complex setup, usually we just let it die with the process)."""
        self.running = False
        logger.info("   🚪 Closing The Gate.")
