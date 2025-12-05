
from flask import request, jsonify, redirect, Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import re
import json as _json
import os as _os
import math
import os
import logging

# --- Import project modules ---
from Core.Foundation.tool_executor import ToolExecutor
from integrations.agent_proxy import AgentProxy
from infra.web_sanctum import WebSanctum
from tools.visualize_kg import render_kg, render_placeholder
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Elysia.guardian import Guardian # Import Guardian to access its components

# --- Bridge for cross-module communication ---
from applications import elysia_bridge

# --- App Initialization ---
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
# Initialize SocketIO and associate it with the bridge
socketio = SocketIO(app, cors_allowed_origins="*")
elysia_bridge.socketio = socketio

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Instantiate Core Components ---
# We instantiate Guardian to get access to its fully wired-up components
# This avoids re-instantiating everything.
guardian = Guardian()
cognition_pipeline = guardian.daemon.cognition_pipeline # Use the 'cognition_pipeline' attribute
tool_executor = ToolExecutor()
agent_proxy = AgentProxy()
sanctum = WebSanctum()


# --- WebSocket Event Handlers ---

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handles incoming chat messages from the user."""
    user_input = data.get('message', '')
    logger.info(f"Received chat message: '{user_input}' from {request.sid}")

    try:
        # Process the message through the cognition pipeline
        response, emotional_state = cognition_pipeline.process_message(user_input)

        # Prepare the payload to be sent back to the client
        payload = {}
        resp_type = (response or {}).get('type', 'text')

        if resp_type == 'text':
            payload = {'type': 'text', 'text': response.get('text', '')}
        elif resp_type in ('creative_visualization', 'image'):
            payload = {
                'type': 'image',
                'text': response.get('text', ''),
                'image_path': response.get('image_path') or response.get('image_url', '')
            }
        else:
            payload = {'type': 'text', 'text': str(response)}

        # Emit the response back to the client
        socketio.emit('chat_response', {'response': payload})

    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        socketio.emit('chat_response', {
            'response': {'type': 'text', 'text': 'An error occurred in my thought process.'}
        })

# --- New function to push hypothesis questions to the client ---
def send_hypothesis_to_client(hypothesis):
    """Emits a hypothesis question to all connected clients."""
    if elysia_bridge.socketio:
        logger.info(f"Pushing hypothesis to client: {hypothesis}")
        elysia_bridge.socketio.emit('hypothesis_question', hypothesis)

# Inject this function into the HypothesisHandler for it to use
# This is a form of dependency injection to avoid circular imports
guardian.daemon.cognition_pipeline.hypothesis_handler.ask_user_via_ui = send_hypothesis_to_client

@socketio.on('hypothesis_response')
def handle_hypothesis_response(data):
    """Handles the user's response (approve/deny) to a hypothesis question."""
    hypothesis = data.get('hypothesis')
    response = data.get('response')
    logger.info(f"Received hypothesis response: '{response}' for hypothesis: {hypothesis.get('head')}")

    if not all([hypothesis, response]):
        logger.warning("Invalid hypothesis response received.")
        return

    # To reuse the existing handler logic, we simulate the context it expects.
    from Project_Elysia.architecture.context import ConversationContext

    # Create a temporary context for this single interaction
    temp_context = ConversationContext()
    temp_context.pending_hypothesis = hypothesis

    # Convert 'approve'/'deny' to a natural language equivalent that the handler understands
    message = "응, 허락한다" if response == 'approve' else "아니, 거부한다"

    # Get the current emotional state
    emotional_state = guardian.emotional_engine.get_current_state()

    # Call the handler to process the response
    result = guardian.daemon.pipeline.hypothesis_handler.handle_response(message, temp_context, emotional_state)

    # Send the final confirmation message back to the user
    if result and result.get('text'):
        socketio.emit('chat_response', {'response': {'type': 'text', 'text': result['text']}})


# --- Existing HTTP Endpoints (unchanged) ---

@app.after_request
def _no_cache(resp):
    # ... (no change)
    return resp

@app.route('/world_status')
def world_status():
    world = guardian.cellular_world
    alive_mask = getattr(world, "is_alive_mask", None)
    population = int(alive_mask.sum()) if alive_mask is not None else 0
    time_step = int(getattr(world, "time_step", 0))
    reason = getattr(guardian.daemon.cognition_pipeline, "last_reason", "Waiting for insight")
    angle = (time_step % 360) * (math.pi / 180)
    vector = {"x": math.cos(angle), "y": math.sin(angle)}
    focus_spread = round(1.0 + ((time_step % 6) * 0.1), 2)
    return jsonify({
        "time_step": time_step,
        "population": population,
        "last_reason": reason,
        "focus_vector": vector,
        "focus_spread": focus_spread,
    })

# ... (all other HTTP routes like /tool/decide, /monitor, etc. remain here) ...


# --- Background Task for Guardian's Dream Cycle ---
def run_guardian_dream_cycle():
    """Periodically runs the Guardian's idle cycle to allow for learning and hypothesis generation."""
    import time
    logger.info("Guardian's dream cycle (background task) started.")
    while True:
        try:
            # In a real scenario, you'd check if Elysia is actually 'IDLE'
            # For now, we just run the learning part of the cycle.
            guardian.trigger_learning()
            guardian._process_high_confidence_hypotheses()
        except Exception as e:
            logger.error(f"Error in Guardian's dream cycle: {e}", exc_info=True)

        # Use the learning interval from the Guardian's config
        interval = getattr(guardian, 'learning_interval', 60)
        socketio.sleep(interval)

@socketio.on('connect')
def on_connect_start_dream():
    # Start the dream cycle when the first client connects.
    # The `if not hasattr(app, 'dream_started')` ensures it only starts once.
    if not hasattr(app, 'dream_started'):
        socketio.start_background_task(target=run_guardian_dream_cycle)
        app.dream_started = True
        logger.info("First client connected, starting Guardian's dream cycle.")


# --- Server Execution ---
if __name__ == '__main__':
    logger.info("Starting Flask-SocketIO server.")
    socketio.run(app, debug=True, port=5000)
