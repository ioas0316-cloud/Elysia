from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from Project_Sophia.cognition_pipeline import CognitionPipeline
import os
import logging

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'elysia_bridge_errors.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
app_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret_elysia_key' # In a real app, use a proper secret key
socketio = SocketIO(app, cors_allowed_origins="*")

cognition_pipeline = CognitionPipeline()

# Define the upload folder and ensure it exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    """Serves the main chat interface."""
    return render_template('index.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serves generated images from the data directory."""
    return send_from_directory(os.path.join('..', 'data'), filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serves uploaded files from the uploads directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    app_logger.info(f"Client connected: {request.sid}")
    emit('response', {'data': 'Elysia is connected.'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    app_logger.info(f"Client disconnected: {request.sid}")


@socketio.on('chat_message')
def handle_chat_message(data):
    """Handles incoming chat messages via WebSocket."""
    user_input = data.get('message', '')
    app_logger.info(f"Received message from {request.sid}: {user_input}")
    if not user_input:
        app_logger.warning("Received empty chat message.")
        return

    try:
        response, emotional_state = cognition_pipeline.process_message(user_input)

        emotion_dict = None
        if emotional_state:
             emotion_dict = {
                'primary_emotion': emotional_state.primary_emotion,
                'secondary_emotions': emotional_state.secondary_emotions,
                'valence': emotional_state.valence,
                'arousal': emotional_state.arousal,
                'dominance': emotional_state.dominance
            }

        emit('chat_response', {
            'response': response,
            'emotional_state': emotion_dict
        })
    except Exception as e:
        app_logger.exception(f"Error processing chat message: {user_input}")
        emit('error', {'error': 'An internal server error occurred.'})


@app.route('/visualize', methods=['POST'])
def visualize():
    """Handles visualization requests."""
    data = request.get_json()
    if not data or 'concept' not in data:
        app_logger.error("Invalid visualize request: 'concept' key is missing.")
        return jsonify({'error': 'Invalid request. "concept" key is required.'}), 400

    concept = data['concept']

    try:
        image_path = cognition_pipeline.sensory_cortex.visualize_concept(concept)
        url_path = image_path.replace(os.sep, '/')
        return jsonify({'image_path': url_path})
    except Exception as e:
        app_logger.exception(f"Error visualizing concept: {concept}")
        return jsonify({'error': 'An internal server error occurred during visualization.'}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        app_logger.info(f"File '{filename}' uploaded successfully to {save_path}")

        # Notify the client via WebSocket
        socketio.emit('file_uploaded', {
            'filename': filename,
            'path': f'/uploads/{filename}'
        })

        # Here you could also trigger a cognitive process for the file
        # cognition_pipeline.process_message(f"I've received a file: {filename}")

        return jsonify({'message': 'File uploaded successfully'}), 200


# This check is essential for running with `python -m`, but was incorrect before.
# Now, we ensure the server runs regardless.
if __name__ == '__main__' or __name__ == 'applications.elysia_bridge':
    socketio.run(app, host='0.0.0.0', port=5000)