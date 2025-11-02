from flask import Flask, request, jsonify, render_template, send_from_directory
from Project_Sophia.cognition_pipeline import CognitionPipeline
import os
import logging # Import logging module

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'elysia_bridge_errors.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also log to console for immediate feedback
    ]
)
app_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

app = Flask(__name__, template_folder='templates')
cognition_pipeline = CognitionPipeline()


@app.route('/')
def index():
    """Serves the main chat interface."""
    return render_template('index.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serves generated images from the data directory."""
    return send_from_directory(os.path.join('..', 'data'), filename)


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests."""
    data = request.get_json()
    if not data or 'message' not in data:
        app_logger.error("Invalid chat request: 'message' key is missing.")
        return jsonify({'error': 'Invalid request. "message" key is required.'}), 400

    user_input = data['message']
    try:
        response_data, emotional_state = cognition_pipeline.process_message(user_input)

        # Prepare the final response based on the type
        final_response = {}
        response_type = response_data.get('type')

        if response_type == 'text':
            final_response = {
                'type': 'text',
                'text': response_data.get('text', '')
            }
        elif response_type == 'creative_visualization':
            image_path = response_data.get('image_path', '')
            image_url = image_path.replace(os.sep, '/')
            if not image_url.startswith('data/'):
                image_url = 'data/' + os.path.basename(image_url)
            
            final_response = {
                'type': 'image',
                'text': response_data.get('text', ''),
                'image_url': image_url
            }
        elif response_type == 'tool_code':
             # Assuming tool_code response is handled here
            final_response = response_data
        else:
            # Fallback for unknown types
            final_response = {'type': 'text', 'text': str(response_data)}


        emotion_dict = None
        if emotional_state:
            emotion_dict = {
                'primary_emotion': emotional_state.primary_emotion,
                'secondary_emotions': emotional_state.secondary_emotions,
                'valence': emotional_state.valence,
                'arousal': emotional_state.arousal,
                'dominance': emotional_state.dominance
            }

        return jsonify({
            'response': final_response,
            'emotional_state': emotion_dict
        })
    except Exception as e:
        app_logger.exception(f"Error processing chat message: {user_input}")
        return jsonify({'error': 'An internal server error occurred during chat processing.'}), 500


@app.route('/journal')
def journal():
    """Serves the journal page."""
    try:
        entries = cognition_pipeline.journal_cortex.get_entries(limit=20)
        return render_template('journal.html', entries=reversed(entries))
    except Exception as e:
        app_logger.exception("Error fetching journal entries.")
        return "An error occurred while trying to load the journal.", 500


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)