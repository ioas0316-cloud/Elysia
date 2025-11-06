from flask import Flask, request, jsonify, render_template, send_from_directory
from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.response_orchestrator import ResponseOrchestrator
from Project_Sophia.conversation_state import WorkingMemory, TopicTracker
from Project_Sophia.config_loader import load_config
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

# --- Offline-first hardening (no external API / no local model download) ---
# Avoid flaky external calls without changing core pipeline code.
try:
    _cfg = load_config()
    # Apply optional prefixes from local config
    _pfx = _cfg.get('prefixes', {}) if isinstance(_cfg, dict) else {}
    if _pfx and hasattr(cognition_pipeline, 'prefixes'):
        cognition_pipeline.prefixes.update(_pfx)
        if 'visual_learning' in _pfx:
            cognition_pipeline.visual_learning_prefix = _pfx['visual_learning']
    cognition_pipeline.api_available = False
    if hasattr(cognition_pipeline, 'use_local_llm'):
        cognition_pipeline.use_local_llm = False
    # Ensure a clean Korean prefix for visual learning
    if hasattr(cognition_pipeline, 'visual_learning_prefix'):
        cognition_pipeline.visual_learning_prefix = '?닿쾬??洹몃젮蹂댁옄:'
    # Disable inquisitive mind external lookup
    if hasattr(cognition_pipeline, 'inquisitive_mind'):
        def _no_external_llm(topic: str) -> str:
            return "吏�湲덉? ?몃? 吏�??議고쉶媛� 鍮꾪솢?깊솕?섏뼱 ?덉뼱?? ?ㅻⅨ 諛⑹떇?쇰줈 媛숈씠 ?앷컖?대낵源뚯슂?"
        cognition_pipeline.inquisitive_mind.ask_external_llm = _no_external_llm
except Exception:
    pass

# Post-override to ensure clean Korean strings regardless of earlier encoding
try:
    if hasattr(cognition_pipeline, 'visual_learning_prefix'):
        cognition_pipeline.visual_learning_prefix = '?닿쾬??洹몃젮蹂댁옄:'
    if hasattr(cognition_pipeline, 'inquisitive_mind'):
        cognition_pipeline.inquisitive_mind.ask_external_llm = (
            lambda topic: "吏�湲덉? ?몃? 吏�??議고쉶媛� 鍮꾪솢?깊솕?섏뼱 ?덉뼱?? ?ㅻⅨ 諛⑹떇?쇰줈 媛숈씠 ?앷컖?대낵源뚯슂?")
except Exception:
    pass

# --- Lightweight conversation state (offline context) ---
wm = WorkingMemory(size=10)
topics = TopicTracker()
orchestrator = ResponseOrchestrator()

# Replace pipeline's internal response generator with offline orchestrator
def _offline_internal_response(message, emotional_state, context):
    try:
        enriched = cognition_pipeline._enrich_context(context or {}, message)
        echo = enriched.get('echo', {}) or {}
        topics.step()
        topics.reinforce_from_echo(echo)
        wm.add('user', message)
        text = orchestrator.generate(message, emotional_state, enriched, wm, topics)
        return {'type': 'text', 'text': text}, emotional_state
    except Exception:
        return {
            'type': 'text',
            'text': (
                '吏�湲덉? ?몃? 紐⑤뜽 ?놁씠 ?앷컖???뺣━?섍퀬 ?덉뼱?? '
                '議곌툑 ??援ъ껜?곸쑝濡?留먯???二쇱떆硫? ?쒓? 媛�吏?寃쏀뿕怨?媛쒕뀗?쇰줈 ?듯빐蹂쇨쾶??'
            )
        }, emotional_state

try:
    cognition_pipeline._generate_internal_response = _offline_internal_response
except Exception:
    pass


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

        # Update conversation state
        try:
            topics.step()
            wm.add('user', user_input)
        except Exception:
            pass

        # Prepare the final response based on the type
        final_response = {}
        response_type = response_data.get('type')

        if response_type == 'text':
            clean_text = response_data.get('text', '')
            # If the text looks like a generic fallback or mojibake, generate a richer offline response
            generic_markers = [
                "I'm sorry, I'm having trouble thinking clearly right now.",
                "An error occurred while I was trying to respond.",
            ]
            has_mojibake = (isinstance(clean_text, str) and ('\ufffd' in clean_text))
            needs_offline = (
                (isinstance(clean_text, str) and ('占? in clean_text or clean_text in generic_markers))
            )
            if has_mojibake:
                needs_offline = True
            if needs_offline:
                try:
                    enriched = cognition_pipeline._enrich_context({}, user_input)  # private but practical
                    echo = enriched.get('echo', {}) or {}
                    topics.reinforce_from_echo(echo)
                    better = orchestrator.generate(user_input, emotional_state, enriched, wm, topics)
                    clean_text = better
                except Exception:
                    # Fallback to a readable Korean message
                    clean_text = (
                        "吏�湲덉? ?몃? 紐⑤뜽 ?놁씠 ?앷컖???뺣━?섍퀬 ?덉뼱?? "
                        "議곌툑 ??援ъ껜?곸쑝濡?留먯???二쇱떆硫? ?쒓? 媛�吏?寃쏀뿕怨?媛쒕뀗?쇰줈 ?듯빐蹂쇨쾶??"
                    )
            final_response = {
                'type': 'text',
                'text': clean_text
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

        print(f"!!! FINAL PAYLOAD TO FRONTEND: {{'response': {final_response}, 'emotional_state': {emotion_dict}}}") # DEBUGGING PRINT
        return jsonify({
            'response': final_response,
            'emotional_state': emotion_dict
        })
    except Exception as e:
        print("!!! elysia_bridge.py ?먯꽌 ?덉쇅 諛쒖깮???ъ갑??!!!") # DEBUGGING PRINT
        app_logger.exception(f"Error processing chat message: {user_input}")
        return jsonify({'error': '諛깆뿏???ㅻ쪟 肄붾뱶: ?뚰뙆-7'}), 500


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
