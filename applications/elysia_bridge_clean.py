from __future__ import annotations

from flask import Flask, request, jsonify, render_template, send_from_directory
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.conversation_state import WorkingMemory, TopicTracker
from Project_Sophia.response_orchestrator import ResponseOrchestrator
from tools.bg_control import status as bg_status, start_daemon as bg_start, stop_daemon as bg_stop
from tools.self_status import aggregate as self_aggregate


# Logging
log_file_path = os.path.join(os.path.dirname(__file__), 'elysia_bridge_clean_errors.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
app_logger = logging.getLogger(__name__)


app = Flask(__name__, template_folder='templates')
cognition_pipeline = CognitionPipeline()


@app.after_request
def _no_cache(resp):
    try:
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp


# Offline-first hardening (no external API / no local model download)
try:
    cognition_pipeline.api_available = False
    if hasattr(cognition_pipeline, 'use_local_llm'):
        cognition_pipeline.use_local_llm = False
    if hasattr(cognition_pipeline, 'visual_learning_prefix'):
        cognition_pipeline.visual_learning_prefix = 'Draw this:'
    if hasattr(cognition_pipeline, 'inquisitive_mind'):
        def _no_external_llm(topic: str) -> str:
            return "External lookup is disabled. Let's think another way."
        cognition_pipeline.inquisitive_mind.ask_external_llm = _no_external_llm
except Exception:
    pass


# Lightweight conversation state (offline context)
wm = WorkingMemory(size=10)
topics = TopicTracker()
orchestrator = ResponseOrchestrator()


@app.route('/')
def index():
    return render_template('chat-ui.html')


@app.route('/chat-ui')
def chat_ui():
    return render_template('chat-ui.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(os.path.join('..', 'data'), filename)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    if 'message' not in data:
        return jsonify({'error': 'Invalid request. "message" key is required.'}), 400

    user_input = data['message']

    # Early status (ASCII keywords)
    try:
        q = user_input if isinstance(user_input, str) else ''
        if any(k in q for k in ['status', 'background', 'learning']):
            st = self_aggregate()
            bg = st.get('background', {}) if isinstance(st, dict) else {}
            acts = st.get('activities', {}) if isinstance(st, dict) else {}
            running = bool(bg.get('running'))
            enabled = bool(bg.get('enabled'))
            prof = st.get('flow_profile')
            quiet = st.get('quiet_mode')
            auto_act = st.get('auto_act')
            active_names = ", ".join([n for n, rec in acts.items() if isinstance(rec, dict) and rec.get('state') == 'running']) or 'none'
            txt = (
                "Status\n"
                f"- Background: {'running' if running else 'stopped'} / {'enabled' if enabled else 'disabled'}\n"
                f"- Activities: {active_names}\n"
                f"- Flow: {prof} / Quiet={'ON' if quiet else 'OFF'} / Autonomy={'ON' if auto_act else 'OFF'}"
            )
            return jsonify({'response': {'type': 'text', 'text': txt}, 'emotional_state': None})
    except Exception:
        pass

    try:
        response_data, emotional_state = cognition_pipeline.process_message(user_input)

        # Update conversation state
        try:
            topics.step()
            wm.add('user', user_input)
        except Exception:
            pass

        final_response: dict[str, str] = {}
        response_type = (response_data or {}).get('type') if isinstance(response_data, dict) else 'text'

        if response_type == 'text':
            clean_text = response_data.get('text', '') if isinstance(response_data, dict) else str(response_data)
            if not isinstance(clean_text, str):
                clean_text = str(clean_text)
            # Sanitize garbled text
            if '\ufffd' in clean_text:
                clean_text = (
                    "I'm having trouble generating a clean response. "
                    "Could you rephrase or be a bit more specific?"
                )
            final_response = {'type': 'text', 'text': clean_text}

        elif response_type in ('creative_visualization', 'image'):
            image_path = response_data.get('image_path') if isinstance(response_data, dict) else ''
            if not image_path and isinstance(response_data, dict):
                image_path = response_data.get('image_url', '')
            image_url = (image_path or '').replace(os.sep, '/')
            if image_url and not image_url.startswith('data/'):
                image_url = 'data/' + os.path.basename(image_url)
            final_response = {
                'type': 'image',
                'text': response_data.get('text', '') if isinstance(response_data, dict) else '',
                'image_url': image_url
            }
        else:
            final_response = {'type': 'text', 'text': str(response_data)}

        emotion_dict = None
        if emotional_state:
            emotion_dict = {
                'primary_emotion': getattr(emotional_state, 'primary_emotion', ''),
                'secondary_emotions': getattr(emotional_state, 'secondary_emotions', []),
                'valence': getattr(emotional_state, 'valence', 0.0),
                'arousal': getattr(emotional_state, 'arousal', 0.0),
                'dominance': getattr(emotional_state, 'dominance', 0.0),
            }

        return jsonify({'response': final_response, 'emotional_state': emotion_dict})
    except Exception:
        app_logger.exception(f"Error processing chat message: {user_input}")
        return jsonify({'error': 'Backend error code: ALPHA-7'}), 500


@app.route('/bg/status', methods=['GET'])
def bg_get_status():
    try:
        st = bg_status()
        return jsonify(st)
    except Exception:
        app_logger.exception("Error getting background status")
        return jsonify({'error': 'status error'}), 500


@app.route('/bg/on', methods=['POST'])
def bg_on():
    try:
        body = request.get_json(silent=True) or {}
        interval = body.get('interval_sec')
        st = bg_start(interval)
        return jsonify(st)
    except Exception:
        app_logger.exception("Error starting background daemon")
        return jsonify({'error': 'start error'}), 500


@app.route('/bg/off', methods=['POST'])
def bg_off():
    try:
        st = bg_stop()
        return jsonify(st)
    except Exception:
        app_logger.exception("Error stopping background daemon")
        return jsonify({'error': 'stop error'}), 500


@app.route('/self/status', methods=['GET'])
def self_status():
    try:
        st = self_aggregate()
        return jsonify(st)
    except Exception:
        app_logger.exception("Error aggregating self status")
        return jsonify({'error': 'self status error'}), 500


def _tail_file(path: str, max_lines: int = 50):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-max_lines:]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        return []


@app.route('/trace/recent', methods=['GET'])
def trace_recent():
    try:
        base = os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry')
        day = datetime.utcnow().strftime('%Y%m%d')
        path = os.path.join(base, day, 'events.jsonl')
        events = _tail_file(path, max_lines=100)
        filtered = [e for e in events if e.get('event_type') in ('flow.decision', 'route.arc')]
        return jsonify({'events': filtered})
    except Exception:
        app_logger.exception("Error reading recent trace")
        return jsonify({'events': []})


@app.route('/visualize', methods=['POST'])
def visualize():
    body = request.get_json(silent=True) or {}
    concept = (body.get('concept') or '').strip()
    if not concept:
        return jsonify({'error': 'concept required'}), 400
    try:
        from tools.visualize_kg import render_kg
        img_path = render_kg(start_node_id=concept, out_name=None)
        return jsonify({'image_path': img_path.replace(os.sep, '/')})
    except Exception:
        app_logger.exception("Error in visualize")
        return jsonify({'error': 'visualize error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

