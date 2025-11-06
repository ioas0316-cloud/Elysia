# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, send_from_directory
import subprocess
import sys
from pathlib import Path
from tools.bg_control import status as bg_status, start_daemon as bg_start, stop_daemon as bg_stop
from tools.self_status import aggregate as self_aggregate
import os
import json
from datetime import datetime
import os
import logging

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.conversation_state import WorkingMemory, TopicTracker
from Project_Sophia.response_orchestrator import ResponseOrchestrator


# --- Logging ---
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

# --- Offline-first hardening (no external API / no local model download) ---
try:
    cognition_pipeline.api_available = False
    if hasattr(cognition_pipeline, 'use_local_llm'):
        cognition_pipeline.use_local_llm = False
    if hasattr(cognition_pipeline, 'visual_learning_prefix'):
        cognition_pipeline.visual_learning_prefix = '이것을 그려보자:'
    if hasattr(cognition_pipeline, 'inquisitive_mind'):
        def _no_external_llm(topic: str) -> str:
            return "지금은 외부 지식 조회가 비활성화되어 있어요. 다른 방식으로 함께 생각해볼까요?"
        cognition_pipeline.inquisitive_mind.ask_external_llm = _no_external_llm
except Exception:
    pass


# --- Lightweight conversation state (offline context) ---
wm = WorkingMemory(size=10)
topics = TopicTracker()
orchestrator = ResponseOrchestrator()


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
                '지금은 모델이 생각을 정리하는 데 어려움이 있어요. '
                '조금 더 구체적으로 말씀해 주시면, 함께 경험과 개념으로 풀어볼게요.'
            )
        }, emotional_state


try:
    cognition_pipeline._generate_internal_response = _offline_internal_response
except Exception:
    pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(os.path.join('..', 'data'), filename)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    if 'message' not in data:
        return jsonify({'error': 'Invalid request. "message" key is required.'}), 400

    user_input = data['message']
    try:
        # Quick self-status intent
        if isinstance(user_input, str) and any(k in user_input for k in ["지금 뭐", "상태", "학습 중", "배경"]):
            st = self_aggregate()
            bg = st.get('background', {})
            acts = st.get('activities', {})
            running = bg.get('running')
            enabled = bg.get('enabled')
            prof = st.get('flow_profile')
            quiet = st.get('quiet_mode')
            auto_act = st.get('auto_act')
            active_names = ", ".join([n for n, rec in acts.items() if rec.get('state')=='running']) or '없음'
            txt = (
                f"지금 상태를 알려드릴게요.\n"
                f"- 배경 학습: {'실행 중' if running else '정지'} / {'활성' if enabled else '비활성'}\n"
                f"- 활동: {active_names}\n"
                f"- 대화 흐름 프로필: {prof} / Quiet={'ON' if quiet else 'OFF'} / 자율={'ON' if auto_act else 'OFF'}"
            )
            return jsonify({'response': {'type': 'text', 'text': txt}, 'emotional_state': None})

        response_data, emotional_state = cognition_pipeline.process_message(user_input)

        # Update conversation state
        try:
            topics.step()
            wm.add('user', user_input)
        except Exception:
            pass

        final_response = {}
        response_type = response_data.get('type')

        if response_type == 'text':
            clean_text = response_data.get('text', '')
            generic_markers = [
                "I'm sorry, I'm having trouble thinking clearly right now.",
                "An error occurred while I was trying to respond.",
            ]
            has_mojibake = (isinstance(clean_text, str) and ('\ufffd' in clean_text or '�' in clean_text))
            needs_offline = (isinstance(clean_text, str) and (clean_text in generic_markers)) or has_mojibake
            if needs_offline:
                try:
                    enriched = cognition_pipeline._enrich_context({}, user_input)
                    echo = enriched.get('echo', {}) or {}
                    topics.reinforce_from_echo(echo)
                    better = orchestrator.generate(user_input, emotional_state, enriched, wm, topics)
                    clean_text = better
                except Exception:
                    clean_text = (
                        '지금은 모델이 생각을 정리하는 데 어려움이 있어요. '
                        '조금 더 구체적으로 말씀해 주시면, 함께 경험과 개념으로 풀어볼게요.'
                    )
            final_response = {'type': 'text', 'text': clean_text}

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
        else:
            final_response = {'type': 'text', 'text': str(response_data)}

        emotion_dict = None
        if emotional_state:
            emotion_dict = {
                'primary_emotion': emotional_state.primary_emotion,
                'secondary_emotions': emotional_state.secondary_emotions,
                'valence': emotional_state.valence,
                'arousal': emotional_state.arousal,
                'dominance': emotional_state.dominance,
            }

        return jsonify({'response': final_response, 'emotional_state': emotion_dict})
    except Exception:
        app_logger.exception(f"Error processing chat message: {user_input}")
        return jsonify({'error': '백엔드 오류 코드: 알파-7'}), 500


@app.route('/bg/status', methods=['GET'])
def bg_get_status():
    try:
        st = bg_status()
        return jsonify(st)
    except Exception as e:
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


def _tail_file(path, max_lines=50):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-max_lines:]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        return []


@app.route('/trace/recent', methods=['GET'])
def trace_recent():
    try:
        # telemetry path: data/telemetry/YYYYMMDD/events.jsonl
        base = os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry')
        day = datetime.utcnow().strftime('%Y%m%d')
        path = os.path.join(base, day, 'events.jsonl')
        events = _tail_file(path, max_lines=100)
        # filter relevant
        filtered = [e for e in events if e.get('event_type') in ('flow.decision','route.arc')]
        return jsonify({'events': filtered})
    except Exception:
        app_logger.exception("Error reading recent trace")
        return jsonify({'events': []})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
