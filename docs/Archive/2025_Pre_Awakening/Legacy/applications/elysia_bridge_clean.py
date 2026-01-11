from __future__ import annotations

from flask import Flask, request, jsonify, render_template, send_from_directory
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Core.FoundationLayer.Foundation.conversation_state import WorkingMemory, TopicTracker
from Core.FoundationLayer.Foundation.response_orchestrator import ResponseOrchestrator
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.scheduler import Scheduler
from nano_core.message import Message
import shlex
from nano_core.bots.linker import LinkerBot
from nano_core.bots.validator import ValidatorBot
from nano_core.bots.summarizer import SummarizerBot
from nano_core.bots.composer import ComposerBot
from nano_core.bots.explainer import ExplainerBot
from nano_core.intent_gate import interpret as interpret_intent
from nano_core.intent_gate_ko import interpret as interpret_intent_ko


def _validate_act(verb: str, slots: dict) -> tuple[bool, str]:
    verb = (verb or '').lower()
    s = slots or {}
    missing = []
    if verb in ('link', 'verify'):
        if not s.get('subject'): missing.append('subject=...')
        if not s.get('object'): missing.append('object=...')
    elif verb in ('summarize', 'summary'):
        if not s.get('target'): missing.append('target=...')
    elif verb in ('compose',):
        if not s.get('a'): missing.append('a=...')
        if not s.get('b'): missing.append('b=...')
    elif verb in ('explain',):
        if not s.get('target'): missing.append('target=...')
        if not s.get('text'): missing.append('text="..."')
    if missing:
        example = {
            'link': "nano: link subject=concept:a object=concept:b",
            'verify': "nano: verify subject=concept:a object=concept:b",
            'summarize': "nano: summarize target=concept:x",
            'compose': "nano: compose a=concept:a b=concept:b",
            'explain': "nano: explain target=concept:x text=\"...\"",
        }.get(verb, 'nano: link subject=concept:a object=concept:b')
        hint = f"Missing slots for {verb}: {', '.join(missing)}\nExample: {example}"
        return False, hint
    return True, ''
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

    # Nano command(s): "nano: verb key=value ... ; verb2 key=value ..."
    try:
        text = user_input.strip() if isinstance(user_input, str) else ''
        if text.lower().startswith('nano:'):
            cmd_str = text[5:].strip()
            commands = [c.strip() for c in cmd_str.split(';') if c.strip()]
            bus = MessageBus()
            reg = ConceptRegistry()
            bots = [LinkerBot(), ValidatorBot(), SummarizerBot(), ComposerBot(), ExplainerBot()]
            sched = Scheduler(bus, reg, bots)
            info = []
            for cmd in commands:
                # robust tokenization: support quotes ("text with spaces")
                try:
                    parts = shlex.split(cmd)
                except Exception:
                    parts = [p for p in cmd.split() if p]
                verb = parts[0].lower() if parts else 'link'
                kv = {}
                for p in parts[1:]:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        v = v.strip()
                        # array syntax: [a,b,c]
                        if v.startswith('[') and v.endswith(']'):
                            inner = v[1:-1]
                            kv[k.strip()] = [t.strip() for t in inner.split(',') if t.strip()]
                        else:
                            kv[k.strip()] = v
                if verb in ('link', 'verify'):
                    subj = kv.get('subject') or kv.get('subj') or ''
                    obj = kv.get('object') or kv.get('obj') or ''
                    rel = kv.get('rel', 'related_to')
                    slots = {'subject': subj, 'object': obj, 'rel': rel}
                    ok, msg = _validate_act(verb, slots)
                    if not ok:
                        return jsonify({'response': {'type': 'text', 'text': msg}})
                    bus.post(Message(verb=verb, slots=slots, strength=1.0, ttl=3))
                    info.append(f"{verb} subject={subj} object={obj} rel={rel}")
                elif verb in ('summarize', 'summary'):
                    tgt = kv.get('target') or kv.get('tgt') or ''
                    ok, msg = _validate_act('summarize', {'target': tgt})
                    if not ok:
                        return jsonify({'response': {'type': 'text', 'text': msg}})
                    bus.post(Message(verb='summarize', slots={'target': tgt}, strength=0.8, ttl=2))
                    info.append(f"summarize target={tgt}")
                elif verb in ('compose',):
                    # allow arrays: a=[x,y] → multiple compose
                    a_vals = kv.get('a') if isinstance(kv.get('a'), list) else [kv.get('a') or '']
                    b_vals = kv.get('b') if isinstance(kv.get('b'), list) else [kv.get('b') or '']
                    rel2 = kv.get('rel2', '')
                    for a in a_vals:
                        for b in b_vals:
                            if not a or not b:
                                continue
                            slots = {'a': a, 'b': b}
                            if rel2:
                                slots['rel2'] = rel2
                            ok, msg = _validate_act('compose', slots)
                            if not ok:
                                return jsonify({'response': {'type': 'text', 'text': msg}})
                            bus.post(Message(verb='compose', slots=slots, strength=0.9, ttl=2))
                            info.append(f"compose a={a} b={b} rel2={rel2}")
                elif verb in ('explain',):
                    tgt = kv.get('target') or ''
                    textv = kv.get('text') or ''
                    # support multi evidence targets: evidence=[concept:a,concept:b]
                    ev = kv.get('evidence')
                    slots = {'target': tgt, 'text': textv}
                    if ev:
                        slots['evidence'] = ev
                    ok, msg = _validate_act('explain', slots)
                    if not ok:
                        return jsonify({'response': {'type': 'text', 'text': msg}})
                    bus.post(Message(verb='explain', slots=slots, strength=0.8, ttl=2))
                    info.append(f"explain target={tgt}")
                else:
                    info.append(f"unknown:{verb}")
            processed = sched.step(max_steps=100)
            text_out = "Nano done: " + "; ".join([s for s in info if s]) + f" (processed {processed})"
            return jsonify({'response': {'type': 'text', 'text': text_out}, 'emotional_state': None})
    except Exception:
        pass

    # Intent/DSL gate: natural language → nano messages (EN→KO order)
    try:
        utext = user_input if isinstance(user_input, str) else ''
        acts = interpret_intent(utext) or interpret_intent_ko(utext)
        if acts:
            bus = MessageBus()
            reg = ConceptRegistry()
            bots = [LinkerBot(), ValidatorBot(), SummarizerBot(), ComposerBot(), ExplainerBot()]
            sched = Scheduler(bus, reg, bots)
            info = []
            for a in acts:
                ok, msg = _validate_act(a['verb'], a.get('slots', {}))
                if not ok:
                    return jsonify({'response': {'type': 'text', 'text': msg}})
                bus.post(Message(verb=a['verb'], slots=a.get('slots', {}), strength=0.9, ttl=2))
                # brief
                info.append(f"{a['verb']}")
            processed = sched.step(max_steps=50)
            txt = "Intent mapped: " + ", ".join(info) + f" (processed {processed})"
            return jsonify({'response': {'type': 'text', 'text': txt}, 'emotional_state': None})
    except Exception:
        pass

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
        filtered = [e for e in events if e.get('event_type') in (
            'flow.decision', 'route.arc', 'bus.message', 'bot.run', 'concept.update', 'concept.summary'
        )]
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
