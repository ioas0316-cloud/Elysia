from flask import request, jsonify
from flask import redirect
from flask import Flask
from flask import render_template, send_from_directory
from Project_Sophia.tool_executor import ToolExecutor
from integrations.agent_proxy import AgentProxy
from infra.web_sanctum import WebSanctum
from tools.visualize_kg import render_kg, render_placeholder
from tools.kg_manager import KGManager
try:
    from tools.textbook_ingestor import ingest_subject
except Exception:
    ingest_subject = None
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.lens_profile import LensProfile
import math
import os


import atexit

app = Flask(__name__, template_folder='templates')

# --- Elysia Continuity Protocol: Memory Persistence ---
from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.core_memory import CoreMemory

STATE_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'elysia_state.json'))

def load_persistent_memory() -> Optional[CoreMemory]:
    """Loads CoreMemory from the state file if it exists."""
    if os.path.exists(STATE_FILE_PATH):
        print(f"INFO: Found persistent memory state at {STATE_FILE_PATH}. Loading...")
        return CoreMemory(file_path=STATE_FILE_PATH)
    print("INFO: No persistent memory state found. Starting with a fresh memory.")
    return None

# Load memory from file, or start fresh if no file exists.
# Then, inject it into the CognitionPipeline.
persistent_memory = load_persistent_memory()
cognition_pipeline = CognitionPipeline(core_memory=persistent_memory)

# Set the file path for the new memory object if it was created fresh
if not persistent_memory:
    cognition_pipeline.core_memory.file_path = STATE_FILE_PATH

def save_persistent_memory():
    """Saves the current CoreMemory state to the file."""
    print(f"INFO: Application shutting down. Saving memory state to {STATE_FILE_PATH}...")
    cognition_pipeline.core_memory._save_memory()
    print("INFO: Memory state saved.")

# Register the save function to be called on application exit.
atexit.register(save_persistent_memory)
# --- End of Continuity Protocol ---


tool_executor = ToolExecutor()
agent_proxy = AgentProxy()
sanctum = WebSanctum()

# Attempt a one-time initial render so the monitor isn't blank on first load
try:
    # Try full KG render
    # ensure KG has basics before render\n    kgm_boot = KGManager()\n    if not kgm_boot.kg.get('nodes'):\n        try: ingest_subject('geometry_primitives', kgm_boot); ingest_subject('social_interaction', kgm_boot)\n        except Exception: pass\n    render_kg(start_node_id=None, out_name='monitor_kg.png')
    # Place a placeholder echo if none exists yet
    import os as _os
    _echo_path = _os.path.join('data', 'monitor_echo.png')
    if not _os.path.exists(_echo_path):
        render_placeholder('monitor_echo.png', '아직 에코가 없어요 · Render KG를 눌러주세요')
except Exception:
    pass


@app.route('/tool/decide', methods=['POST'])
def tool_decide():
    data = request.get_json(silent=True) or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400
    try:
        decision = cognition_pipeline.action_cortex.decide_action(prompt)
        if not decision:
            return jsonify({'decision': None})
        prepared = tool_executor.prepare_tool_call(decision)
        return jsonify({'decision': prepared})
    except Exception:
        return jsonify({'error': 'decision error'}), 500


@app.route('/tool/execute', methods=['POST'])
def tool_execute():
    data = request.get_json(silent=True) or {}
    decision = data.get('decision')
    if not decision:
        return jsonify({'error': 'decision required'}), 400
    try:
        result = tool_executor.execute_tool(decision)
        # Loop back observation into cognition so Elysia also "sees" the outcome
        try:
            if isinstance(result, dict):
                if 'error' in result:
                    summary = f"tool error: {result.get('error')}"
                elif 'sanitized_text' in result:
                    summary = result.get('sanitized_text','')[:200]
                elif 'content' in result:
                    summary = result.get('content','')[:200]
                elif 'body' in result:
                    summary = result.get('body','')[:200]
                else:
                    import json as _json
                    summary = _json.dumps({k: result[k] for k in list(result)[:4]}, ensure_ascii=False)[:200]
            else:
                summary = str(result)[:200]
            obs = f"{cognition_pipeline.observation_prefix} {summary}"
            cognition_pipeline.process_message(obs)
        except Exception:
            pass
        return jsonify({'result': result})
    except Exception:
        return jsonify({'error': 'execution error'}), 500


@app.route('/agent/proxy', methods=['POST'])
def agent_proxy_route():
    data = request.get_json(silent=True) or {}
    route = data.get('route', '/')
    payload = data.get('payload', {})
    if not agent_proxy.available():
        return jsonify({'error': 'Agent proxy not configured'}), 400
    try:
        result = agent_proxy.call(route, payload)
        return jsonify({'result': result})
    except Exception:
        return jsonify({'error': 'proxy error'}), 500


@app.route('/web/fetch', methods=['POST'])
def web_fetch():
    data = request.get_json(silent=True) or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'url required'}), 400
    try:
        result = sanctum.safe_fetch(url)
        # Loop back observation so Elysia "sees" fetched content summary
        try:
            if isinstance(result, dict):
                if result.get('sanitized_text'):
                    summary = result['sanitized_text'][:200]
                elif result.get('error'):
                    summary = f"web error: {result.get('error')}"
                else:
                    import json as _json
                    summary = _json.dumps({k: result[k] for k in list(result)[:4]}, ensure_ascii=False)[:200]
            else:
                summary = str(result)[:200]
            obs = f"{cognition_pipeline.observation_prefix} {summary}"
            cognition_pipeline.process_message(obs)
        except Exception:
            pass
        # If decision requires confirmation, bubble up the hint
        if result.get('decision') == 'confirm' and not data.get('confirm'):
            return jsonify({'confirm_required': True, 'result': result})
        if result.get('decision') == 'block':
            return jsonify({'blocked': True, 'result': result})
        return jsonify({'result': result})
    except Exception:
        return jsonify({'error': 'fetch error'}), 500


@app.route('/monitor')
def monitor_page():
    # Render template that auto-refreshes image generated by /monitor/echo
    return render_template('monitor.html')


@app.route('/')
def root_redirect():
    # In case browser opens the root URL, send users to the monitor page
    return redirect('/monitor')


# Serve generated images under data/
import os as _os
_DATA_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', 'data'))

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(_DATA_DIR, filename)


# --- Simple Chat UI and API ---
@app.route('/chat-ui')
def chat_ui():
    # Reuse existing chat template if present
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json(silent=True) or {}
    if 'message' not in data:
        return jsonify({'error': 'message required'}), 400
    user_input = data['message']
    try:
        response, emotional_state = cognition_pipeline.process_message(user_input)
        resp_type = (response or {}).get('type') if isinstance(response, dict) else 'text'
        payload = {}
        if resp_type == 'text':
            payload = {
                'response': {
                    'type': 'text',
                    'text': response.get('text', '') if isinstance(response, dict) else str(response)
                }
            }
        elif resp_type in ('creative_visualization', 'image'):
            image_path = response.get('image_path') or response.get('image_url') or ''
            payload = {
                'response': {
                    'type': 'image',
                    'text': response.get('text', ''),
                    'image_path': image_path
                }
            }
        else:
            payload = {'response': {'type': 'text', 'text': str(response)}}

        # include emotional state summary if available
        if emotional_state:
            payload['emotional_state'] = {
                'primary_emotion': getattr(emotional_state, 'primary_emotion', ''),
                'valence': getattr(emotional_state, 'valence', 0.0),
                'arousal': getattr(emotional_state, 'arousal', 0.0)
            }
        return jsonify(payload)
    except Exception:
        return jsonify({'error': 'chat error'}), 500


@app.route('/monitor/kg')
def monitor_kg():
    start = request.args.get('start')
    try:
        path = render_kg(start_node_id=start, out_name='monitor_kg.png')
        url_path = path.replace('\\', '/')
        return jsonify({'image_path': url_path})
    except Exception:
        return jsonify({'error': 'render error'}), 500


@app.route('/monitor/echo')
def monitor_echo():
    try:
        # pick top topic if available
        topics = cognition_pipeline.topic_tracker.snapshot() if hasattr(cognition_pipeline, 'topic_tracker') else {}
        start = next(iter(topics.keys())) if topics else None
        path = render_kg(start_node_id=start, out_name='monitor_echo.png')
        url_path = path.replace('\\', '/')
        # Check existence for easier debugging on the client
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        if not exists:
            # Fallback: try full KG, then placeholder
            try:
                p2 = render_kg(start_node_id=None, out_name='monitor_echo.png')
                abs2 = os.path.abspath(p2)
                exists2 = os.path.exists(abs2)
                if exists2:
                    return jsonify({'image_path': p2.replace('\\', '/'), 'start': start, 'exists': True})
            except Exception:
                pass
            try:
                p3 = render_placeholder('monitor_echo.png', '아직 에코가 없어요 · Render KG를 눌러주세요')
                return jsonify({'image_path': p3.replace('\\', '/'), 'start': start, 'exists': True})
            except Exception:
                pass
        return jsonify({'image_path': url_path, 'start': start, 'exists': exists})
    except Exception:
        return jsonify({'error': 'render error'}), 500


@app.route('/monitor/status')
def monitor_status():
    """
    Returns simple metrics for the monitor panel so users can predict outcomes.
    - start: the node used to seed the echo (top topic if available)
    - echo_radius: average spatial distance of active concepts
    - entropy: diversity of activation (higher means broader focus)
    - top_topics: current top topics from tracker
    - anchors: current anchors used by the spatial lens
    """
    try:
        kgm = KGManager()
        kg = kgm.kg
        topics = cognition_pipeline.topic_tracker.snapshot() if hasattr(cognition_pipeline, 'topic_tracker') else {}
        start = (None if not topics else next(iter(topics.keys())))
        qstart = (request.args.get('start') or '').strip()
        if qstart:
            start = qstart
        wm = WaveMechanics(kgm)
        echo = {}
        if start:
            echo = wm.spread_activation(start)
        else:
            echo = {n['id']: 1.0 for n in kg.get('nodes', [])}

        total = sum(echo.values()) or 1.0
        probs = [v/total for v in echo.values()]
        entropy = -sum(p*math.log(p+1e-12) for p in probs) if probs else 0.0

        # spatial radius
        pos = {n['id']: n.get('position', {'x':0,'y':0,'z':0}) for n in kg.get('nodes', [])}
        cx = sum((pos[k]['x'] if k in pos else 0.0) * (echo[k]/total) for k in echo)
        cy = sum((pos[k]['y'] if k in pos else 0.0) * (echo[k]/total) for k in echo)
        cz = sum((pos[k]['z'] if k in pos else 0.0) * (echo[k]/total) for k in echo)
        dists = []
        zs = []
        for k, e in echo.items():
            if k in pos:
                p = pos[k]
                d = math.sqrt((p['x']-cx)**2 + (p['y']-cy)**2 + (p['z']-cz)**2)
                dists.append(d)
                zs.append(p['z'])
        radius = (sum(dists)/len(dists)) if dists else 0.0
        z_span = (max(zs)-min(zs)) if zs else 0.0

        anchors = LensProfile()._pick_anchors(kg)
        top3 = list(topics.keys())[:3] if topics else []
        last_out = getattr(cognition_pipeline, 'last_output_summary', None)
        return jsonify({
            'start': start,
            'echo_radius': radius,
            'entropy': entropy,
            'top_topics': top3,
            'anchors': anchors,
            'last_output': last_out,
            'z_span': z_span
        })
    except Exception:
        return jsonify({'error': 'status error'}), 500


@app.route('/monitor/force')
def monitor_force():
    """
    Seeds the KG (if empty) and renders KG and Echo images in one go.
    Optional query: start=<node>
    """
    try:
        # Seed textbooks if possible and needed
        if ingest_subject is not None:
            kgm = KGManager()
            if not kgm.kg.get('nodes'):
                try:
                    ingest_subject('geometry_primitives', kgm)
                    ingest_subject('social_interaction', kgm)
                except Exception:
                    pass
        # Render images
        start = (request.args.get('start') or '').strip() or None
        kg_path = render_kg(start_node_id=None, out_name='monitor_kg.png')
        echo_path = render_kg(start_node_id=start, out_name='monitor_echo.png') if start else render_kg(start_node_id=None, out_name='monitor_echo.png')

        kg_abs = os.path.abspath(kg_path)
        echo_abs = os.path.abspath(echo_path)
        return jsonify({
            'kg_image': kg_path.replace('\\', '/'),
            'kg_exists': os.path.exists(kg_abs),
            'echo_image': echo_path.replace('\\', '/'),
            'echo_exists': os.path.exists(echo_abs),
            'start': start
        })
    except Exception:
        return jsonify({'error': 'force error'}), 500


@app.route('/monitor/check')
def monitor_check():
    """
    Returns existence of monitor images and absolute paths to help debugging.
    """
    try:
        kg_rel = os.path.join('data', 'monitor_kg.png')
        echo_rel = os.path.join('data', 'monitor_echo.png')
        kg_abs = os.path.abspath(kg_rel)
        echo_abs = os.path.abspath(echo_rel)
        return jsonify({
            'kg': {'path': kg_rel.replace('\\', '/'), 'abs': kg_abs, 'exists': os.path.exists(kg_abs)},
            'echo': {'path': echo_rel.replace('\\', '/'), 'abs': echo_abs, 'exists': os.path.exists(echo_abs)}
        })
    except Exception:
        return jsonify({'error': 'check error'}), 500
