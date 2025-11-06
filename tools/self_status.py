import json
import os
from pathlib import Path
from typing import Dict, Any

from tools.bg_control import status as bg_status


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8').strip()
    except Exception:
        return ''


def load_prefs(base: Path) -> Dict[str, Any]:
    try:
        return json.loads((base / 'data' / 'preferences.json').read_text(encoding='utf-8'))
    except Exception:
        return {}


def get_flow_profile(base: Path) -> str:
    p = base / 'data' / 'flows' / 'profile.txt'
    return read_text(p) or 'generic'


def load_activities(base: Path):
    try:
        p = base / 'data' / 'background' / 'activities.json'
        return json.loads(p.read_text(encoding='utf-8')).get('activities', {})
    except Exception:
        return {}


def aggregate() -> Dict[str, Any]:
    base = Path(__file__).resolve().parent.parent
    prefs = load_prefs(base)
    bg = bg_status()
    prof = get_flow_profile(base)
    acts = load_activities(base)
    # high-level flags
    learning_mode = (prof != 'generic') or bool(prefs.get('auto_act'))
    busy = any(rec.get('state') == 'running' for rec in acts.values()) or bool(bg.get('running'))
    return {
        'flow_profile': prof,
        'quiet_mode': bool(prefs.get('quiet_mode', False)),
        'auto_act': bool(prefs.get('auto_act', False)),
        'background': bg,
        'activities': acts,
        'learning_mode': learning_mode,
        'busy': busy,
    }

