import json
import os
from typing import Any, Dict

SANCTUM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Elysia_Input_Sanctum'))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(filename: str, data: Dict[str, Any]):
    _ensure_dir(SANCTUM_DIR)
    path = os.path.join(SANCTUM_DIR, filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_json(filename: str) -> Dict[str, Any]:
    path = os.path.join(SANCTUM_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

