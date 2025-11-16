from __future__ import annotations

import json
import os
from typing import Dict, Any

from .layers import LAYERS


def save_view_state(path: str, camera_state: Dict[str, Any], show_only_selected: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "camera": camera_state,
        "layers": dict(LAYERS),
        "show_only_selected": bool(show_only_selected),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_view_state(path: str) -> Dict[str, Any]:
    default = {
        "camera": {"pos": [0.0, 0.0], "zoom": 1.0},
        "layers": dict(LAYERS),
        "show_only_selected": False,
    }
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "camera" in data and "pos" in data["camera"] and "zoom" in data["camera"]:
            return data
        return default
    except Exception:
        return default

