# [Genesis: 2025-12-02] Purified by Elysia
import json
import os
from typing import Any, Dict


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        # If malformed, ignore and return empty
        return {}


def load_config() -> Dict[str, Any]:
    """Loads base config.json and overlays config.local.json if present."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_cfg = _read_json(os.path.join(base_dir, 'config.json'))
    local_cfg = _read_json(os.path.join(base_dir, 'config.local.json'))

    # Shallow merge: local overrides base
    merged = dict(base_cfg)
    for k, v in local_cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            # Shallow merge nested dicts
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged
