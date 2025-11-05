import json
import os
from typing import Any, Dict, Optional, List
from infra.telemetry import Telemetry


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


def validate_config(config: Dict[str, Any], telemetry: Optional[Telemetry] = None) -> Dict[str, Any]:
    """Validates and lightly coerces config structure; emits warnings via telemetry.

    - Ensures filesystem config has expected types; drops invalid values.
    - Unknown keys are allowed but reported.
    """
    try:
        cfg = dict(config) if isinstance(config, dict) else {}
        fs = cfg.get('filesystem')
        if fs is None:
            return cfg
        if not isinstance(fs, dict):
            if telemetry:
                telemetry.emit('config.warn', {'section': 'filesystem', 'issue': 'not_a_dict'})
            cfg['filesystem'] = {}
            return cfg

        fs_cfg = dict(fs)
        # Known keys and type expectations
        known = {
            'enabled': bool,
            'root': str,
            'read_only': bool,
            'allowed_exts': list,
            'ignore_globs': list,
            'max_file_mb': int,
            'hash_algo': (str, type(None)),
            'auto_index_on_start': bool,
            'save_index': (str, type(None)),
        }
        unknown = [k for k in fs_cfg.keys() if k not in known]
        if unknown and telemetry:
            telemetry.emit('config.warn', {'section': 'filesystem', 'issue': 'unknown_keys', 'keys': unknown})

        # Coerce/validate fields
        def _bool(key: str) -> Optional[bool]:
            v = fs_cfg.get(key)
            if isinstance(v, bool):
                return v
            return bool(v) if v is not None else None

        def _list_str(key: str) -> Optional[List[str]]:
            v = fs_cfg.get(key)
            if v is None:
                return None
            if isinstance(v, list):
                return [str(x) for x in v]
            if telemetry:
                telemetry.emit('config.warn', {'section': 'filesystem', 'issue': f'{key}_not_list'})
            return None

        out: Dict[str, Any] = {}
        out['enabled'] = _bool('enabled') if fs_cfg.get('enabled') is not None else False
        if isinstance(fs_cfg.get('root'), str):
            out['root'] = fs_cfg['root']
        else:
            if telemetry:
                telemetry.emit('config.warn', {'section': 'filesystem', 'issue': 'root_missing_or_invalid'})
        ro = _bool('read_only')
        if ro is not None:
            out['read_only'] = ro
        ae = _list_str('allowed_exts')
        if ae is not None:
            out['allowed_exts'] = ae
        ig = _list_str('ignore_globs')
        if ig is not None:
            out['ignore_globs'] = ig
        try:
            if fs_cfg.get('max_file_mb') is not None:
                out['max_file_mb'] = int(fs_cfg.get('max_file_mb'))
        except Exception:
            if telemetry:
                telemetry.emit('config.warn', {'section': 'filesystem', 'issue': 'max_file_mb_invalid'})
        ha = fs_cfg.get('hash_algo')
        if isinstance(ha, str) or ha is None:
            out['hash_algo'] = ha
        ai = _bool('auto_index_on_start')
        if ai is not None:
            out['auto_index_on_start'] = ai
        si = fs_cfg.get('save_index')
        if isinstance(si, str) or si is None:
            out['save_index'] = si

        cfg['filesystem'] = out
        return cfg
    except Exception:
        return config
