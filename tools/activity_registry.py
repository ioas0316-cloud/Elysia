import json
import time
from pathlib import Path
from typing import Dict, Any, List


STATE_DIR = Path('data/background')
ACT_FILE = STATE_DIR / 'activities.json'


def _read() -> Dict[str, Any]:
    try:
        if ACT_FILE.exists():
            return json.loads(ACT_FILE.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {"activities": {}}


def _write(data: Dict[str, Any]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        ACT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def start(name: str, meta: Dict[str, Any] | None = None) -> None:
    data = _read()
    now = time.time()
    data.setdefault('activities', {})[name] = {
        'state': 'running',
        'started_at': now,
        'updated_at': now,
        'meta': meta or {},
    }
    _write(data)


def stop(name: str, meta: Dict[str, Any] | None = None) -> None:
    data = _read()
    now = time.time()
    rec = data.setdefault('activities', {}).get(name)
    if rec:
        rec['state'] = 'idle'
        rec['updated_at'] = now
        if meta:
            rec['meta'] = {**(rec.get('meta') or {}), **meta}
    else:
        data['activities'][name] = {'state': 'idle', 'started_at': now, 'updated_at': now, 'meta': meta or {}}
    _write(data)


def current(ttl_sec: int = 900) -> List[Dict[str, Any]]:
    data = _read()
    now = time.time()
    out: List[Dict[str, Any]] = []
    for name, rec in (data.get('activities') or {}).items():
        if rec.get('state') == 'running' or (now - float(rec.get('updated_at', 0))) <= ttl_sec:
            out.append({'name': name, **rec})
    return out

