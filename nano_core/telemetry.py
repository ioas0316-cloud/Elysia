from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Append a JSONL event under data/telemetry/YYYYMMDD/events.jsonl."""
    try:
        base = os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry')
        base = os.path.abspath(base)
        day = datetime.now(datetime.UTC).strftime('%Y%m%d')
        out_dir = os.path.join(base, day)
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, 'events.jsonl')
        rec = {
            'ts': datetime.now(datetime.UTC).isoformat() + 'Z',
            'event_type': event_type,
            'payload': payload,
        }
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        # Best-effort only; never raise
        pass

