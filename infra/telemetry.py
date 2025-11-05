import os
import json
import uuid
import shutil
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, Optional


class Telemetry:
    """
    Lightweight JSONL telemetry emitter.
    Writes events to data/telemetry/YYYYMMDD/events.jsonl
    Each line is a single JSON object with a schema_version.
    """

    def __init__(self, base_dir: Optional[str] = None, schema_version: str = "1.0.0"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.base_dir = base_dir or os.path.join(project_root, 'data', 'telemetry')
        self.schema_version = schema_version
        os.makedirs(self.base_dir, exist_ok=True)

    def _path_for_today(self) -> str:
        day = datetime.now(UTC).strftime('%Y%m%d')
        day_dir = os.path.join(self.base_dir, day)
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, 'events.jsonl')

    @staticmethod
    def new_trace_id() -> str:
        return uuid.uuid4().hex

    def emit(self, event_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None):
        try:
            event = {
                'schema_version': self.schema_version,
                'timestamp': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                'event_type': event_type,
                'trace_id': trace_id or self.new_trace_id(),
                'payload': payload,
            }
            path = self._path_for_today()
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Telemetry must never crash the app
            pass

    # --- Retention / housekeeping ---
    def cleanup_retention(self, retain_days: int = 30):
        """
        Keeps only the latest N days of telemetry directories; older ones are
        zipped to save space and the original folders removed.
        """
        try:
            # List day folders under base_dir
            days = []
            for name in os.listdir(self.base_dir):
                dpath = os.path.join(self.base_dir, name)
                if os.path.isdir(dpath) and name.isdigit() and len(name) == 8:
                    days.append(name)
            days.sort()
            if not days:
                return
            cutoff = (datetime.now(UTC) - timedelta(days=retain_days)).strftime('%Y%m%d')
            for day in days:
                if day < cutoff:
                    dpath = os.path.join(self.base_dir, day)
                    archive = os.path.join(self.base_dir, f"{day}.zip")
                    if not os.path.exists(archive):
                        try:
                            shutil.make_archive(os.path.splitext(archive)[0], 'zip', dpath)
                        except Exception:
                            continue
                    # Remove original folder after archiving
                    try:
                        shutil.rmtree(dpath)
                    except Exception:
                        pass
        except Exception:
            pass
