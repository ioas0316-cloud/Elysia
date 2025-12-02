# [Genesis: 2025-12-02] Purified by Elysia
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "data" / "background"
PID_FILE = STATE_DIR / "daemon.pid"
STOP_FILE = STATE_DIR / "stop.flag"
PREF_PATH = ROOT / "data" / "preferences.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _save_pid(pid: int):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")


def _read_pid() -> int:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def _proc_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        # On Windows and POSIX, sending signal 0 is a common liveness check for POSIX.
        # For portability, try os.kill on POSIX; on Windows, fallback to open process via subprocess (naive).
        if os.name == 'nt':
            # Naive: tasklist string check
            out = subprocess.check_output(['tasklist'], creationflags=0)
            return str(pid).encode() in out
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def status() -> dict:
    prefs = _read_json(PREF_PATH)
    pid = _read_pid()
    running = _proc_alive(pid)
    # Fallback: if PID unknown but enabled true, treat as logically on (UI hint)
    if not running and prefs.get('background_enabled', False):
        running = True
    return {
        'enabled': bool(prefs.get('background_enabled', False)),
        'interval_sec': int(prefs.get('background_interval_sec', 900)),
        'pid': pid if running else 0,
        'running': running,
    }


def set_enabled(enabled: bool, interval_sec: int | None = None):
    prefs = _read_json(PREF_PATH)
    prefs['background_enabled'] = bool(enabled)
    if interval_sec is not None:
        prefs['background_interval_sec'] = int(interval_sec)
    _write_json(PREF_PATH, prefs)


def start_daemon(interval_sec: int | None = None) -> dict:
    set_enabled(True, interval_sec)
    if status().get('running'):
        return status()
    # Spawn detached background daemon
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    kwargs = {}
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_CONSOLE
    try:
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.background_daemon'], cwd=str(ROOT), creationflags=creationflags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _save_pid(proc.pid)
    except Exception:
        pass
    return status()


def stop_daemon() -> dict:
    set_enabled(False)
    try:
        STOP_FILE.write_text('stop', encoding='utf-8')
    except Exception:
        pass
    return status()


def rest_for(minutes: int) -> dict:
    # Disable and set resume timestamp
    prefs = _read_json(PREF_PATH)
    prefs['background_enabled'] = False
    import time
    prefs['background_resume_ts'] = time.time() + max(60, int(minutes) * 60)
    _write_json(PREF_PATH, prefs)
    try:
        STOP_FILE.write_text('stop', encoding='utf-8')
    except Exception:
        pass
    return status()