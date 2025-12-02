# [Genesis: 2025-12-02] Purified by Elysia
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PREF = DATA / 'preferences.json'
BG_DIR = DATA / 'background'
STOP_FLAG = BG_DIR / 'stop.flag'


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _write_json(p: Path, obj: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def quiet_all() -> dict:
    # 1) Signal background daemon to stop and disable
    try:
        from tools import bg_control
        bg_control.stop_daemon()
    except Exception:
        pass

    try:
        BG_DIR.mkdir(parents=True, exist_ok=True)
        STOP_FLAG.write_text('stop', encoding='utf-8')
    except Exception:
        pass

    # 2) Preferences: disable background + quiet mode + disable auto_act
    prefs = _read_json(PREF)
    prefs['background_enabled'] = False
    prefs['quiet_mode'] = True
    prefs['auto_act'] = False
    if 'background_interval_sec' not in prefs:
        prefs['background_interval_sec'] = 900
    _write_json(PREF, prefs)

    # 3) Best-effort: remove Windows scheduled task if present
    try:
        if os.name == 'nt':
            subprocess.run(['schtasks', '/Delete', '/TN', 'ElysiaGrowthSprint', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    return {
        'background_enabled': prefs.get('background_enabled', False),
        'quiet_mode': prefs.get('quiet_mode', False),
        'auto_act': prefs.get('auto_act', False),
        'stop_flag': STOP_FLAG.exists(),
    }


def main() -> None:
    st = quiet_all()
    print('[quiet-all] Applied:')
    for k, v in st.items():
        print(f'  - {k}: {v}')
    print('\nNote: If a Flask server is running on this console, press Ctrl+C to stop it.')


if __name__ == '__main__':
    main()
