from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PREF = DATA / 'preferences.json'


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


def resume_all(interval_sec: Optional[int] = None) -> dict:
    # Preferences: enable background, disable quiet, enable auto_act
    prefs = _read_json(PREF)
    prefs['background_enabled'] = True
    prefs['quiet_mode'] = False
    prefs['auto_act'] = True
    if interval_sec is not None:
        prefs['background_interval_sec'] = int(interval_sec)
    elif 'background_interval_sec' not in prefs:
        prefs['background_interval_sec'] = 900
    _write_json(PREF, prefs)

    # Start daemon with desired interval
    try:
        from tools.bg_control import start_daemon
        st = start_daemon(prefs.get('background_interval_sec', 900))
    except Exception:
        st = {'enabled': True, 'running': False, 'pid': 0, 'interval_sec': prefs.get('background_interval_sec', 900)}

    return {
        'background_enabled': prefs.get('background_enabled', False),
        'quiet_mode': prefs.get('quiet_mode', True),
        'auto_act': prefs.get('auto_act', False),
        'interval_sec': prefs.get('background_interval_sec', 900),
        'bg_status': st,
    }


def main() -> None:
    st = resume_all()
    print('[resume-all] Applied:')
    for k, v in st.items():
        print(f'  - {k}: {v}')
    print('\nTip: Use BG OFF any time to pause background learning.')


if __name__ == '__main__':
    main()

