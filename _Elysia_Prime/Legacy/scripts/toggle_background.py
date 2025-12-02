# [Genesis: 2025-12-02] Purified by Elysia
"""
Toggle background micro-learner on/off and optionally write a stop flag.
Usage:
  python -m scripts.toggle_background --on
  python -m scripts.toggle_background --off
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREF_PATH = ROOT / "data" / "preferences.json"
STATE_DIR = ROOT / "data" / "background"
STOP_FILE = STATE_DIR / "stop.flag"


def load_prefs():
    try:
        return json.loads(PREF_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(prefs):
    try:
        PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREF_PATH.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--interval", type=int, default=None, help="Background interval seconds (default 900)")
    args = ap.parse_args()
    p = load_prefs()
    if args.on:
        p["background_enabled"] = True
    if args.off:
        p["background_enabled"] = False
        # signal running daemon to stop once
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            STOP_FILE.write_text("stop", encoding="utf-8")
        except Exception:
            pass
    if args.interval is not None:
        p["background_interval_sec"] = int(args.interval)
    save_prefs(p)
    print("Background:", "ON" if p.get("background_enabled") else "OFF",
          "interval:", p.get("background_interval_sec", 900))


if __name__ == "__main__":
    main()
