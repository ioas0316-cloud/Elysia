"""
Toggle quiet mode on/off in data/preferences.json.

Usage:
  python -m scripts.toggle_quiet_mode --on
  python -m scripts.toggle_quiet_mode --off
"""
from __future__ import annotations

import argparse
from tools.preferences import load_prefs, save_prefs, ensure_defaults


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--on', action='store_true')
    g.add_argument('--off', action='store_true')
    args = p.parse_args()

    prefs = ensure_defaults(load_prefs())
    prefs['quiet_mode'] = bool(args.on)
    save_prefs(prefs)
    print('Quiet mode:', 'ON' if args.on else 'OFF')


if __name__ == '__main__':
    main()

