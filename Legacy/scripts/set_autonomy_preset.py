"""
Set autonomy/quiet presets.

Presets:
  quiet:    quiet_mode=ON,  auto_act=OFF, min_arousal=0.6, cooldown=3600
  balanced: quiet_mode=OFF, auto_act=ON,  min_arousal=0.5, cooldown=1800
  lively:   quiet_mode=OFF, auto_act=ON,  min_arousal=0.4, cooldown=900
"""
from __future__ import annotations

import argparse
from tools.preferences import load_prefs, save_prefs, ensure_defaults


def apply_preset(name: str):
    prefs = ensure_defaults(load_prefs())
    if name == 'quiet':
        prefs['quiet_mode'] = True
        prefs['auto_act'] = False
        prefs['min_arousal_for_proposal'] = 0.6
        for k in prefs['proposal_cooldowns']:
            prefs['proposal_cooldowns'][k] = 3600
        prefs['autonomy_intensity'] = 'low'
    elif name == 'balanced':
        prefs['quiet_mode'] = False
        prefs['auto_act'] = True
        prefs['min_arousal_for_proposal'] = 0.5
        for k in prefs['proposal_cooldowns']:
            prefs['proposal_cooldowns'][k] = 1800
        prefs['autonomy_intensity'] = 'medium'
    elif name == 'lively':
        prefs['quiet_mode'] = False
        prefs['auto_act'] = True
        prefs['min_arousal_for_proposal'] = 0.4
        for k in prefs['proposal_cooldowns']:
            prefs['proposal_cooldowns'][k] = 900
        prefs['autonomy_intensity'] = 'high'
    else:
        raise SystemExit('Unknown preset: ' + name)
    save_prefs(prefs)
    return prefs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=['quiet', 'balanced', 'lively'], required=True)
    args = p.parse_args()
    prefs = apply_preset(args.preset)
    print('Applied preset:', args.preset)
    print('Current preferences:', prefs)


if __name__ == '__main__':
    main()

