# [Genesis: 2025-12-02] Purified by Elysia
#!/usr/bin/env python3
import sys
import json


def main():
    raw = sys.stdin.read()
    try:
        decision = json.loads(raw)
    except Exception:
        print(raw)
        return 1

    if not decision.get('confirm_required'):
        print(json.dumps(decision))
        return 0

    tool = decision.get('tool_name', 'unknown')
    print(f"Confirm executing tool '{tool}'? [y/N] ", end='', flush=True)
    ans = sys.stdin.readline().strip().lower()
    if ans in ('y', 'yes'):
        decision['confirm'] = True
        decision.pop('confirm_required', None)
    print(json.dumps(decision))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
