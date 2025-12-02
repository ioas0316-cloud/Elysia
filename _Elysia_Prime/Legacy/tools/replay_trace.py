# [Genesis: 2025-12-02] Purified by Elysia
import os
import json
import sys


def replay(trace_id: str, day: str = None):
    base = os.path.join('data', 'telemetry')
    if not day:
        # scan most recent day
        days = sorted([d for d in os.listdir(base) if d.isdigit()])
        if not days:
            print('No telemetry days found')
            return
        day = days[-1]
    path = os.path.join(base, day, 'events.jsonl')
    if not os.path.exists(path):
        print('No events file at', path)
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('trace_id') == trace_id:
                print(json.dumps(ev, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/replay_trace.py <trace_id> [day]')
        raise SystemExit(1)
    replay(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
