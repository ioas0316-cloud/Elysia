# [Genesis: 2025-12-02] Purified by Elysia
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime


def telemetry_path(base='data/telemetry', day=None):
    if not day:
        day = datetime.utcnow().strftime('%Y%m%d')
    return os.path.join(base, day, 'events.jsonl')


def load_events(path):
    events = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return events


def summarize_routes(events):
    agg = defaultdict(lambda: {'count': 0, 'latency_sum': 0.0, 'errors': 0})
    for ev in events:
        if ev.get('event_type') != 'route.arc':
            continue
        p = ev.get('payload', {})
        key = (p.get('from_mod'), p.get('to_mod'))
        agg[key]['count'] += 1
        agg[key]['latency_sum'] += float(p.get('latency_ms', 0.0))
        if str(p.get('outcome', 'ok')).lower() not in ('ok', 'empty'):
            agg[key]['errors'] += 1
    rows = []
    for (frm, to), a in agg.items():
        cnt = a['count']
        avg = (a['latency_sum'] / cnt) if cnt else 0.0
        err = (a['errors'] / cnt) if cnt else 0.0
        rows.append({'from': frm, 'to': to, 'count': cnt, 'avg_ms': round(avg, 2), 'error_rate': round(err, 3)})
    rows.sort(key=lambda r: (r['error_rate'], r['avg_ms'], -r['count']), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description='Summarize route congestion from telemetry events.')
    ap.add_argument('--day', help='YYYYMMDD (defaults to today)')
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()

    path = telemetry_path(day=args.day)
    events = load_events(path)
    routes = summarize_routes(events)

    print(f"[traffic] Telemetry: {path}")
    print("from_mod -> to_mod | count | avg_ms | error_rate")
    for r in routes[: args.top]:
        print(f"{r['from']} -> {r['to']} | {r['count']} | {r['avg_ms']} | {r['error_rate']}")


if __name__ == '__main__':
    main()
