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


def scale(val, vmin, vmax, out_min, out_max):
    if vmax <= vmin:
        return out_min
    x = (val - vmin) / (vmax - vmin)
    return out_min + x * (out_max - out_min)


def main():
    ap = argparse.ArgumentParser(description='Update planet routes from route.arc telemetry.')
    ap.add_argument('--day', help='YYYYMMDD (defaults to today)')
    ap.add_argument('--telemetry', default=None)
    ap.add_argument('--planet', default=os.path.join('data', 'world', 'planet.json'))
    args = ap.parse_args()

    tel_path = args.telemetry or telemetry_path(day=args.day)
    events = load_events(tel_path)

    # aggregate
    agg = defaultdict(lambda: {'count': 0, 'lat_sum': 0.0, 'errors': 0})
    for ev in events:
        if ev.get('event_type') != 'route.arc':
            continue
        p = ev.get('payload', {})
        key = (p.get('from_mod'), p.get('to_mod'))
        agg[key]['count'] += 1
        agg[key]['lat_sum'] += float(p.get('latency_ms', 0.0))
        if str(p.get('outcome', 'ok')).lower() not in ('ok', 'empty'):
            agg[key]['errors'] += 1

    # derive scales
    counts = [a['count'] for a in agg.values()] or [1]
    lats = [a['lat_sum'] / max(1, a['count']) for a in agg.values()] or [0.0]
    cmin, cmax = min(counts), max(counts)
    lmin, lmax = min(lats), max(lats)

    # load planet
    try:
        with open(args.planet, 'r', encoding='utf-8') as f:
            planet = json.load(f)
    except FileNotFoundError:
        planet = {'meta': {'version': '0.1.0'}, 'cities': [], 'routes': []}

    routes = []
    for (frm, to), a in agg.items():
        cnt = a['count']
        avg = a['lat_sum'] / max(1, cnt)
        err = a['errors'] / max(1, cnt)
        thickness = round(scale(cnt, cmin, cmax, 0.2, 3.0), 3)
        # color by avg latency (green->yellow->red). store as value; renderer will map.
        heat = round(scale(avg, lmin, lmax if lmax > lmin else lmin + 1.0, 0.0, 1.0), 3)
        routes.append({
            'from_mod': frm,
            'to_mod': to,
            'count': cnt,
            'avg_ms': round(avg, 2),
            'error_rate': round(err, 3),
            'thickness': thickness,
            'latency_heat': heat
        })

    planet['routes'] = routes

    os.makedirs(os.path.dirname(args.planet), exist_ok=True)
    with open(args.planet, 'w', encoding='utf-8') as f:
        json.dump(planet, f, ensure_ascii=False, indent=2)

    print(f"[planet] updated routes: {len(routes)} from {tel_path} -> {args.planet}")


if __name__ == '__main__':
    main()

