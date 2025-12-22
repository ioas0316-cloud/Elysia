import os
import json
from collections import Counter, defaultdict
from datetime import datetime


def load_events(day_dir: str):
    path = os.path.join(day_dir, 'events.jsonl')
    if not os.path.exists(path):
        print(f"No telemetry file at {path}")
        return []
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def summarize(events):
    kinds = Counter(e.get('event_type') for e in events)
    violations = [e for e in events if e.get('event_type') == 'policy_violation']
    blocked = [e for e in events if e.get('event_type') == 'action_blocked']
    echo_updates = [e for e in events if e.get('event_type') == 'echo_updated']
    echo_spatial = [e for e in events if e.get('event_type') == 'echo_spatial_stats']
    convo_q = [e for e in events if e.get('event_type') == 'conversation_quality']
    confirms = [e for e in events if e.get('event_type') == 'action_confirm_required']
    episodes = [e for e in events if e.get('event_type') == 'episode_summary_saved']

    # Average echo entropy
    avg_entropy = None
    if echo_updates:
        vals = [e.get('payload', {}).get('entropy', 0.0) for e in echo_updates]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            avg_entropy = sum(vals) / len(vals)

    # Average spatial radius
    avg_radius = None
    if echo_spatial:
        vals = [e.get('payload', {}).get('avg_dist', 0.0) for e in echo_spatial]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            avg_radius = sum(vals) / len(vals)

    # Conversation quality aggregates
    cq = defaultdict(int)
    for e in convo_q:
        p = e.get('payload', {})
        for k in ['questions', 'gratitude', 'apology', 'empathy', 'consent']:
            if isinstance(p.get(k), int):
                cq[k] += p.get(k)

    summary = {
        'total_events': len(events),
        'by_type': kinds,
        'violations': len(violations),
        'blocked_actions': len(blocked),
        'echo_updates': len(echo_updates),
        'avg_echo_entropy': avg_entropy,
        'avg_echo_radius': avg_radius,
        'confirm_requests': len(confirms),
        'episode_summaries': len(episodes),
        'conversation_quality': dict(cq)
    }
    return summary


def main(day: str = None):
    base = os.path.join('data', 'telemetry')
    if not day:
        day = datetime.utcnow().strftime('%Y%m%d')
    day_dir = os.path.join(base, day)
    events = load_events(day_dir)
    s = summarize(events)
    print("Telemetry summary for", day)
    for k, v in s.items():
        print(f"- {k}: {v}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
