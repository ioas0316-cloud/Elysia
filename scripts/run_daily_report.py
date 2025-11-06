"""
Aggregate today's artifacts (journal, summary, optional book report, story),
write a markdown overview and render a PNG daily card.

Usage:
  python -m scripts.run_daily_report [--date YYYY-MM-DD]
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from Project_Mirror.report_renderer import ReportRenderer
from tools.kg_manager import KGManager
import json
import os
from datetime import datetime as _dt
from scripts.generate_metrics import summarize_kg


def _find_today_files(base: Path, date_str: str):
    j_txt = base / "journal" / f"{date_str}.txt"
    j_sum = base / "journal" / f"{date_str}_summary.txt"
    reports_dir = base / "reports"
    writings_dir = base / "writings"
    # Best-effort: pick latest writing for today
    latest_story = None
    if writings_dir.exists():
        candidates = sorted(writings_dir.glob(f"{date_str.replace('-', '')}_*.md"))
        latest_story = candidates[-1] if candidates else None
    # Book report for today (if any)
    book_report = None
    if reports_dir.exists():
        # pick any report; daily matching is optional
        items = sorted(reports_dir.glob("*_report.md"))
        book_report = items[-1] if items else None
    return j_txt, j_sum, book_report, latest_story


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    data_dir = Path("data")
    j_txt, j_sum, b_rep, story = _find_today_files(data_dir, date_str)

    # Markdown summary
    out_dir = data_dir / "reports" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"daily_{date_str}.md"
    lines = [f"# Daily Report — {date_str}", ""]
    if j_txt.exists():
        lines += ["## Journal", f"- Entry: {j_txt}", f"- Summary: {j_sum if j_sum.exists() else '(missing)'}", ""]
    if b_rep and b_rep.exists():
        lines += ["## Book Report", f"- Report: {b_rep}", ""]
    if story and story.exists():
        lines += ["## Creative Writing", f"- Story: {story}", ""]

    # Value mass changes (if any)
    trace_path = data_dir / "value_mass_trace.json"
    if trace_path.exists():
        try:
            entries = json.loads(trace_path.read_text(encoding="utf-8"))
            todays = [e for e in entries if str(e.get('timestamp','')).startswith(date_str)]
            if todays:
                # Sort by absolute delta
                for e in todays:
                    try:
                        e['delta'] = float(e.get('after', 0)) - float(e.get('before', 0))
                    except Exception:
                        e['delta'] = 0.0
                top = sorted(todays, key=lambda x: abs(x.get('delta', 0.0)), reverse=True)[:5]
                lines += ["## Value Mass Changes", "(top by absolute change)"]
                for e in top:
                    lines.append(f"- {e.get('value')}: {e.get('before')} -> {e.get('after')} (? {round(e.get('delta',0.0), 3)})")
                lines.append("")
        except Exception:
            pass
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Metrics snapshot (Top concepts, supports/refutes, values)
    try:
        kgm = KGManager()
        summ = summarize_kg(kgm.kg)
        lines += ["## Metrics Snapshot", ""]
        lines += [f"- Nodes: {summ['nodes']}  Edges: {summ['edges']}  Supports: {summ['supports']}  Refutes: {summ['refutes']}"]
        lines += ["- Top Concepts:"]
        for name, c in summ['top_concepts']:
            lines.append(f"  - {name}: {c}")
        lines.append("")
    except Exception:
        pass

    # Reasoning snapshot from telemetry (last 50)
    try:
        tel_dir = os.path.join('data','telemetry', _dt.utcnow().strftime('%Y%m%d'))
        tel_path = os.path.join(tel_dir, 'events.jsonl')
        if os.path.exists(tel_path):
            with open(tel_path, 'r', encoding='utf-8') as f:
                lines_json = f.readlines()[-100:]
            events = []
            for l in lines_json:
                try:
                    events.append(json.loads(l))
                except Exception:
                    pass
            flows = [e for e in events if e.get('event_type') == 'flow.decision']
            if flows:
                lines += ["## Reasoning Snapshot (flow decisions)"]
                # Count top choices
                from collections import Counter
                cnt = Counter(e.get('payload',{}).get('top_choice') for e in flows)
                lines += ["- Top choice counts:"]
                for k, v in cnt.most_common():
                    lines.append(f"  - {k}: {v}")
                # Show last few with echo focus
                lines += ["- Recent decisions:"]
                for e in flows[-5:]:
                    evd = (e.get('payload',{}).get('evidence',{}) or {}).get('echo_top') or []
                    lines.append(f"  - {e.get('payload',{}).get('top_choice')}  echo:{','.join(evd)}")
                lines.append("")
    except Exception:
        pass

    # PNG card
    info = {
        "Journal": str(j_txt) if j_txt.exists() else "(none)",
        "Summary": str(j_sum) if j_sum.exists() else "(none)",
        "BookReport": str(b_rep) if b_rep and b_rep.exists() else "(none)",
        "Story": str(story) if story and story.exists() else "(none)",
    }
    card = ReportRenderer().render_daily_card(date_str, info)

    # Anchor in KG
    kg = KGManager()
    node = f"daily_report_{date_str}"
    kg.add_node(node, properties={"type": "daily_report", "md": str(md_path), "card": str(card)})
    if j_txt.exists():
        kg.add_edge(node, f"journal_entry_{date_str}", "summarizes")
    kg.save()

    from datetime import datetime as __now
    ts = __now.now().strftime('%H:%M:%S')
    print(f"[{ts}] Daily MD:", md_path)
    print(f"[{ts}] Daily PNG:", card)


if __name__ == "__main__":
    main()

