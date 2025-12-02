# [Genesis: 2025-12-02] Purified by Elysia
"""
Builder Timeline (text view)

Parses BUILDER_LOG.md and prints time-ordered changes with tags.
Optional: output a simple DOT graph to stdout with --dot.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("BUILDER_LOG.md")

ENTRY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \| \[(.*?)\] \| (.*?) \| (.*?) \| (.*)$")


def parse_entries(text: str):
    entries = []
    for line in text.splitlines():
        m = ENTRY_RE.match(line.strip())
        if not m:
            continue
        ts, layer, what, why, proto = m.groups()
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        except Exception:
            continue
        entries.append({
            "ts": dt,
            "layer": layer,
            "what": what,
            "why": why,
            "proto": proto,
        })
    entries.sort(key=lambda e: e["ts"])  # ascending
    return entries


def print_text(entries):
    for e in entries:
        print(f"{e['ts']:%Y-%m-%d %H:%M} | [{e['layer']}] | {e['what']} -> {e['why']} | {e['proto']}")


def print_dot(entries):
    print("digraph BuilderTimeline {")
    print("  rankdir=LR;")
    last_id = None
    for i, e in enumerate(entries):
        node_id = f"n{i}"
        label = f"{e['ts']:%m-%d %H:%M}\\n[{e['layer']}]\\n{e['what']}"
        color = "#5DADE2" if e['layer'].upper().startswith('STARTER') else ("#58D68D" if e['layer'].upper().startswith('CELLWORLD') else "#B2BABB")
        print(f"  {node_id} [shape=box, style=filled, color=\"{color}\", label=\"{label}\"];\n")
        if last_id is not None:
            print(f"  {last_id} -> {node_id};")
        last_id = node_id
    print("}")


def main(argv: list[str]):
    if not LOG_PATH.exists():
        print("No BUILDER_LOG.md found", file=sys.stderr)
        return 1
    text = LOG_PATH.read_text(encoding="utf-8")
    entries = parse_entries(text)
    if "--dot" in argv:
        print_dot(entries)
    else:
        print_text(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
