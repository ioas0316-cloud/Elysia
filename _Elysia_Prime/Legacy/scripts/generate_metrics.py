# [Genesis: 2025-12-02] Purified by Elysia
import os
import json
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from tools.kg_manager import KGManager


def summarize_kg(kg):
    nodes = kg.get('nodes', [])
    edges = kg.get('edges', [])
    n_nodes = len(nodes)
    n_edges = len(edges)
    # supports/refutes ratio
    rel_counter = Counter(e.get('relation', 'unknown') for e in edges)
    supports = rel_counter.get('supports', 0)
    refutes = rel_counter.get('refutes', 0)
    # top concepts by degree (supports in or out)
    deg = Counter()
    for e in edges:
        if e.get('relation') == 'supports':
            deg[e.get('source')] += 1
            deg[e.get('target')] += 1
    # map id->label
    label = {}
    for n in nodes:
        nid = n.get('id')
        if not nid:
            continue
        label[nid] = n.get('label') or n.get('id')
    # pick concept:* only
    concept_scores = [(nid, c) for nid, c in deg.items() if str(nid).startswith('concept:')]
    concept_scores.sort(key=lambda x: x[1], reverse=True)
    top_concepts = [(label.get(nid, nid), c) for nid, c in concept_scores[:10]]
    # value mass snapshot
    values = []
    for n in nodes:
        if str(n.get('id','')).startswith('value:'):
            values.append((n.get('id'), n.get('mass', 0)))
    return {
        'nodes': n_nodes,
        'edges': n_edges,
        'relations': rel_counter,
        'supports': supports,
        'refutes': refutes,
        'top_concepts': top_concepts,
        'values': values,
    }


def write_md(report_dir: Path, summary: dict):
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d')
    p = report_dir / f'metrics_{ts}.md'
    lines = []
    lines.append(f"# Daily Metrics — {ts}")
    lines.append("")
    lines.append(f"- Nodes: {summary['nodes']}")
    lines.append(f"- Edges: {summary['edges']}")
    lines.append(f"- Supports: {summary['supports']}  Refutes: {summary['refutes']}")
    lines.append("")
    lines.append("## Top Concepts (by supports degree)")
    for name, c in summary['top_concepts']:
        lines.append(f"- {name}: {c}")
    lines.append("")
    lines.append("## Values (mass snapshot)")
    for vid, m in summary['values']:
        lines.append(f"- {vid}: mass={m}")
    p.write_text("\n".join(lines), encoding='utf-8')
    return p


def main():
    kgm = KGManager()
    summary = summarize_kg(kgm.kg)
    out = write_md(Path('data/reports/daily'), summary)
    print('[metrics] Wrote', out)


if __name__ == '__main__':
    main()
