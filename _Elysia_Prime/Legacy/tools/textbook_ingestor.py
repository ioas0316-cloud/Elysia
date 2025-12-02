# [Genesis: 2025-12-02] Purified by Elysia
import argparse
import json
import os
import sys

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tools.kg_manager import KGManager


def ingest_subject(subject: str, kg: KGManager, textbooks_dir: str = None) -> int:
    """
    Ingests a textbook JSON into the KG.
    - Adds nodes by 'name'
    - Adds edges from 'causality' triples: [source, relation, target]
    Returns number of edges added.
    """
    textbooks_dir = textbooks_dir or os.path.join(PROJECT_ROOT, 'data', 'textbooks')
    path = os.path.join(textbooks_dir, f"{subject}.json")
    if not os.path.exists(path):
        print(f"[ingestor] Skipping missing textbook: {path}")
        return 0

    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    added_edges = 0
    for item in items:
        name = item.get('name')
        if not name:
            continue
        kg.add_node(name, properties={'category': 'textbook', 'subject': subject})

        # optional: add property nodes (e.g., has_property -> attribute)
        for triple in item.get('causality', []):
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            src, rel, tgt = triple
            src = str(src)
            rel = str(rel)
            tgt = str(tgt)
            try:
                kg.add_edge(src, tgt, rel)
                added_edges += 1
            except Exception:
                pass

    kg.save()
    return added_edges


def main():
    parser = argparse.ArgumentParser(description='Ingest textbooks into the Knowledge Graph.')
    parser.add_argument('--subjects', nargs='*', default=['geometry_primitives'], help='List of textbook subjects to ingest')
    parser.add_argument('--dir', default=None, help='Optional textbooks directory (defaults to data/textbooks)')
    args = parser.parse_args()

    kg = KGManager()
    total_edges = 0
    for subject in args.subjects:
        edges = ingest_subject(subject, kg, args.dir)
        print(f"[ingestor] Subject '{subject}': +{edges} edges")
        total_edges += edges
    print(f"[ingestor] Done. KG summary: {kg.get_summary()} (edges added: {total_edges})")


if __name__ == '__main__':
    main()
