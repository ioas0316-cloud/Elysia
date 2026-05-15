import json
from collections import Counter

def analyze_kg(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get('nodes', {})
    edges = data.get('edges', [])

    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")

    # Analyze node layers
    layers = Counter(n.get('layer', 'unknown') for n in nodes.values())
    print(f"Layers: {dict(layers)}")

    # Analyze specific concept evolution: LOVE
    love_related = [n for n in nodes.keys() if 'love' in n.lower()]
    print(f"Love-related nodes ({len(love_related)}): {love_related[:10]}")

    # Sample edges
    if edges:
        print(f"Sample Edges (first 5):")
        for edge in edges[:5]:
            print(f"  {edge.get('source')} --({edge.get('relation')})--> {edge.get('target')}")

if __name__ == "__main__":
    analyze_kg('data/knowledge/kg_with_embeddings.json')
