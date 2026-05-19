"""
Script: Audit Dark Matter (Meaningless Node Detection)
======================================================
Counts nodes that are likely 'meaningless' (IDs matching patterns).
Patterns:
- Star-*
- Wikipedia_* (generic)
- Pure Numbers
"""

import sys
import os
import re

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.Foundation.Graph.torch_graph import get_torch_graph

def audit():
    print("🕵️ Auditing Dark Matter...")
    graph = get_torch_graph()
    
    # Force load if empty (should check rainbow)
    if len(graph.id_to_idx) < 100:
        print("   Graph empty. Loading Rainbow...")
        graph.load_rainbow_bridge()

    nodes = list(graph.id_to_idx.keys())
    total = len(nodes)
    
    # Patterns
    star_pattern = re.compile(r"^Star-\d+$")
    wiki_pattern = re.compile(r"^Wikipedia_\d+$")
    num_pattern = re.compile(r"^\d+$")
    
    count_star = 0
    count_wiki = 0
    count_num = 0
    
    for nid in nodes:
        if star_pattern.match(str(nid)): count_star += 1
        elif wiki_pattern.match(str(nid)): count_wiki += 1
        elif num_pattern.match(str(nid)): count_num += 1
        
    meaningless = count_star + count_wiki + count_num
    ratio = meaningless / total if total > 0 else 0
    
    print(f"\n📊 Audit Results:")
    print(f"   Total Nodes: {total}")
    print(f"   -------------------")
    print(f"   Star-* IDs:      {count_star}")
    print(f"   Wikipedia_*:     {count_wiki}")
    print(f"   Pure Numbers:    {count_num}")
    print(f"   Total Dark Matter: {meaningless} ({ratio*100:.1f}%)")
    
    if ratio > 0.5:
        print("\n⚠️ WARNING: Majority of graph is Dark Matter.")
        print("   Recommendation: Implement Phase 14 (The Naming).")
    else:
        print("\n✅ Healthy: Meaningful nodes dominate.")

if __name__ == "__main__":
    audit()
