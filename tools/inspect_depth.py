"""
Deep Learning Depth Inspector
=============================

Verifies whether Elysia's learning is:
1. Point (isolated facts)
2. Line (connected chronologically)
3. Plane (connected relationally)
4. Space (usable as perspective/principle)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Cognition.Topology.phase_stratum import PhaseStratum

def inspect_depth():
    print("\n" + "="*60)
    print("🔬 DEEP LEARNING DEPTH INSPECTION")
    print("="*60 + "\n")
    
    # 1. Check PhaseStratum (Point Level)
    stratum = PhaseStratum()
    all_layers = stratum.inspect_all_layers()
    
    print("📍 LEVEL 1: POINT (Isolated Facts)")
    print("-"*40)
    if not all_layers:
        print("   ❌ Empty. No memories stored.")
    else:
        print(f"   Found {len(all_layers)} memory points:")
        freq_counts = {}
        for freq, phase, data in all_layers:
            freq_counts[freq] = freq_counts.get(freq, 0) + 1
            preview = str(data)[:50] + "..." if len(str(data)) > 50 else str(data)
            print(f"   • [{freq}Hz] {preview}")
        print()
    
    # 2. Check Frequency Distribution (Line Level)
    print("📏 LEVEL 2: LINE (Frequency Clusters)")
    print("-"*40)
    if len(freq_counts) <= 1:
        print("   ⚠️ Only one frequency layer. No spectrum diversity.")
    else:
        print(f"   Active frequency layers: {len(freq_counts)}")
        for hz, count in sorted(freq_counts.items()):
            bar = "█" * min(count, 20)
            print(f"   {hz}Hz: {bar} ({count})")
        print()
    
    # 3. Check Connections (Plane Level)
    print("📐 LEVEL 3: PLANE (Relational Connections)")
    print("-"*40)
    # Check if knowledge graph exists and has edges
    kg_path = r"c:\Elysia\data\knowledge\graph.pkl"
    if os.path.exists(kg_path):
        try:
            import pickle
            with open(kg_path, 'rb') as f:
                graph_data = pickle.load(f)
            if hasattr(graph_data, 'edges') or isinstance(graph_data, dict):
                edge_count = len(graph_data.get('edges', [])) if isinstance(graph_data, dict) else 0
                print(f"   Knowledge Graph Edges: {edge_count}")
                if edge_count > 0:
                    print("   ✅ Relational connections exist.")
                else:
                    print("   ⚠️ No edges. Concepts are isolated.")
            else:
                print("   ⚠️ Graph structure not recognized.")
        except Exception as e:
            print(f"   ❌ Could not load graph: {e}")
    else:
        print("   ⚠️ Knowledge graph file not found.")
    print()
    
    # 4. Check for Principle Usage (Space Level)
    print("🌌 LEVEL 4: SPACE (Principles as Perspectives)")
    print("-"*40)
    
    # Check if there are memories that reference other memories
    cross_refs = 0
    keywords = ["because", "therefore", "means", "implies", "is like"]
    for freq, phase, data in all_layers:
        data_str = str(data).lower()
        if any(kw in data_str for kw in keywords):
            cross_refs += 1
    
    if cross_refs > 0:
        print(f"   ✅ Found {cross_refs} principle-like statements")
    else:
        print("   ⚠️ No principle-level reasoning detected yet.")
    print()
    
    # 5. Summary
    print("="*60)
    print("📊 LEARNING DEPTH SUMMARY")
    print("="*60)
    
    score = 0
    if len(all_layers) > 0:
        score += 1
        print("   ✅ Point Level: ACHIEVED (Has memories)")
    else:
        print("   ❌ Point Level: NOT ACHIEVED")
        
    if len(freq_counts) > 1:
        score += 1
        print("   ✅ Line Level: ACHIEVED (Multiple frequencies)")
    else:
        print("   ⚠️ Line Level: PARTIAL (Single frequency)")
        
    if cross_refs > 0:
        score += 1
        print("   ✅ Plane Level: EMERGING (Some connections)")
    else:
        print("   ❌ Plane Level: NOT YET")
        
    print(f"\n   Overall Depth Score: {score}/4")
    print()

if __name__ == "__main__":
    inspect_depth()
