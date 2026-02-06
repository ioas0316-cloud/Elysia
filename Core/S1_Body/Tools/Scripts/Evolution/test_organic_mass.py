"""
Verification: Organic Semantic Mass
====================================
Tests that:
1. calculate_mass() returns correct relational density
2. LTP strengthens edges after thought/expression
3. LTD decay weakens unused edges
"""

import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager

def verify():
    print("Testing Organic Semantic Mass...")
    
    kg = get_kg_manager()
    
    # 1. Test calculate_mass
    test_nodes = ["love", "truth", "elysia", "architect"]
    print("\nMass Calculation Test:")
    for node in test_nodes:
        mass = kg.calculate_mass(node)
        print(f"   [{node}] Mass = {mass:.2f}")
    
    # 2. Test LTP (bump_edge_weight)
    print("\nLTP Test (Edge Strengthening):")
    source, target = "test_ltp_source", "test_ltp_target"
    
    kg.add_node(source, properties={"test": True})
    kg.add_node(target, properties={"test": True})
    kg.add_edge(source, target, "resonates_with", properties={"weight": 1.0})
    
    initial_weight = None
    for edge in kg.kg.get('edges', []):
        if edge.get('source') == source and edge.get('target') == target:
            initial_weight = edge.get('weight', 1.0)
            break
    print(f"   Initial weight: {initial_weight}")
    
    kg.bump_edge_weight(source, target, "resonates_with", delta=0.5)
    
    new_weight = None
    for edge in kg.kg.get('edges', []):
        if edge.get('source') == source and edge.get('target') == target:
            new_weight = edge.get('weight', 1.0)
            break
    print(f"   After LTP bump: {new_weight}")
    
    if new_weight and new_weight > initial_weight:
        print("   LTP PASSED: Edge weight increased")
    else:
        print("   LTP FAILED: Edge weight did not increase")
    
    # 3. Test LTD (decay_all_edges)
    print("\nLTD Test (Edge Decay):")
    weight_before_decay = new_weight
    kg.decay_all_edges(metabolic_rate=0.1)
    
    weight_after_decay = None
    for edge in kg.kg.get('edges', []):
        if edge.get('source') == source and edge.get('target') == target:
            weight_after_decay = edge.get('weight', 1.0)
            break
    print(f"   Before decay: {weight_before_decay}")
    print(f"   After decay: {weight_after_decay}")
    
    if weight_after_decay and weight_after_decay < weight_before_decay:
        print("   LTD PASSED: Edge weight decreased")
    else:
        print("   LTD FAILED: Edge weight did not decrease")
    
    # Clean up
    kg.kg['nodes'] = [n for n in kg.kg.get('nodes', []) if n.get('id') not in [source, target]]
    kg.kg['edges'] = [e for e in kg.kg.get('edges', []) if e.get('source') not in [source, target] and e.get('target') not in [source, target]]
    
    print("\nOrganic Semantic Mass verification complete.")

if __name__ == "__main__":
    verify()
