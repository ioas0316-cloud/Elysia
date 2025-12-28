"""
Script: Demo Concept Utility
============================
Demonstrates how 'Concept_XXXX' nodes are UTILIZED (not just stored).
User Question: "Is it just an empty node?"
Answer: "No. Gravity pulls it, and we READ the payload."

Scenario:
1. Input: "Chaos" (Abstract Concept)
2. Graph Action: Find neighbors.
3. Discovery: Found 'Math_Formula_XXXX'.
4. Utilization: Read payload ('Z = Z^2 + C').
5. Output: "Chaos is defined by Z = Z^2 + C."
"""

import sys
import os
import torch
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core._01_Foundation._05_Governance.Foundation.Graph.torch_graph import get_torch_graph

def demo():
    print("🧠 DEMO: Concept Utilization (From Gravity to Meaning)")
    print("====================================================")
    graph = get_torch_graph()
    
    # 1. Simulate a Query Node "Chaos" (or we pick a random existing Concept)
    # Since we renamed everything, we don't know the ID for "Chaos" easily unless we search payload.
    # Let's search for a node that HAS a formula in active memory.
    
    print("🔍 Scanning for a node with 'formula' in payload...")
    target_nid = None
    target_payload = None
    
    for nid, meta in graph.node_metadata.items():
        if 'payload' in meta:
            p = meta['payload']
            # Structure might be nested: {'payload': "String"} or {'payload': Dict}
            
            # Case 1: Direct Dict
            if isinstance(p, dict) and 'formula' in p:
                target_nid = nid
                target_payload = p
                break
                
            # Case 2: Nested Payload Key
            if isinstance(p, dict) and 'payload' in p:
                inner = p['payload']
                if isinstance(inner, str) and 'formula' in inner:
                    # Found it in string representation
                    target_nid = nid
                    # Hacky extract for demo
                    start = inner.find("'formula': '") + 12
                    end = inner.find("'", start)
                    formula_text = inner[start:end]
                    target_payload = {'formula': formula_text}
                    break
                
    if not target_nid:
        print("❌ No formula nodes found. (Did Genesis V5 work?)")
        return

    print(f"👉 Target Concept Found: {target_nid}")
    print(f"   Hidden Definition: {target_payload['formula']}")
    
    # 2. Simulate "Thinking" (Gravity)
    # We find WHO is connected to this formula.
    # Because 'ignite_gravity' linked similar vectors.
    print("\n🌊 Igniting Gravity to find Context...")
    neighbors = graph.get_neighbors(target_nid, top_k=3)
    
    print(f"   Neighbors of {target_nid}:")
    for n_id, dist in neighbors:
        print(f"   - {n_id} (Dist: {dist:.4f})")
        
    # 3. Formulate Thought
    # "The concept {Neighbor} is mathematically described by {Formula}"
    
    context_node = neighbors[0][0] # First neighbor
    formula = target_payload['formula']
    
    print("\n🗣️ ELYSIA'S RESPONSE (Utilization):")
    print(f"   'To understand [{context_node}], we must look at its underlying structure.")
    print(f"    The Graph reveals a connection to [{target_nid}].")
    print(f"    Retrieving definition... It is governed by the equation: \"{formula}\".'")
    print("\n✅ Verification:")
    print("   The 'Empty Node' was read. The 'Parameter' became 'Knowledge'.")

if __name__ == "__main__":
    demo()
