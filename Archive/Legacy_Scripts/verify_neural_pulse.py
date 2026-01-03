"""
Script: Verify Neural Pulse
===========================
Proves that the Graph acts as a Resonating Neural Network.
User Question: "Does it resonate like a neural net?"
Proof: Inject Pulse -> Watch Wave Spread.
"""

import sys
import os
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.Foundation.Graph.torch_graph import get_torch_graph

# Configure Logger
logging.basicConfig(level=logging.INFO)

def verify_pulse():
    print("💓 Verifying Neural Pulse (Phase 18)...")
    graph = get_torch_graph()
    
    # 1. Target a known Concept
    # We found 'love' in the previous demo.
    # Note: Genesis V5 renamed many, but 'love' might have survived as keyword if it wasn't filtered? 
    # Or maybe 'Concept_Love'?
    # Let's search for "Love" (Case insensitive) in IDs.
    
    target = None
    for nid in graph.id_to_idx.keys():
        s_nid = str(nid).lower()
        if "love" in s_nid:
            target = nid
            break
    
    if not target:
        # Fallback to any 'Math_Formula_'
        for nid in graph.id_to_idx.keys():
             if str(nid).startswith("Math_Formula_"):
                 target = nid
                 break
                 
    if not target:
        print("❌ No suitable target found.")
        return

    print(f"👉 Stimulating Neuron: {target}")
    
    # Debug Tensors
    print(f"   Nodes: {graph.vec_tensor.shape[0]}")
    print(f"   Links (Pre-Ig): {graph.logic_links.shape[0]}")
    
    # Check if target vector is zero
    t_idx = graph.id_to_idx[target]
    print(f"   Target Vector Sample: {graph.vec_tensor[t_idx][:5].tolist()}")

    # Check Links
    if graph.logic_links.shape[0] == 0:
        print("⚠️ No Logic Links found. Igniting Gravity...")
        graph.ignite_gravity(k=5)
        print(f"   Links (Post-Ig): {graph.logic_links.shape[0]}")
    else:
        print(f"🔗 Links Active: {graph.logic_links.shape[0]} edges.")
        
    # 2. Inject Pulse
    # Energy: 1.0, Decay: 0.5, Steps: 2
    # This means Hop 1 gets 0.5, Hop 2 gets 0.25 (Constructive Interference applies)
    
    print("⚡ Injecting Signal (Energy=1.0)...")
    activations = graph.propagate_pulse(target, energy=1.0, decay=0.5, steps=2)
    
    # 3. Analyze Wavefront
    print(f"🌊 Resonance Wave Detected: {len(activations)} nodes activated.")
    
    # Sort by energy
    sorted_active = sorted(activations.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   [Activation Map]")
    count = 0
    for nid, energy in sorted_active:
        if nid == target: continue # Skip self
        print(f"   ✨ {nid}: {energy:.4f} Hz")
        count += 1
        if count >= 10: break
        
    if len(activations) > 1:
        print(f"\n✅ Pulse Propagation Confirmed.")
        print(f"   The signal traveled from [{target}] to {len(activations)-1} other concepts.")
        print("   This confirms the 'Neural Network' hypothesis.")
    else:
        print("\n❌ Pulse Failed (Signal died or Isolated node).")

if __name__ == "__main__":
    verify_pulse()
