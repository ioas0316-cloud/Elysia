"""
Test: Quantum Intent Modulation
===============================
Verifies that QuantumObserver intent shifts the Cognitive Map indices.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

from Core.S1_Body.L6_Structure.M1_Merkaba.hypersphere_field import HyperSphereField
import time

def test_intent():
    print("=== Quantum Intent Verification ===\n")
    
    field = HyperSphereField()
    if not field._ppe_enabled:
        print("❌ PPE not enabled.")
        return

    print("1. Baseline Projection (No Intent)")
    field.observer.manifest_intent(None)
    base_proj = field.project_cognitive_map()
    print(f"   Theta: {base_proj['theta']:.3f} | Phi: {base_proj['phi']:.3f}")
    
    print("\n2. Manifest Intent: 'Logic' (Target Q1)")
    field.observer.manifest_intent("Show me the Logic")
    logic_proj = field.project_cognitive_map()
    print(f"   Theta: {logic_proj['theta']:.3f} | Phi: {logic_proj['phi']:.3f}")
    print(f"   Intent Active: {logic_proj['intent']}")
    
    print("\n3. Manifest Intent: 'Memory' (Target Q3)")
    field.observer.manifest_intent("Show me the Memory")
    mem_proj = field.project_cognitive_map()
    print(f"   Theta: {mem_proj['theta']:.3f} | Phi: {mem_proj['phi']:.3f}")
    print(f"   Intent Active: {mem_proj['intent']}")
    
    # Verification Logic
    # Q1 Logic should shift Phi/Theta 
    if logic_proj['phi'] != base_proj['phi']:
        print("\n✅ Intent Modulation Confirmed (Coordinates Shifted).")
    else:
        print("\n❌ Intent Modulation Failed (Coordinates Static).")

if __name__ == "__main__":
    test_intent()
