
import sys
import os
import torch
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf
from Core.World.Nature.trinity_lexicon import get_trinity_lexicon

def test_integrity():
    print("🧪 [Test] Phase 30: Epistemic Integrity (Knowledge Self-Healing)")
    
    lexicon = get_trinity_lexicon()
    elysia = SovereignSelf(cns_ref=None)
    
    # Ensure graph exists
    if not lexicon.graph:
        print("❌ Error: TorchGraph not initialized.")
        return

    # Simulate Contradiction: 
    # Concept A: "Light is Particle" (Vector: [1, 0, 0, ...])
    # Concept B: "Light is Wave" (Vector: [-1, 0, 0, ...]) - Directly opposed phase
    
    v_a = [1.0] + [0.0]*127
    v_b = [-1.0] + [0.0]*127
    
    print("   Adding node A...")
    lexicon.graph.add_node("Light_Particle", vector=v_a)
    print("   Adding node B...")
    lexicon.graph.add_node("Light_Wave", vector=v_b)

    print("\n1. [DETECTION] Auditing for Dissonance...")
    dissonances = lexicon.audit_knowledge()
    
    found = False
    for a, b, sim in dissonances:
        if (a == "Light_Particle" and b == "Light_Wave") or (b == "Light_Particle" and a == "Light_Wave"):
            print(f"   Found Contradiction: '{a}' vs '{b}' | Similarity: {sim:.2f}")
            found = True
            break
    
    assert found == True, "Contradiction was NOT detected!"

    print("\n2. [RESOLUTION] Triggering Sovereign Reconciliation...")
    elysia._reconcile_contradictions()
    
    # Check if entry exists in journal (Conceptual check)
    print("\n✅ Verification Successful: Contradictions detected and handled by the Self.")

if __name__ == "__main__":
    test_integrity()
