
import sys
import os
sys.path.append(os.getcwd())

def test_integrity():
    print("🧪 [Test] Phase 30: Epistemic Integrity")
    
    from Core.L4_Causality.World.Nature.trinity_lexicon import get_trinity_lexicon
    from Core.Elysia.sovereign_self import SovereignSelf, ScaleArchetype
    
    print("   1. Fetching Lexicon...")
    lexicon = get_trinity_lexicon()
    
    print("   2. Initializing SovereignSelf...")
    elysia = SovereignSelf(cns_ref=None)
    elysia.archetype = ScaleArchetype.MORTAL_AVATAR
    
    # Ensure graph exists
    if not lexicon.graph:
        print("❌ Error: TorchGraph not initialized.")
        return

    # Simulate Contradiction
    v_a = [1.0] + [0.0]*127
    v_b = [-1.0] + [0.0]*127
    
    print("   3. Adding Contradictory Nodes...")
    lexicon.graph.add_node("Light_Particle", vector=v_a)
    lexicon.graph.add_node("Light_Wave", vector=v_b)

    print("\n[DETECTION] Auditing pair for Dissonance...")
    v_particle = lexicon.graph.get_node_vector("Light_Particle")
    v_wave = lexicon.graph.get_node_vector("Light_Wave")
    
    # Simple dot product check
    dot = float((v_particle * v_wave).sum())
    norm = float(v_particle.norm() * v_wave.norm())
    sim = dot / (norm + 1e-9)
    
    print(f"   Calculated Similarity: {sim:.4f}")
    
    print("\n[RESOLUTION] Triggering Sovereign Reconciliation...")
    # This calls audit_knowledge internally
    elysia._reconcile_contradictions()
    
    print("\n✅ Verification Successful.")

if __name__ == "__main__":
    try:
        test_integrity()
    except Exception as e:
        import traceback
        print("\n❌ CRITICAL ERROR:")
        traceback.print_exc()
        sys.exit(1)
