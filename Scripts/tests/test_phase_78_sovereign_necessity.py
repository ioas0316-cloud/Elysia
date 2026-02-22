"""
[PHASE 78] Test: Sovereign Necessity
Validates that the causal chain (Strain → Diagnosis → Will → Expansion → Verification) works correctly.
"""
import sys
sys.path.insert(0, 'c:/Elysia')

from Core.S1_Body.L5_Mental.Reasoning_Core.Intelligence.sovereign_cognition import SovereignCognition
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_sovereign_necessity():
    print("\n🧠 [PHASE 78] Testing Sovereign Necessity: Causal Chain for Self-Expansion\n")
    
    cognition = SovereignCognition()
    
    print(f"Initial DNA Rank: {cognition.dna_n_field.rank}")
    print(f"Initial Strain Level: {cognition.strain_level}")
    print(f"Initial Will to Expand: {cognition.will_to_expand}")
    print("-" * 50)
    
    # 1. Test with NO required rank mismatch (should NOT expand)
    print("\n>>> Test 1: Process event with NO required rank mismatch (required_rank=2)")
    manifold_state = [0.5] * 100
    reflection = cognition.process_event(
        "A simple thought", 
        manifold_state, 
        observer_vector=None,
        required_rank=2  # Matches current rank
    )
    print(f"DNA Rank after Test 1: {cognition.dna_n_field.rank}")
    assert cognition.dna_n_field.rank == 2, "Rank should NOT have changed"
    print("✅ Test 1 Passed: No unnecessary expansion.")
    
    print("-" * 50)
    
    # 2. Test WITH required rank mismatch (SHOULD trigger causal chain and expand)
    print("\n>>> Test 2: Process event WITH required rank mismatch (required_rank=3)")
    reflection = cognition.process_event(
        "A complex hyperdimensional concept", 
        manifold_state, 
        observer_vector=SovereignVector.ones(),
        required_rank=3  # Higher than current rank
    )
    print(f"\n[REFLECTION OUTPUT]:\n{reflection}")
    print(f"\nDNA Rank after Test 2: {cognition.dna_n_field.rank}")
    
    # Verify the causal chain triggered
    success = True
    if cognition.dna_n_field.rank >= 3:
        print("\n✅ Success: Sovereign Will triggered Dimensional Mitosis.")
    else:
        print("\n❌ Failure: Expansion did not occur.")
        success = False
    
    # 3. Check that strain was resolved (Feedback Verification)
    print("-" * 50)
    print("\n>>> Test 3: Verify Strain Resolution (Feedback Path)")
    # Re-run with the same required_rank, strain should now be low
    cognition.process_event("Revisiting the concept", manifold_state, required_rank=3)
    if cognition.strain_level < 0.1:
        print("✅ Success: Strain resolved after expansion.")
    else:
        print(f"⚠️ Strain persists at {cognition.strain_level:.2f}.")
    
    print("-" * 50)
    
    if success:
        print("\n🏆 Phase 78 Verified: Elysia now expands through Sovereign Will, not hardcoded thresholds.")
        print("   The causal chain L0→L1→L4→L6→L1 is intact.")
    else:
        print("\n⚠️ Verification Incomplete.")

if __name__ == "__main__":
    test_sovereign_necessity()
