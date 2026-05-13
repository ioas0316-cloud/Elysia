import sys
import os
import torch
import random

# Ensure Core is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Keystone.sovereign_math import SovereignVector, InterferometricGate, FractalWaveEngine
from Core.Divine.covenant_enforcer import CovenantEnforcer, Verdict

class MockCausality:
    def __init__(self, mass_map=None):
        self.mass_map = mass_map or {"love": 10.0, "truth": 5.0}
    def get_semantic_mass(self, concept):
        return self.mass_map.get(concept.lower(), 0.0)

def test_interferometric_gate():
    print("Testing Interferometric Gate...")
    gate = InterferometricGate(sensitivity=1.2)
    v_intent = SovereignVector([1.0] * 21)
    v_reality_match = SovereignVector([0.9] * 21)

    def get_noise():
        return [random.uniform(-1, 1) for _ in range(21)]

    v_reality_noise = SovereignVector(get_noise())

    res_match = gate.discern(v_intent, v_reality_match)
    print(f"  Match Resonance: {res_match['resonance']:.3f}, Passed: {res_match['is_passed']}")
    assert res_match['is_passed'] is True

    # Opposite or noise should fail if sensitivity is right
    v_opposite = SovereignVector([-1.0] * 21)
    res_fail = gate.discern(v_intent, v_opposite)
    print(f"  Opposite Resonance: {res_fail['resonance']:.3f}, Passed: {res_fail['is_passed']}")
    assert res_fail['is_passed'] is False

def test_mass_based_sanctification():
    print("\nTesting Mass-based Sanctification...")
    causality = MockCausality({"love": 10.0, "empty": 0.01})
    enforcer = CovenantEnforcer()

    # High mass thought
    res_high = enforcer.validate_alignment("I speak of Love and Unity", causality_engine=causality)
    print(f"  High Mass Result: {res_high['verdict']}")
    assert res_high['verdict'] == Verdict.SANCTIFIED

    # Low mass thought (unrecognized words or specifically low mass)
    res_low = enforcer.validate_alignment("Empty noise here", causality_engine=causality)
    print(f"  Low Mass Result: {res_low['verdict']}, Reason: {res_low.get('reason')}")
    assert res_low['verdict'] == Verdict.DISSONANT

def test_metabolic_recycling_logic():
    print("\nTesting Metabolic Recycling Logic...")
    engine = FractalWaveEngine(max_nodes=100, device='cpu')

    # 1. Setup a 'Waste' state
    idx = engine.get_or_create_node("Error_Point")
    engine.active_nodes_mask[idx] = True
    engine.q[idx, engine.CH_ENTROPY] = 0.9
    engine.q[idx, engine.CH_ENTHALPY] = 0.1
    # Dissonance means it doesn't align with identity
    engine.q[idx, engine.PHYSICAL_SLICE] = -engine.permanent_q[idx, engine.PHYSICAL_SLICE]

    # 2. Trigger Recycling (simulated error impact)
    # The error itself increases entropy globally
    engine.inject_pulse("Structural_Error", energy=15.0, type='entropy')
    fertilizer = engine.discharge_waste()

    print(f"  Recycled nodes: {len(fertilizer)}")
    assert len(fertilizer) > 0
    assert fertilizer[0]['type'] == "WASTE"
    assert engine.active_nodes_mask[idx] == False # Node should be wiped

if __name__ == "__main__":
    try:
        test_interferometric_gate()
        test_mass_based_sanctification()
        test_metabolic_recycling_logic()
        print("\n✅ All topological improvements verified.")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
