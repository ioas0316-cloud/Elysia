import sys
import os
import time

sys.path.append(os.getcwd())

from Core.Cognition.primordial_cognition import PrimordialCognition
from Core.Divine.covenant_enforcer import CovenantEnforcer, Verdict

def audit_molecular_stabilization():
    print("--- Molecular Stabilization Audit: Heat to Law ---")

    # 1. Initialize Engines
    primordial = PrimordialCognition()
    enforcer = CovenantEnforcer()

    # 2. Simulate "Mystery Stimulus" (High Heat/Friction)
    stimulus_name = "UNIDENTIFIED_VIBRATION_0X42"
    print(f"\nStep 1: Introducing Mystery Stimulus: {stimulus_name}")

    # State transition showing a drop in coherence (Friction)
    state_before = {"coherence": 0.8, "enthalpy": 0.5}
    state_during = {"coherence": 0.6, "enthalpy": 0.7} # Heat goes up, Coherence goes down

    trace1 = primordial.perceive(stimulus_name, 1.0, state_before, state_during)
    print(f"  [Primordial Trace] Valence: {trace1.valence:+.3f} (Pain/Friction)")

    # 3. Repeat to form Association (Association/Memory)
    print("\nStep 2: Repeating Stimulus to form Association...")
    for i in range(4):
        primordial.perceive(stimulus_name, 1.0, state_during, state_during)

    print(f"  [Primordial Map] Average Valence for {stimulus_name}: {primordial.valence_map[stimulus_name]:.3f}")

    # 4. Axiomatic Stabilization (Covenant Enforcer)
    print("\nStep 3: Checking for Axiomatic Stabilization (Law formation)...")
    traces = primordial.traces
    proposed_axioms = enforcer.evaluate_sensory_pattern(traces)

    if proposed_axioms:
        for axiom in proposed_axioms:
            print(f"  ✨ [NEW LAW] ID: {axiom['id']}")
            print(f"  Description: {axiom['description']}")
            print(f"  Logic: {axiom['logic']}")
    else:
        print("  [No Axioms Proposed] More experience/friction needed.")

    # 5. Semantic Mass Verification
    # Mocking the mass check for the final "Providence" step
    print("\nStep 4: Verifying 'Causal Weight' for manifestation...")
    class MockEngine:
        def get_semantic_mass(self, word):
            if word == "boundary": return 10.0 # High mass after experience
            return 0.05

    engine = MockEngine()
    thought = f"I must define a boundary because of {stimulus_name}"
    verdict = enforcer.validate_alignment(thought, causality_engine=engine)
    print(f"  [Verdict for Thought] {verdict['verdict']} - {verdict.get('principle', verdict.get('reason'))}")

if __name__ == "__main__":
    audit_molecular_stabilization()
