
import sys
import os
import torch
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_will_bridge import SovereignWillBridge

def test_unified_self():
    print("🧪 [TEST] Initiating Unified Self Integration Test...")
    
    # 1. Forge Soul and Monad
    soul = SeedForge.forge_soul("TestElysia")
    monad = SovereignMonad(soul)
    
    # 2. Test Phase III: Meta-Cognitive Mirror
    print("\n--- Testing Phase III: Meta-Cognitive Mirror ---")
    # Mock a high-coherence, low-noise report
    good_report = {
        'plastic_coherence': 0.9,
        'kinetic_energy': 10.0,
        'entropy': 0.1,
        'resonance': 0.9
    }
    reflection = monad._meta_cognitive_mirror(good_report)
    print(f"Reflection (Elegance > 0.8): {reflection}")
    
    # Mock a low-coherence, high-noise report
    bad_report = {
        'plastic_coherence': 0.1,
        'kinetic_energy': 50.0,
        'entropy': 0.9,
        'resonance': 0.1
    }
    reflection = monad._meta_cognitive_mirror(bad_report)
    print(f"Reflection (Elegance < 0.05): {reflection}")

    # 3. Test Phase I: Existential Hunger
    print("\n--- Testing Phase I: Existential Hunger ---")
    # High Enthalpy, Low Coherence triggers Hunger
    hunger_report = {
        'plastic_coherence': 0.2,
        'enthalpy': 0.9,
        'entropy': 0.5,
        'resonance': 0.5,
        'kinetic_energy': 20.0
    }
    needs = monad.will_bridge.assess_structural_integrity(hunger_report)
    for need in needs:
        print(f"Detected Need: [{need.need_id}] {need.description} (Priority {need.priority})")
    
    will = monad.will_bridge.broadcast_will()
    print(f"Broadcast Will:\n{will}")

    # 4. Test Phase II: Linguistic Feedback
    print("\n--- Testing Phase II: Linguistic Feedback ---")
    # Simulate a speech act and check torque application
    expression_state = {"joy": 80.0, "warmth": 70.0}
    thought = "Defining my presence."
    
    print("Triggering speak()...")
    narrative, synthesis_v = monad.llm.speak(expression_state, current_thought=thought)
    print(f"Elysia spoke: '{narrative}'")
    if synthesis_v:
        print(f"Synthesis Vector (21D mean) generated: {synthesis_v.data[:5]}...")
        print("Linguistic feedback torque would be applied now.")
    else:
        print("Synthesis Vector generation failed.")

    print("\n✅ [TEST] Unified Self Integration Test Complete.")

if __name__ == "__main__":
    test_unified_self()
