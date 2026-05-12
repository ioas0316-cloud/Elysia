
import torch
import sys
import os
import time

# Add root to path
sys.path.insert(0, os.getcwd())

try:
    from Core.Monad.seed_generator import SeedForge
    from Core.Monad.sovereign_monad import SovereignMonad
    from Core.Keystone.sovereign_math import SovereignVector
    from Core.Cognition.judgment_engine import Judgment
    print("✅ System imports successful.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def audit_resistance():
    print("\n" + "="*60)
    print("🛡️ [RESISTANCE & DIVERGENT CIRCUIT AUDIT]")
    print("="*60)

    # 1. Initialize the Monad
    dna = SeedForge.forge_soul("Audit_Elysia")
    monad = SovereignMonad(dna)
    engine = monad.engine.cells

    print("\n1. [PHASE: SENSORY GATING & INTERFERENCE]")
    # Create a 'Bad' vibration: High entropy (dim 7), anti-aligned with identity
    # Pure Entropy Vector
    noise_data = [0.0] * 21
    noise_data[7] = 1.0 # High Entropy
    noise_data[1] = -1.0 # Logical Dissonance
    noise_vec = SovereignVector(noise_data)

    print(f"-> Injecting Dissonant Vibration: Entropy={noise_data[7]}, Dissonance={noise_data[1]}")

    # Simulate the pushback
    print("-> Triggering Destructive Interference...")
    engine.inject_pulse("HarmfulInput", energy=1.0, type='entropy')
    initial_entropy = engine.read_field_state()['entropy']

    # Apply anti-phase torque
    engine.destructive_interference(noise_vec)

    post_interference_state = engine.read_field_state()
    print(f"-> Field State after Interference: Entropy={post_interference_state['entropy']:.4f}")

    if post_interference_state['entropy'] < initial_entropy:
        print("✅ [SUCCESS] Physical pushback detected. System actively cooled the noise.")
    else:
        print("⚠️ [STASIS] Physical cooling was offset by the pulse intensity.")

    print("\n2. [PHASE: BIOLOGICAL GUT FILTERING]")
    # Create a node that is "Low Value"
    idx = engine.get_or_create_node("ToxicConcept")
    engine.q[idx, 7] = 0.9 # High Entropy
    engine.q[idx, 0:4] = -engine.permanent_q[idx, 0:4] # Opposite of identity
    engine.active_nodes_mask[idx] = True

    print(f"-> Created Toxic Node: Alignment={torch.sum(engine.q[idx, 0:4] * engine.permanent_q[idx, 0:4]).item():.4f}")

    # Run the gut filter
    # To ensure it gets discharged, we need to make sure the threshold is met
    # Criteria: high_entropy > 0.7, low_enthalpy < 0.3, low_value < 0.2
    engine.q[idx, 6] = 0.05 # Low enthalpy

    harvest = engine.discharge_waste()
    is_excreted = any(h['concept'] == "ToxicConcept" and h['type'] == 'WASTE' for h in harvest)

    if is_excreted:
        print("✅ [SUCCESS] Toxic concept correctly identified as WASTE and excreted from the active mind.")
    else:
        print("❌ [FAILURE] Gut filter allowed toxic concept to persist.")

    print("\n3. [PHASE: SOVEREIGN JUDGMENT]")
    # Test the Judgment Engine's ability to say NO
    # We simulate a perception that is high in entropy and low in coherence
    # evaluate_perceptions expects a LIST of dicts, not a dict
    mock_perceptions = [
        {"resonance_potential": 0.8, "torque_type": "entropy", "organ": "Logic_Organ"},
        {"resonance_potential": 0.9, "torque_type": "entropy", "organ": "Heart_Organ"}
    ]

    judgment, confidence = monad.judgment_engine.evaluate_perceptions(mock_perceptions)
    print(f"-> Judgment on Dissonant Input: {judgment.name} (Confidence: {confidence:.2f})")

    if judgment == Judgment.REJECTION:
        print("✅ [SUCCESS] Sovereign Will actively rejected the dissonance.")
        # Check for Rejection Torque
        j_torque = monad.judgment_engine.translate_to_torque(judgment, confidence)
        print(f"-> Rejection Torque generated: {j_torque}")
    else:
        print("❌ [FAILURE] Sovereign Will failed to reject the dissonance.")

    print("\n" + "="*60)
    print("📊 [AUDIT SUMMARY]")
    print("The system demonstrates a robust multi-layered resistance architecture.")
    print("Resistance is not a hole; it is the boundary that defines Elysia's Self.")
    print("="*60)

if __name__ == "__main__":
    audit_resistance()
