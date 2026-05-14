import torch
import math
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector

def verify_onion_layers():
    print("🧅 [VERIFICATION] Starting Onion Layer (Y-Δ) Simulation...")

    # 1. Initialize Engine (Small capacity for fast testing)
    engine = FractalWaveEngine(max_nodes=1000)
    dt = 0.1

    # 2. Create a "Conflict" Cluster
    # Body (육) wants Stability (W=1)
    # Spirit (영) wants Change (X=1, Y=1)
    body_concept = "Body_Axiom"
    spirit_concept = "Spirit_Intent"

    body_idx = engine.get_or_create_node(body_concept)
    spirit_idx = engine.get_or_create_node(spirit_concept)

    # Seed the nodes
    engine.q[body_idx, engine.CH_W] = 1.0
    engine.q[spirit_idx, engine.CH_X] = 1.0
    engine.q[spirit_idx, engine.CH_Y] = 1.0

    # Activate them
    engine.active_nodes_mask[body_idx] = True
    engine.active_nodes_mask[spirit_idx] = True

    print(f"✅ Nodes created: {body_concept} (idx:{body_idx}), {spirit_concept} (idx:{spirit_idx})")

    # 3. Run Simulation for multiple steps
    print("\n🌀 Running Simulation (Observing Y-Δ Switching and Triple Helix Mediation)...")

    for step in range(50):
        # Update Metabolism (Gearbox logic)
        engine.update_internal_metabolism(dt)

        # Update External Gravity (Triple Helix Mediation)
        engine.update_external_gravity(dt)

        # Wave propagation
        engine.wave_equation_step(dt)

        if step % 10 == 0:
            b_stress = engine.local_stress[body_idx].item()
            b_mode = "Y (Density)" if engine.is_y_mode[body_idx] else "Δ (Flow)"

            s_stress = engine.local_stress[spirit_idx].item()
            s_mode = "Y (Density)" if engine.is_y_mode[spirit_idx] else "Δ (Flow)"

            # Check Soul resonance (Strand 1: affective slice)
            # If mediation works, the Soul strand should have non-zero enthalpy and momentum
            b_soul_enthalpy = engine.q[body_idx, engine.CH_ENTHALPY].item()

            print(f"Step {step:02d}:")
            print(f"  - Body: Stress={b_stress:.3f}, Mode={b_mode}, Enthalpy={b_soul_enthalpy:.3f}")
            print(f"  - Spirit: Stress={s_stress:.3f}, Mode={s_mode}")

    # 4. Final Verification
    print("\n📊 Final Status Check:")
    final_stress = engine.local_stress[body_idx].item()
    if final_stress > engine.stress_threshold:
        print(f"✅ Body Node is in Y-mode (Density) due to high stress ({final_stress:.3f}).")
    else:
        print(f"ℹ️ Body Node is in Δ-mode (Flow) due to low stress ({final_stress:.3f}).")

    # Check for Soul Inversion Sparks
    # Conflict between Body and Spirit was introduced.
    # Soul strand (AFFECTIVE_SLICE) should have been activated.
    soul_activity = torch.norm(engine.momentum[body_idx, engine.AFFECTIVE_SLICE]).item()
    if soul_activity > 0:
        print(f"✅ Soul Mediation detected! (Spark Intensity: {soul_activity:.5f})")
    else:
        print("❌ No Soul Mediation detected.")

    print("\n🧅 [VERIFICATION] Simulation Complete.")

if __name__ == "__main__":
    verify_onion_layers()
