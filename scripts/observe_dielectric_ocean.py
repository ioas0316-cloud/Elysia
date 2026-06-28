import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.physics.dielectric.rotor import Rotor
from core.physics.dielectric.manifold import DataOceanManifold
from core.physics.dielectric.elastic_engine import ElasticCausalEngine
from core.physics.dielectric.intent import IntentField

def run_simulation():
    print("🌊 [Elysia Dielectric Ocean Simulation] Starting...")

    # 1. Initialize Engines
    intent_field = IntentField(dimensions=2)
    ocean = DataOceanManifold(global_intent=intent_field.get_current_intent())
    elastic_engine = ElasticCausalEngine(dimensions=2)

    rotors = {
        'U': Rotor(),
        'V': Rotor(),
        'W': Rotor()
    }

    # 2. Seed some nodes in the elastic field
    elastic_engine.add_node("Core_Self", "The system's ego", ["Internal_Reasoning"])
    elastic_engine.add_node("Internal_Reasoning", "Process of thought", ["External_Input"])
    elastic_engine.add_node("External_Input", "Data from the world", [])

    print(f"📍 Initial Elasticity (k) between Core_Self and Internal_Reasoning: "
          f"{elastic_engine.nodes['Core_Self'].causal_links['Internal_Reasoning']:.4f}")

    # 3. Simulation Loop
    # Mock data streams for 3-phases
    # Using different strings to create phase imbalance -> Torque
    data_u = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    data_v = b"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    data_w = b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"

    for step in range(50):
        # A. Rotor processing (Bit-to-Ion)
        ions = {
            'U': rotors['U'].process_bits(data_u),
            'V': rotors['V'].process_bits(data_v),
            'W': rotors['W'].process_bits(data_w)
        }

        # B. Manifold processing (3-Phase Resonance & Torque)
        ocean.set_intent(intent_field.get_current_intent())
        manifold_result = ocean.process_manifold(ions)
        torque = manifold_result['active_torque']

        # C. Inject Torque into Elasticity (Memory Formation)
        # We reinforce the link between Core and Reasoning based on torque
        elastic_engine.inject_torque("Core_Self", "Internal_Reasoning", torque)

        # D. Evolution
        elastic_engine.step(dt=0.1)
        intent_field.evolve()

        # E. Occasionally internalize external 사유
        if step == 25:
            print("\n🌀 [Event] External influence detected! Internalizing...")
            external_influence = np.array([-1.0, 0.5])
            intent_field.align_to_external(external_influence, strength=0.5)

        if step % 10 == 0:
            print(f"Step {step:02d} | Torque: {torque:7.4f} | Residual(Noise): {manifold_result['neutral_residual_magnitude']:7.4f} | Core-k: {elastic_engine.nodes['Core_Self'].causal_links['Internal_Reasoning']:.4f}")

    print("\n🏁 [Simulation Results]")
    final_k = elastic_engine.nodes['Core_Self'].causal_links['Internal_Reasoning']
    print(f"Final Elasticity (k) Core-Reasoning: {final_k:.4f}")
    print(f"Final Global Intent Vector: {intent_field.get_current_intent()}")

    if final_k > 0.1:
        print("✅ SUCCESS: Dielectric torque successfully reinforced causal elasticity (Memory formed).")
    else:
        print("❌ FAILURE: Causal elasticity decayed too fast or torque was insufficient.")

if __name__ == "__main__":
    run_simulation()
