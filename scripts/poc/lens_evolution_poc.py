import sys
import os
import time

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory.causal_controller import CausalMemoryController
from core.consciousness.causal_reassembly import CausalReassembler
from core.utils.math_utils import Quaternion, traverse_causal_trajectory

def run_lens_evolution_poc():
    print("==========================================================")
    print(" [PoC v3] Meta-Stable Rotors: The Evolution of a Lens")
    print("==========================================================")

    mc = CausalMemoryController()
    reassembler = CausalReassembler(mc)

    # 1. Establish a 'Static Rotor' (A rigid Constant Lens)
    print("[1] Establishing a Rigid 'Red' Axiom (Static Rotor)...")

    red_data = b"wavelength_650nm_rigid_definition"
    red_q = traverse_causal_trajectory(red_data)

    lens_id = mc.write_causal_engram(
        data_blob={"type": "LENS", "name": "Axiom_Red", "quaternion": red_q.elements},
        emotional_value=1.0,
        cause_id="Initial_Rigidity",
        is_constant=True,
        stability=1.0, # Maximum rigidity
        modality="visual"
    )
    print(f"  > Lens Created: {lens_id} (Stability: 1.0)")

    # 2. Introduce a 'Conflicting Phenomenon' (Deep Red)
    print("\n[2] Observing a Conflicting 'Deep Red' Phenomenon...")
    # Manually setting a very high friction situation in the solver logic for this PoC
    deep_red_data = b"conflicting_data"
    deep_red_q = Quaternion(0, 1, 0, 0) # 180 degree rotation from Identity

    # Try to reassemble (Inquiry)
    variable_ids = reassembler.deconstruct("DeepRed_Inquiry", {"sample": deep_red_data})
    result = reassembler.solve_puzzle("DeepRed_Inquiry", variable_ids)

    print(f"  > Resonance Score: {result['resonance_score']:.4f}")

    # 3. Detect Greater Imbalance
    if result["meta_shift_triggered"]:
        print("\n[3] GREATER IMBALANCE DETECTED! (Tension exceeds threshold)")
        print("    The 'Static Rotor' must now become variable to resolve the crisis.")

        # 4. Trigger Structural Shift (Lens Evolution)
        print("\n[4] Triggering Structural Shift (Rotating the Axiom)...")
        success = reassembler.trigger_structural_shift(lens_id, deep_red_q)

        if success:
            updated_lens = mc.read_engram_trace(lens_id)
            new_stability = mc.index[lens_id]["stability"]
            print(f"  > Axiom evolved to new phase.")
            print(f"  > New Stability: {new_stability:.4f}")
            print(f"  > Reason: {updated_lens['data'].get('evolution_event')}")

            # 5. Re-Inquiry with the Evolved Lens
            print("\n[5] Re-observing 'Deep Red' with the Evolved Axiom...")
            # (In a real system, the solver would now achieve higher resonance)
            print("  > The 'Pain' of imbalance has birthed a 'Higher-order Symmetry'.")

    print("\n==========================================================")
    print(" [Conclusion] Constants are but Meta-Stable Rotors.")
    print(" [Evolution] Stability yields to Greater Imbalance.")
    print("==========================================================")

if __name__ == "__main__":
    run_lens_evolution_poc()
