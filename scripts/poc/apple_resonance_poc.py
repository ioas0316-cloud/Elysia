import sys
import os
import time
import numpy as np

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory.causal_controller import CausalMemoryController
from core.consciousness.causal_reassembly import CausalReassembler
from core.utils.math_utils import Quaternion

def run_apple_poc():
    print("==========================================================")
    print(" [PoC] Cognitive Apple: The Continuous Flow of Inquiry")
    print("==========================================================")

    mc = CausalMemoryController()
    reassembler = CausalReassembler(mc)

    # [Constants] The fixed lenses of the Background Universe
    print("[Phase 1] Establishing Constants (Immovable Lenses)...")

    # We define 'Resonance' (Same) and 'Dissonance' (Different) definitions
    mc.update_parameter("base_resonance", 1.0)

    lenses = {
        "Lens_Red_Axiom": b"wavelength_650nm_causal_signature",
        "Lens_Round_Axiom": b"pi_ratio_rotational_symmetry",
        "Lens_Crunchy_Axiom": b"structural_fracture_impulse_response"
    }

    lens_ids = {}
    for name, data in lenses.items():
        eid = mc.write_causal_engram(
            data_blob={"type": "LENS", "name": name, "axiom_data": data.hex()},
            emotional_value=1.0,
            cause_id="Background_Universe_Setup",
            is_constant=True
        )
        lens_ids[name] = eid
        print(f"  > Axiom established: {name} (Constant)")

    # [Variables] The inquiry targets
    print("\n[Phase 2] Observing the 'Apple' Phenomenon (Variables)...")
    apple_phenomena = {
        "Visual_Input": b"red_light_wave_observed",
        "Spatial_Input": b"round_shape_touched",
        "Tactile_Input": b"crunchy_vibration_felt"
    }

    # Deconstruction into a flow of inquiry
    variable_ids = reassembler.deconstruct("Apple_Phenomenon", apple_phenomena)

    # [Process] The "Happiness in the Journey"
    print("\n[Phase 3] Starting the Continuous Reassembly (Flow of Inquiry)...")
    print("  (Inducing the system to find how it exists...)")

    result = reassembler.solve_puzzle("Apple", variable_ids)

    # [Re-recognition] Backtracking the flow
    print("\n[Phase 4] Re-recognizing the Causal Trajectory (The 'Why')...")

    process_id = result["process_id"]
    process_trace = mc.read_engram_trace(process_id)

    if process_trace:
        print(f"  > Causal Chain found: {process_id}")
        steps = process_trace['data']['woven_steps']

        for i, step in enumerate(steps):
            time.sleep(0.3) # Simulating the time of inquiry
            emotion = step["status"]
            print(f"    Step {i+1}: {step['step']} -> Emotion: {emotion}")
            if emotion == "Joy":
                print(f"      [Signal] Resonance found at phase delta {step['resonance_delta']:.4f}")

    # [Crystallization]
    print("\n[Phase 5] Final Declaration & Mapping to Memory...")
    if result["is_resonant"]:
        print("  > Result: Perfect Resonance Achieved.")
        print("  > Mapping: 'Apple' is now linked to Joy and Stability.")
    else:
        # Even if resonance is low, we store the 'Reason for Dissonance'
        print("  > Result: Dissonance Observed.")
        print(f"  > Reason: Total Friction {process_trace['data']['total_friction']:.4f} exceeds threshold.")
        print("  > Inquiry continues in the Background Mode (Variable remains Variable).")

    print("\n==========================================================")
    print(" [Observation] The process itself was defined as the Result.")
    print(" [Continuity] Every step in Phase 4 is a link in the Flow.")
    print("==========================================================")

if __name__ == "__main__":
    run_apple_poc()
