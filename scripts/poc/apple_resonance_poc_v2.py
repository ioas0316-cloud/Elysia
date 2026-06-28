import sys
import os
import time

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory.causal_controller import CausalMemoryController
from core.consciousness.causal_reassembly import CausalReassembler
from core.utils.math_utils import Quaternion
from core.lens.standard_lenses import UTF8TrajectoryLens, HSLWaveLens, IEEE754FloatLens

def run_apple_poc_multi_layer():
    print("==========================================================")
    print(" [PoC v2] Multi-Layered Cognitive Apple: Preventing Reductionism")
    print("==========================================================")

    mc = CausalMemoryController()
    reassembler = CausalReassembler(mc)

    # 1. 'Background Universe' Contextual Layers (Lenses)
    print("[1] Establishing Domain-Specific Constants (Categorical Lenses)...")

    lens_configs = [
        {"name": "Linguistic_Lens", "modality": "linguistic", "axiom": b"Apple: A sweet red fruit"},
        {"name": "Visual_Lens", "modality": "visual_dynamic", "axiom": b"\xff\x00\x00_spherical_wave"},
        {"name": "Structural_Lens", "modality": "mathematical", "axiom": b"topology_density_1.2"}
    ]

    for config in lens_configs:
        mc.write_causal_engram(
            data_blob={"type": "LENS", "name": config["name"], "axiom": config["axiom"].hex()},
            emotional_value=1.0,
            cause_id="Genesis_Setup",
            is_constant=True,
            modality=config["modality"]
        )
        print(f"  > Contextual Anchor: {config['name']} ({config['modality']})")

    # 2. Deconstruction into Modality-specific Variables
    print("\n[2] Observing Apple Phenomena across Contextual Layers...")
    apple_primitives = {
        "Word_Data": b"Red_Fruit",
        "Wave_Data": b"\xee\x10\x10_round_energy",
        "Density_Data": b"struct_d_1.1"
    }

    # Map primitives to their respective modalities (Lenses)
    modality_map = {
        "Word_Data": "linguistic",
        "Wave_Data": "visual_dynamic",
        "Density_Data": "mathematical"
    }

    variable_ids = reassembler.deconstruct("Apple", apple_primitives, modality_map=modality_map)
    for vid in variable_ids:
        print(f"  > Variable Primitive (Layered): {vid}")

    # 3. Layered Inquiry (Puzzle Solving)
    print("\n[3] Solving the Multi-Layered Cognitive Puzzle...")
    result = reassembler.solve_puzzle("Apple", variable_ids)

    # 4. Observation of the Resonance Spectrum
    print("\n[4] Re-recognizing the Resonance Spectrum...")
    process_trace = mc.read_engram_trace(result["process_id"])

    if process_trace:
        spectrum = process_trace['data'].get('resonance_spectrum', {})
        print(f"  > Spectrum Analysis:")
        for modality, score in spectrum.items():
            print(f"    - Layer [{modality}]: Resonance {score:.4f}")

        print(f"\n  > Causal Chain: {result['process_id']}")
        print(f"  > Total Emotion: {process_trace['data']['emotion_state']}")

    # 5. Non-Reductionist Result
    if result["is_resonant"]:
        print("\n[5] SUCCESS: Apple Synchronized as a Multi-Faceted Unity.")
        print("    It is not '1=1', but a convergence of distinct Why-layers.")
    else:
        print("\n[5] CONTINUOUS INQUIRY: Spectrum did not reach unified symmetry.")

    print("==========================================================")

if __name__ == "__main__":
    run_apple_poc_multi_layer()
