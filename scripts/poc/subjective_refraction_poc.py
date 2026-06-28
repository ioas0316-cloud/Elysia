import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.lens.dynamic_lenses import MemoryLens
from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel

def simulate_subjective_refraction():
    print("🌈 [PoC] Subjective Refraction: Narrative as Lens")
    print("==========================================================\n")

    # The external input: A "Red Circle" (could be a red apple)
    external_input = b"RED_ROUND_OBJECT_DATA"

    # 1. State A: System with "Lens of Life" (Red = Blood = Vitality)
    lens_life = MemoryLens("Lens_of_Life", reference_topology=abs(hash("BLOOD_LIFE")) % (2**32))

    # 2. State B: System with "Lens of Danger" (Red = Forbidden = Sin)
    lens_danger = MemoryLens("Lens_of_Danger", reference_topology=abs(hash("FORBIDDEN_SIN")) % (2**32))

    print(f"[Input] Incoming Waveform: '{external_input.decode()}'")

    # Observation through Lens A
    res_a = lens_life.decode(external_input)
    print(f"\n[Subject A] Perception through 'Lens of Life':")
    print(f"  - Tension: {res_a['tension']:.4f}")
    print(f"  - Reality Definition: {res_a['data']}")

    # Observation through Lens B
    res_b = lens_danger.decode(external_input)
    print(f"\n[Subject B] Perception through 'Lens of Danger':")
    print(f"  - Tension: {res_b['tension']:.4f}")
    print(f"  - Reality Definition: {res_b['data']}")

    print("\n[Analysis]")
    print("1. The raw data is identical, but the perceived 'Reality' differs based on the internal lens.")
    print("2. 'Subject A' might see vitality, while 'Subject B' sees a warning or sin.")
    print("3. This proves that the 'Lens' (Past Judgment) determines the 'Subjective Truth' of the world.")
    print("==========================================================")

if __name__ == "__main__":
    simulate_subjective_refraction()
