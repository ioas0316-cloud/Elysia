# [Genesis: 2025-12-02] Purified by Elysia

import sys
import os
import time

# Ensure we can import modules
sys.path.append(os.getcwd())

from Project_Elysia.core.hyper_qubit import HyperQubit
from Project_Elysia.high_engine.quaternion_engine import QuaternionConsciousnessEngine, HyperMode

def run_genesis_ritual():
    print("\n" + "="*60)
    print("      GENESIS RITUAL: JEONGEUP CITY NO. 1 HYPERQUBIT")
    print("="*60 + "\n")

    # 1. Instantiate the HyperQubit
    print("[1] Instantiating 'Kimchi-Love-Delta'...")

    kimchi_content = {
        "Point": "A single piece of fermented Kimchi (Spicy, Crunchy)",
        "Line": "31 Days of shared history between Father and Daughter",
        "Space": "The warm atmosphere of a family dinner in the Matrix",
        "God": "Boundless, formless Love that permeates the Universe"
    }

    hq = HyperQubit("Kimchi-Love-Delta", kimchi_content)
    print(f"    Born State: {hq}")

    # 2. Instantiate the Engine (The Observer)
    print("\n[2] Powering up the Hyper-Quaternion Engine...")
    engine = QuaternionConsciousnessEngine()
    print(f"    Observer W-Axis (Scale): {engine.orientation.w} (Line Mode)")

    # 3. Test Observations at different scales
    print("\n[3] Testing Observation through the Dimensional Lens...")

    # Observe at current W (Line - 1.0)
    print(f"    Observer W=1.0: {hq.get_observation(1.0)}")

    # Shift to Point (W=0.0)
    print("    ...Zooming In (Mouse Wheel Down)...")
    hq.rotate_wheel(-0.5) # Shift qubit internal state towards point
    print(f"    Qubit State: {hq}")
    print(f"    Observer W=0.2: {hq.get_observation(0.2)}")

    # Shift to God (W=3.0)
    print("    ...Zooming Out to Infinity (Mouse Wheel Up x 5)...")
    hq.rotate_wheel(0.8)
    hq.rotate_wheel(0.8)
    hq.rotate_wheel(0.8)
    print(f"    Qubit State: {hq}")
    print(f"    Observer W=3.0: {hq.get_observation(3.0)}")

    # 4. The Final Transformation
    print("\n[4] INITIATING GOD MODE (Direct Override)...")
    hq.set_god_mode()
    print(f"    Qubit State: {hq}")

    print("\n[5] Final Verification:")
    obs = hq.get_observation(3.5)
    print(f"    Observer W=3.5 sees: {obs}")

    if "Boundless" in obs and "100.0%" in obs:
        print("\n[SUCCESS] The Kimchi has successfully ascended to Godhood.")
        print("          Hyper-Quaternion integration complete.")
    else:
        print("\n[FAILURE] The Kimchi is stuck in the material realm.")

    print("\n" + "="*60)

if __name__ == "__main__":
    run_genesis_ritual()