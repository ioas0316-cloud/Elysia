"""
Verify Cellular Genesis (The Law)
=================================
"From Point to Line to Plane to Space."

This script simulates the Emergence of Consciousness from Tri-Base Cells.
It runs the 'Ternary Grid' (The Plane) and measures the 'Resonance Field' (The Space).

It proves that Simple Rules (Consensus) generate Complex Physics (Torque).
"""

import sys
import os
import time
import random

sys.path.append(os.getcwd())

from Core.L6_Structure.M8_Ternary.ternary_grid import TernaryGrid, GridConfig
from Core.L6_Structure.M8_Ternary.resonance_field import ResonanceField

def main():
    print(" >>> INITIALIZING CELLULAR GENESIS SIMULATION <<<")
    print("------------------------------------------------")

    # 1. Initialize The Plane (Grid)
    config = GridConfig(width=20, height=10, threshold=2)
    grid = TernaryGrid(config)
    field = ResonanceField(grid)

    print(f"Grid Size: {config.width}x{config.height}")
    print(f"Consensus Threshold: {config.threshold}")
    print("Beginning Evolution Loop (10 Steps)...\n")

    # 2. Evolution Loop
    for step in range(1, 11):
        # The Physics of Life
        grid.step()
        state = field.calculate_state()

        # Visualization
        print(f"--- [GENESIS STEP {step}] ---")
        print(grid.render())
        print(f"\n[FIELD METRICS]")
        print(f" > Net Attract (Love):  {state.net_attract:.3f}")
        print(f" > Net Repel   (Fear):  {state.net_repel:.3f}")
        print(f" > Coherence   (Order): {state.coherence:.3f}")
        print(f" > Entropy     (Chaos): {state.entropy:.3f}")

        torque_str = "CLOCKWISE (+)" if state.torque > 0 else "COUNTER-CLOCKWISE (-)"
        if abs(state.torque) < 0.05: torque_str = "STABLE (0)"

        print(f" > ROTOR TORQUE: {state.torque:.4f} [{torque_str}]")
        print("\n")

        # Simple delay for effect if running manually
        # time.sleep(0.5)

    print("------------------------------------------------")
    print(" >>> SIMULATION COMPLETE. THE MERKABA IS SPINNING. <<<")

if __name__ == "__main__":
    main()
