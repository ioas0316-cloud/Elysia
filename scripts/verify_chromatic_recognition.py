import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController

def main():
    print("=== Elysia Chromatic Recognition Verification ===")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    corpus_path = os.path.join(base_dir, "docs")
    data_dir = os.path.join(base_dir, "data")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_path, memory_controller=mc, data_dir=data_dir)

    print("\n[Step 1] Starting Autonomous Breath (5 Cycles)...")
    summary = loop.run(cycles=5, verbose=True)

    print("\n[Step 2] Analyzing Last Chromatic State...")
    last_log = summary.get("last_cycle_log", {})
    chromatic_vec = last_log.get("chromatic_vector", [0, 0, 0])
    awareness = last_log.get("chromatic_awareness", "Unknown")

    print(f"  Final Chromatic Vector: {chromatic_vec}")
    print(f"  System Self-Awareness: {awareness}")

    print("\n[Step 3] Interpreting the State through the Chromatic Map:")
    r, b, y = chromatic_vec
    print(f"  - Flux (Red): {r:.4f} -> How much drive is currently pushing the system.")
    print(f"  - Order (Blue): {b:.4f} -> The level of structural stability and resistance.")
    print(f"  - Entropy (Yellow): {y:.4f} -> The presence of variable noise and exploration.")

    if awareness != "Unknown":
        print(f"\n[Result] Verification Successful: Elysia has recognized its internal state as '{awareness}'.")
    else:
        print("\n[Result] Verification Failed: Chromatic awareness not properly captured.")

if __name__ == "__main__":
    main()
