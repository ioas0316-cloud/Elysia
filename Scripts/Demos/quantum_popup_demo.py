"""
Quantum Pop-Up Protocol Demo
============================
Core.Demos.quantum_popup_demo

Demonstrates the 2D to 3D architectural collapse and Akashic persistence.
"""

import sys
import os
import time

# Ensure root is in path
sys.path.append(os.getcwd())

from Core.L4_Causality.World.Architecture.blueprint_analyzer import BlueprintGenerator
from Core.L4_Causality.World.Architecture.quantum_architect import QuantumArchitect
from Core.L4_Causality.World.Architecture.ascii_slicer import AsciiSlicer
from Core.L4_Causality.World.Architecture.spatial_memory import SpatialMemory

def main():
    print("\n🚀 [Quantum Pop-Up Protocol] Initializing...")

    # 1. Input: Generate Blueprint
    print("👁️  [Input] Scanning Blueprint...")
    bg = BlueprintGenerator()
    blueprint = bg.generate_apartment(width=16, height=10)
    print("    -> Blueprint Acquired (16x10 Apartment)")

    # 2. Process: Quantum Collapse
    print("⚡ [Process] Collapsing Wave Function...")
    architect = QuantumArchitect()
    result = architect.collapse_space(blueprint, height=4)

    voxels = result["voxels"]
    address = result["address"]
    obj_data = result["obj"]

    print(f"    -> Collapse Complete. Generated {len(obj_data)} bytes of OBJ data.")
    print(f"    -> Stored in Hypersphere at Address: {address}")

    # 3. Validation: Recall from Memory
    print("🔮 [Validation] Recalling from Akashic Records...")
    memory = SpatialMemory()
    recalled_obj, meta = memory.recall(address)

    if recalled_obj == obj_data:
        print("    -> Memory Integrity Verified. (Data Match)")
        print(f"    -> Metadata: {meta}")
    else:
        print("    -> [CRITICAL] Memory Corruption Detected!")

    memory.close()

    # 4. Output: ASCII Visualization
    print("\n🖥️  [Output] Cyberpunk Slicing Viewer")
    slicer = AsciiSlicer()
    print(slicer.render_all(voxels))

    print("\n✅ System Ready. Waiting for next observation.")

if __name__ == "__main__":
    main()
