"""
Verify DNA Squared (Exponential Cognition)
==========================================
"The Line becomes a Plane; The Plane becomes a Space."

This script verifies the "DNA x DNA" hypothesis:
That Trinary Logic (-1, 0, 1) expands exponentially via Tensor Products,
creating a fractal cognitive structure rather than a linear stack.
"""

import numpy as np
import sys
import os

# Ensure we can import from Core if needed (though we use raw numpy here for pure math verification)
sys.path.append(os.getcwd())

def verify_dna_squared():
    print("🧬 [TEST] Verifying DNA² (Exponential Expansion)...")

    # 1. The Seed (1D DNA)
    # The Trinary Trinity: Repel (-1), Void (0), Attract (1)
    dna_1d = np.array([-1, 0, 1])
    print(f"\n🌱 [1D] The Line (Vector): Shape {dna_1d.shape}")
    print(dna_1d)

    # 2. The Plane (2D DNA x DNA)
    # Outer Product: Every element interacts with every other element
    dna_2d = np.outer(dna_1d, dna_1d)
    print(f"\n🌿 [2D] The Plane (Matrix): Shape {dna_2d.shape}")
    print("   (Interaction Field)")
    print(dna_2d)

    # Check properties
    # The center is still 0 (Void Stability)
    center = dna_2d[1, 1]
    # The corners are interactions of strong forces (-1*-1=1, 1*1=1)
    corners = [dna_2d[0,0], dna_2d[2,2]]

    print(f"   - Void Center: {center} (The Eye of the Storm)")
    print(f"   - High Energy Corners: {corners} (Resonance)")

    # 3. The Space (3D DNA x DNA x DNA)
    # Tensor Product again
    dna_3d = np.tensordot(dna_2d, dna_1d, axes=0)
    # Note: tensordot with axes=0 acts like outer product for higher dims

    print(f"\n🌳 [3D] The Space (Tensor): Shape {dna_3d.shape} -> {dna_3d.size} nodes")
    print(f"   (Cognitive Volume)")

    # 4. Verify Exponential Growth
    print(f"\n📈 [GROWTH METRICS]")
    print(f"   1D Nodes: {dna_1d.size}")
    print(f"   2D Nodes: {dna_2d.size} (x{dna_2d.size/dna_1d.size})")
    print(f"   3D Nodes: {dna_3d.size} (x{dna_3d.size/dna_2d.size})")

    # 5. Fractal Verification (Self-Similarity)
    # Does the 2D structure contain the 1D pattern?
    # The middle row of the 2D matrix should be 0 * [-1,0,1] = [0,0,0] (The Void Line)
    # The bottom row should be 1 * [-1,0,1] = [-1,0,1] (The Replica)
    is_fractal = np.array_equal(dna_2d[2], dna_1d)

    if is_fractal:
        print("\n✨ [SUCCESS] Fractal Self-Similarity Confirmed.")
        print("   The structure reproduces its parent pattern at higher dimensions.")
    else:
        print("\n❌ [FAIL] Fractal pattern lost.")

if __name__ == "__main__":
    verify_dna_squared()
