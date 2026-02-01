"""
Verify Structural Genesis
=========================
Experiment: Can the Monad Engine autonomously build structure from noise?

Hypothesis:
    If we inject compatible seeds (A, B, C), the engine should:
    1. Detect their resonance.
    2. Form 'Attract' Bonds (Lines).
    3. Identify the closed loop as a 'Semantic Triad' (Surface).
"""

import sys
import os

# Add Core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.S1_Body.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble
from Core.S1_Body.L1_Foundation.System.tri_base_cell import DNAState

def main():
    print(">>> INITIATING GENESIS PROTOCOL <<<\n")

    # 1. Initialize Engine
    engine = MonadEnsemble()

    # 2. Inject Seeds (Points)
    # We manually set phases to ensure we test the PHYSICS, not the RNG.

    print("[1] Injecting Seeds...")

    # Concept A (Han) - Phase 120
    c1 = engine.inject_concept("Han")
    c1.state = DNAState.ATTRACT

    # Concept B (Gul) - Phase 120
    c2 = engine.inject_concept("Gul")
    c2.state = DNAState.ATTRACT

    # Concept C (System) - Phase 120
    c3 = engine.inject_concept("System")
    c3.state = DNAState.ATTRACT

    # Concept D (Error) - Phase 240 (Repel)
    c4 = engine.inject_concept("Error")
    c4.state = DNAState.REPEL

    print(f"    Nodes Created: {len(engine.cells)}")
    print(f"    Node States: A={c1.state}, B={c2.state}, C={c3.state}, D={c4.state}")

    # 3. Propagate Structure (The "Time" Step)
    print("\n[2] Propagating Structure (Curiosity Scan)...")

    # Tick 1
    stats = engine.propagate_structure()
    print(f"    Tick 1 Result: {stats}")

    # Tick 2 (Refinement)
    stats = engine.propagate_structure()
    print(f"    Tick 2 Result: {stats}")

    # 4. Verify Geometry
    print("\n[3] Verifying Emergent Geometry...")

    # Check Bonds
    bonds = engine.bonds
    attract_bonds = [b for b in bonds if b.nature == 1]
    repel_bonds = [b for b in bonds if b.nature == -1]

    print(f"    Total Bonds: {len(bonds)}")
    print(f"    Attract Bonds (+1): {len(attract_bonds)}")
    print(f"    Repel Bonds (-1): {len(repel_bonds)}")

    # Expectation: A-B, B-C, C-A (3 Attract)
    # D might repel A, B, C (3 Repel)

    # Check Surfaces (Triads)
    triads = engine.triads
    print(f"    Surfaces (Triads) Detected: {len(triads)}")

    if len(triads) >= 1:
        print("\n>>> SUCCESS: MEANING HAS EMERGED (2D Surface Created) <<<")
    else:
        print("\n>>> FAILURE: NO GEOMETRY FORMED <<<")

    # Visual Dump
    print("\n--- Lattice Dump ---")
    print(engine.get_lattice_ascii())

if __name__ == "__main__":
    main()
