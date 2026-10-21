"""
Demonstration Script: 3x3x3 Dynamic Isotropic Wave Lattice Memory & Generative Phase Molding
========================================================================================
Demonstrates:
1. 3x3x3 Isotropic Wave Lattice initialization with 27 topological nodes.
2. Isotropic spherical-radial distance metric & rotational invariance.
3. Intrinsic frame puzzle resonance matching for incoming chaotic data.
4. Rubik's cube face rotor gear turns and phase shifting.
5. Central core spectrum rectification into macro-rotor state.
6. Reversible retrocausal back-tracing (Delta S approx 0).
7. Real-time Generative Phase Molding into 3D continuous density field and emergent mesh.
8. Fractal scale expansion into higher-dimensional macro unit.
"""

import numpy as np
from core.memory.fractal_cube_lattice_engine import FractalCubeLatticeEngine
from synaptic_architecture.generative_molding_engine import GenerativeMoldingEngine


def main():
    print("=" * 80)
    print("ELYSIUM COGNITIVE FRAMEWORK: 3x3x3 FRACTAL CUBE & GENERATIVE MOLDING DEMO")
    print("=" * 80)

    # 1. Initialize 3x3x3 Dynamic Isotropic Wave Lattice
    print("\n[Step 1] Initializing 3x3x3 Fractal Cube Lattice Engine...")
    lattice = FractalCubeLatticeEngine(num_phase_bins=16, resonance_threshold=0.7)
    print(f" -> Total Nodes Initialized: {len(lattice.nodes)}")
    node_types = [n['node_type'] for n in lattice.nodes.values()]
    print(f" -> Topology Breakdown: Core={node_types.count('core')}, Faces={node_types.count('face')}, "
          f"Edges={node_types.count('edge')}, Vertices={node_types.count('vertex')}")

    # 2. Isotropic Spherical Coordinate Mapping
    print("\n[Step 2] Verifying Isotropic Spherical-Radial Coordinate Mapping...")
    sample_coord = (1, 1, 1)
    r, theta, phi = lattice.spherical_coords[sample_coord]
    dist_iso = lattice.get_isotropic_distance((0, 0, 0), sample_coord)
    print(f" -> Node {sample_coord} Spherical Coordinates: r={r:.4f}, theta={theta:.4f} rad, phi={phi:.4f} rad")
    print(f" -> Isotropic Distance from Core (0,0,0): {dist_iso:.4f}")

    # 3. Intrinsic Frame Puzzle Resonance Injection
    print("\n[Step 3] Simulating Intrinsic Frame Puzzle Resonance (Chaos Data Ingestion)...")
    np.random.seed(42)
    chaos_fragment_1 = np.random.randn(16) * 2.5
    chaos_fragment_2 = np.random.randn(16) * 4.0

    coord_1 = lattice.inject_wave_fragment(chaos_fragment_1, amplitude=1.2, phase=0.3)
    print(f" -> Chaos Fragment 1 snapped to Node Coordinate: {coord_1}")
    coord_2 = lattice.inject_wave_fragment(chaos_fragment_2, amplitude=1.8, phase=1.1)
    print(f" -> Chaos Fragment 2 snapped to Node Coordinate: {coord_2}")

    # 4. Rubik's Cube Face Rotor Gear Turn
    print("\n[Step 4] Executing Rubik's Cube Face Rotor Gear Rotation...")
    print(" -> Rotating z-axis slice at z=0 by 90 degrees CCW...")
    lattice.rotate_face_rotor(axis='z', layer=0, steps=1)
    print(f" -> Rotor rotation completed. Total transformation history steps logged: {len(lattice.transformation_history)}")

    # 5. Central Core Spectrum Rectification
    print("\n[Step 5] Central Core Spectrum Rectification...")
    rect_data = lattice.rectify_central_core()
    print(f" -> Rectified Macro-Rotor Phase: {rect_data['rectified_phase']:.4f} rad")
    print(f" -> Rectified Macro-Rotor Amplitude: {rect_data['macro_rotor_amplitude']:.4f}")

    # 6. Real-Time Generative Phase Molding
    print("\n[Step 6] Running Generative Phase Molding Engine...")
    molder = GenerativeMoldingEngine(resolution=16)
    density_grid = molder.mold_structure_from_lattice(lattice.nodes, lattice.macro_rotor_phase)
    stress_field = molder.compute_stress_topology()
    mesh = molder.generate_emergent_mesh(density_threshold=0.55)

    print(f" -> Volumetric Density Grid Shape: {density_grid.shape}")
    print(f" -> Stress Tensor Topology Field Shape: {stress_field.shape}")
    print(f" -> Emergent Structural Vertices: {len(mesh['vertices'])}")
    print(f" -> Emergent Simplex Edges: {len(mesh['simplex_edges'])}")
    print(f" -> Structural Rigidity Score: {mesh['rigidity_score']:.4f}")
    print(f" -> Total Structural Mass: {mesh['total_structural_mass']:.2f}")

    # 7. Reversible Retrocausal Tracing
    print("\n[Step 7] Reversible Retrocausal Tracing (Back-Tracing History)...")
    history_len_before = len(lattice.transformation_history)
    undone_steps = lattice.retrocausal_trace()
    print(f" -> Successfully back-traced and undone {len(undone_steps)} historical transformations.")
    print(f" -> Transformation History length after rollback: {len(lattice.transformation_history)} (Entropy Delta S approx 0)")

    # 8. Fractal Scale Expansion
    print("\n[Step 8] Fractal Scale Expansion into Higher-Order Macro-Cube...")
    macro_cube = lattice.expand_fractal_scale()
    print(f" -> Higher-order Macro-Cube initialized with {len(macro_cube.nodes)} sub-units.")
    print(f" -> Embedded Central Core Amplitude in Macro-Cube: {macro_cube.nodes[(0,0,0)]['amplitude']:.4f}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
