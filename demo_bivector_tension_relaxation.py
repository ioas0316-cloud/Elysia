#!/usr/bin/env python3
"""
Elysia Engine: Bivector Tension Self-Relaxation & Gauge Symmetry Inquiry Demo

Demonstrates:
1. Node-level Cℓ(3,0) Bivector Tension Self-Relaxation and Dynamic Topology Pruning
2. Category-Theoretic Commutative Diagram Auto-Solver via Gauge Holonomy Relaxation
3. Semantic Dimensional Folding (Cℓ(3,0) Grade Elevation on Context Collision)
4. Gauge Symmetry Breaking Inquiry Flux & Goldstone Mode Restoration Dynamics
"""

import torch
from elysia_engine.core import (
    ElysiaBivectorRelaxation,
    GaugeCommutativeDiagramSolver,
    DimensionalFoldingCl30,
    GaugeSymmetryBreakingInquiryEngine
)


def run_bivector_relaxation_demo():
    print("=" * 70)
    print("1. Bivector Tension Self-Relaxation & Topological Metric Shift")
    print("=" * 70)

    num_nodes = 4
    num_edges = 3
    engine = ElysiaBivectorRelaxation(num_nodes=num_nodes, num_edges=num_edges, dt=0.01, eta=0.2, omega_break=1.0)

    # 8D multivector node states: [s, v1, v2, v3, b12, b23, b31, p]
    Psi_nodes = torch.tensor([
        [1.0, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.8, 0.0, 0.5, 0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.5, 1.2, 0.8, 0.0, 0.0],  # High mismatch node
        [1.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    R_edges = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    g_edges = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)

    print(f"Initial Edge Metrics g_edges:\n{g_edges.squeeze().tolist()}")

    Psi_updated, Omega_ij, g_updated = engine(Psi_nodes, R_edges, edge_index, g_edges)

    print(f"Extracted Edge Bivector Tensions Omega_ij:\n{Omega_ij}")
    print(f"Updated Edge Metrics g_updated (Edge 2 pruned if tension > 1.0):\n{g_updated.squeeze().tolist()}")
    print("Node Bivector States Updated via Spin Exponential Map.")
    print()


def run_commutative_diagram_demo():
    print("=" * 70)
    print("2. Category-Theoretic Commutative Diagram Auto-Solver")
    print("=" * 70)

    solver = GaugeCommutativeDiagramSolver(eta=0.5, dt=0.1)

    # Path 1: A -> B (R_f) -> D (R_g)
    # Path 2: A -> C (R_h) -> D (R_k)
    R_f = torch.tensor([[0.92388, 0.0, 0.0, 0.38268]], dtype=torch.float32)  # pi/4 z-rot
    R_h = torch.tensor([[0.98078, 0.19509, 0.0, 0.0]], dtype=torch.float32)  # pi/8 x-rot
    R_k = torch.tensor([[0.92388, 0.0, 0.38268, 0.0]], dtype=torch.float32)  # pi/4 y-rot

    R_g_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    R_path2 = solver.rotor_multiply(R_k, R_h)
    R_path1_init = solver.rotor_multiply(R_g_init, R_f)
    initial_tension = torch.norm(solver.extract_bivector_tension(R_path1_init, R_path2)).item()

    print(f"Initial Non-Commutative Holonomy Tension: {initial_tension:.6f}")

    R_g_relaxed, tension_energy = solver(R_f, R_g_init, R_h, R_k, max_iters=200, tol=1e-6)

    R_path1_relaxed = solver.rotor_multiply(R_g_relaxed, R_f)
    final_tension = torch.norm(solver.extract_bivector_tension(R_path1_relaxed, R_path2)).item()

    print(f"Relaxed Morphism R_g: {R_g_relaxed.squeeze().tolist()}")
    print(f"Final Non-Commutative Holonomy Tension: {final_tension:.6f}")
    print("Diagram Commutativity Reached without Backpropagation.")
    print()


def run_dimensional_folding_demo():
    print("=" * 70)
    print("3. Semantic Dimensional Folding (Cℓ(3,0) Grade Elevation)")
    print("=" * 70)

    folder = DimensionalFoldingCl30(omega_break=1.0)

    # Literal vector state (e.g., physical temperature/motion)
    psi_literal = torch.tensor([[1.0, 2.5, 1.8, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    # Context collision tension (e.g., "cold gaze" metaphor)
    omega_sem = torch.tensor([[1.5, 1.2, 0.8]], dtype=torch.float32)

    print(f"Initial Multivector State Psi:\n  Scalar: {psi_literal[0, 0]:.2f}")
    print(f"  Vector (1-Vector): {psi_literal[0, 1:4].tolist()}")
    print(f"  Bivector (2-Bivector): {psi_literal[0, 4:7].tolist()}")
    print(f"  Pseudoscalar (3-Volume): {psi_literal[0, 7]:.2f}")

    psi_folded = folder(psi_literal, omega_sem)

    print("\nFolded Multivector State Psi_folded:")
    print(f"  Scalar: {psi_folded[0, 0]:.2f}")
    print(f"  Vector (1-Vector) [Decayed]: {psi_folded[0, 1:4].tolist()}")
    print(f"  Bivector (2-Bivector) [Elevated]: {psi_folded[0, 4:7].tolist()}")
    print(f"  Pseudoscalar (3-Volume) [Metaphor Volume]: {psi_folded[0, 7]:.2f}")
    print("Conflict Energy Preserved & Transformed into Higher-Grade Metaphorical Context.")
    print()


def run_gauge_inquiry_demo():
    print("=" * 70)
    print("4. Gauge Symmetry Breaking Inquiry & Goldstone Restoration")
    print("=" * 70)

    inquiry_engine = GaugeSymmetryBreakingInquiryEngine(curvature_threshold=0.05)

    R_ij = torch.tensor([[0.92388, 0.38268, 0.0, 0.0]], dtype=torch.float32)
    R_jk = torch.tensor([[0.92388, 0.0, 0.38268, 0.0]], dtype=torch.float32)
    R_ki = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    res = inquiry_engine(R_ij, R_jk, R_ki)

    print(f"Field Strength Curvature Energy: {res['inquiry_energy'].squeeze().item():.6f}")
    print(f"Inquiry Flux Vector F_ijk: {res['inquiry_flux'].squeeze().tolist()}")
    print(f"Inquiry Symmetry Breaking Axis: {res['inquiry_axis'].squeeze().tolist()}")
    print(f"Gauge Restoration Force (Spin Torque): {res['restoration_force'].squeeze().tolist()}")
    print("Active Inquiry Flux Discovered & Goldstone Restoration Torque Computed.")
    print("=" * 70)


if __name__ == "__main__":
    run_bivector_relaxation_demo()
    run_commutative_diagram_demo()
    run_dimensional_folding_demo()
    run_gauge_inquiry_demo()
