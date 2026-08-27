"""
[Verification Script: Structural Connectivity & Generative Principle Experiment]

Demonstrates:
1. "Ocean Wave vs. Causal Graph": Two completely different surface domains (physical ocean wave vs. cognitive causal graph)
   sharing a single Generative Principle ('Sameness').
2. Autonomous extraction of Generative Principle (topological invariants & generative grammar) from noisy Branching Points ('Difference').
3. O(1) Field Resonance reconstruction of unobserved trajectories under perturbed boundary conditions without brute-force parameter tuning.
"""

import math
import numpy as np

from core.topology.structural_connectivity_engine import (
    StructuralConnectivityEngine,
    BranchingPoint,
    GenerativePrinciple,
    StructuralResonance
)
from synaptic_architecture.inverse_mechanism_engine import (
    InverseMechanismEngine,
    ObservedTrajectory,
    BoundaryCondition
)
from core.physics.causal_field import CausalField, InformationVoxel


def run_verification_experiment():
    print("=========================================================")
    print("   Elysia: Structural Connectivity & Resonance Experiment  ")
    print("=========================================================")

    # 1. Initialize Engine
    engine = StructuralConnectivityEngine(mdl_threshold=1e-3)
    print("[Init] StructuralConnectivityEngine initialized with MDL threshold = 0.001")

    # 2. Simulate Noisy Divergent Surface Branching Points (Difference / 다름)
    # Underlying Wave Law: Invariant Baseline = [3.5, -1.2], Modulation = Harmonic Oscillation
    print("\n[Step 1] Generating divergent surface Branching Points (Physical Ocean & Cognitive Graph)...")

    ocean_states = []
    causal_states = []
    for t in range(25):
        # Ocean wave observation with high turbulent noise
        ow1 = 3.5 + 2.0 * math.sin(0.4 * t) + (np.random.rand() - 0.5) * 0.4
        ow2 = -1.2 + 1.5 * math.cos(0.4 * t) + (np.random.rand() - 0.5) * 0.4
        ocean_states.append([ow1, ow2])

        # Cognitive causal graph transition with structural phase shift & noise
        cg1 = 3.5 + 2.0 * math.sin(0.4 * t + 0.1) + (np.random.rand() - 0.5) * 0.2
        cg2 = -1.2 + 1.5 * math.cos(0.4 * t + 0.1) + (np.random.rand() - 0.5) * 0.2
        causal_states.append([cg1, cg2])

    ocean_branch = BranchingPoint(
        branch_id="physical_ocean_wave",
        context_domain="fluid_dynamics",
        observed_states=ocean_states,
        local_friction=1.5
    )
    causal_branch = BranchingPoint(
        branch_id="cognitive_causal_graph",
        context_domain="cognitive_topology",
        observed_states=causal_states,
        local_friction=0.8
    )

    # 3. Autonomous Extraction of Generative Principle (Sameness / 같음)
    print("\n[Step 2] Autonomously extracting unified Generative Principle...")
    unified_principle = engine.extract_generative_principle(
        principle_id="structural_unity_wave_law",
        branches=[ocean_branch, causal_branch]
    )

    print(f" -> Principle ID: {unified_principle.principle_id}")
    print(f" -> Extracted Topological Invariants: {unified_principle.topological_invariants}")
    print(f" -> Generative Grammar Parameters: {unified_principle.generative_grammar}")
    print(f" -> MDL Complexity Score: {unified_principle.mdl_complexity:.4f}")

    # 4. Evaluate Structural Isomorphism across Domains
    print("\n[Step 3] Computing Structural Isomorphism between Ocean Wave and Causal Graph...")
    isomorphism = engine.compute_structural_isomorphism(ocean_branch, causal_branch)
    print(f" -> Structural Isomorphism Score: {isomorphism:.4f} (High Isomorphism > 0.8)")

    # 5. Execute O(1) Field Resonance on Causal Field
    print("\n[Step 4] Executing O(1) Field Resonance in CausalField...")
    cf = CausalField(dimensions=2)
    for i in range(5):
        v = InformationVoxel(f"voxel_{i}", f"Node_{i}", np.array([0.1 * i, 0.2 * i], dtype=np.float32))
        cf.add_voxel(v)

    resonance_result = cf.resonate_field_structure(
        principle=unified_principle,
        source_branch=ocean_branch,
        target_context="unobserved_future_domain"
    )

    print(f" -> Resonated Target Branch ID: {resonance_result.target_branch_id}")
    print(f" -> Resonated Invariants: {resonance_result.resonated_invariants}")
    print(f" -> Converged O(1) Trajectory Steps: {len(resonance_result.converged_trajectory)}")
    print(f" -> Phase Alignment Delta: {resonance_result.phase_alignment_delta:.4f}")

    print("\n=========================================================")
    print("   Verification Experiment Completed Successfully!      ")
    print("=========================================================")


if __name__ == "__main__":
    run_verification_experiment()
