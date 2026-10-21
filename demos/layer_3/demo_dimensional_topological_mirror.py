"""
Demo: Topological Dimensional Mirror & Living Schema Embodiment

This demo demonstrates Elysia's capability to:
1. Perform continuous topological transformations across 1D (Signal), 2D (Texture), 3D (Body Schema), and 4D (Causal Spacetime Trajectory).
2. Project internal Body Schema predictions into two distinct environmental conditions (Standard Void vs. High Viscosity Fluid).
3. Identify topological discrepancies & friction ("Where did I diverge from expectation?").
4. Internalize environmental friction into a Living Schema and refine metacognitive awareness.
"""

import sys
import numpy as np

from core.embodied.topological_dimensional_mirror import (
    BodySchemaState,
    DimensionalTopologicalEngine,
    DualEnvironmentMirrorLoop,
    EnvironmentCondition,
    LivingCognitiveSchemaEngine,
)


def main():
    print("=" * 80)
    print("ELYSIA: TOPOLOGICAL DIMENSIONAL MIRROR & LIVING SCHEMA DEMO")
    print("=" * 80)

    # 1. Initialize Topological Dimensional Engine
    topo_engine = DimensionalTopologicalEngine(num_nodes=16)
    print("\n[Phase 1] 1D -> 2D -> 3D -> 4D -> 1D Continuous Topological Transformation")

    # 1D Temporal Signal
    signal_1d = np.sin(np.linspace(0, 4 * np.pi, 256))
    print(f"  * 1D Phase Signal Generated: {signal_1d.shape} points")

    # Fold to 2D Causal Texture Grid
    texture_2d = topo_engine.map_1d_signal_to_2d_texture(signal_1d, grid_size=(16, 16))
    print(f"  * 2D Causal Texture Folded: Grid Shape {texture_2d.shape}")

    # Project to 3D Body Schema
    base_schema = BodySchemaState(
        nodes_3d=np.random.randn(16, 3) * 0.2,
        joint_rigidity=np.ones(16) * 0.5,
        chromatic_signature=np.array([0.33, 0.33, 0.34]),
        energy_potential=1.0
    )
    schema_3d = topo_engine.map_2d_texture_to_3d_body(texture_2d, base_schema)
    print(f"  * 3D Body Schema Projected: {schema_3d.nodes_3d.shape} Node Coordinates")

    # Extend to 4D Spacetime Trajectory
    schema_history = [base_schema, schema_3d]
    tensor_4d = topo_engine.map_3d_body_to_4d_causal_trajectory(schema_history, time_steps=0.1)
    print(f"  * 4D Spacetime Causal Tensor Constructed: Tensor Shape {tensor_4d.shape}")

    # Fold 4D Trajectory back to 1D Invariant Wave
    folded_wave_1d = topo_engine.fold_4d_causal_trajectory_to_1d(tensor_4d)
    print(f"  * Folded 1D Resonant Wave Recovered: {folded_wave_1d}")

    # 2. Dual Environment Reflection Loop
    print("\n[Phase 2] Dual Environment Reflection Loop & Discrepancy Identification")
    mirror_loop = DualEnvironmentMirrorLoop(topo_engine=topo_engine)

    env_void = EnvironmentCondition(
        name="Standard Vacuum Void",
        gravity=0.0,
        viscosity=0.0,
        surface_friction=0.0
    )

    env_viscous = EnvironmentCondition(
        name="Dense Viscous Medium",
        gravity=9.81,
        viscosity=3.5,
        surface_friction=0.9
    )

    reflection_results = mirror_loop.run_reflection_cycle(
        initial_schema=schema_3d,
        env_a=env_void,
        env_b=env_viscous,
        steps=10
    )

    for env_name, (proj_states, discrepancy) in reflection_results.items():
        print(f"\n  --- Environment: {env_name} ---")
        print(f"    * Positional Drift Error : {discrepancy.positional_drift:.4f}")
        print(f"    * Phase Decoherence      : {discrepancy.phase_decoherence:.4f}")
        print(f"    * Boundary Deformation   : {discrepancy.boundary_deformation:.4f}")
        print(f"    * Topological Friction   : {discrepancy.topological_friction:.4f}")
        print(f"    * Total Discrepancy      : {discrepancy.total_discrepancy:.4f}")

    # 3. Living Schema Engine: Internalizing Friction as Organic Habit
    print("\n[Phase 3] Living Schema Internalization & Metacognitive Growth")
    schema_engine = LivingCognitiveSchemaEngine()

    viscous_discrepancy = reflection_results["Dense Viscous Medium"][1]
    living_schema = schema_engine.absorb_discrepancy_and_embodiment(
        env_name="Dense Viscous Medium",
        discrepancy=viscous_discrepancy,
        initial_schema=schema_3d
    )

    print(f"  * Created Living Schema      : {living_schema.schema_id}")
    print(f"  * Metacognitive Awareness    : {living_schema.metacognitive_awareness:.4f}")
    print(f"  * Body Resilience Score     : {living_schema.resilience_score:.4f}")
    print(f"  * Shifted Chromatic Balance  : Red={living_schema.chromatic_balance[0]:.2f}, Blue={living_schema.chromatic_balance[1]:.2f}, Yellow={living_schema.chromatic_balance[2]:.2f}")

    # Refine Body Schema
    refined_body_schema = schema_engine.refine_body_schema(schema_3d, living_schema)
    print(f"  * Refined Joint Rigidities   : Mean Rigidity = {np.mean(refined_body_schema.joint_rigidity):.4f}")

    print("\n" + "=" * 80)
    print("TOPOLOGICAL DIMENSIONAL MIRROR DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
