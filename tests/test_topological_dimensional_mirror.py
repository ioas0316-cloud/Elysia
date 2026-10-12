"""
Unit and Integration Tests for Topological Dimensional Mirror & Living Schema Engine
"""

import math
import pytest
import numpy as np

from core.embodied.topological_dimensional_mirror import (
    BodySchemaState,
    DimensionalTopologicalEngine,
    DualEnvironmentMirrorLoop,
    EnvironmentCondition,
    LivingCognitiveSchemaEngine,
    LivingSchema,
    TopologicalDiscrepancy,
)


def test_dimensional_topological_engine_mappings():
    engine = DimensionalTopologicalEngine(num_nodes=16)

    # 1D signal to 2D texture
    signal_1d = np.sin(np.linspace(0, 4 * np.pi, 64))
    texture_2d = engine.map_1d_signal_to_2d_texture(signal_1d, grid_size=(16, 16))
    assert texture_2d.shape == (16, 16)
    assert not np.isnan(texture_2d).any()

    # 2D texture to 3D body schema
    base_schema = BodySchemaState(
        nodes_3d=np.zeros((16, 3)),
        joint_rigidity=np.ones(16),
        chromatic_signature=np.array([0.33, 0.33, 0.34]),
        energy_potential=1.0
    )
    schema_3d = engine.map_2d_texture_to_3d_body(texture_2d, base_schema)
    assert schema_3d.nodes_3d.shape == (16, 3)
    assert not np.array_equal(schema_3d.nodes_3d, base_schema.nodes_3d)

    # 3D body schema history to 4D causal trajectory
    schema_history = [base_schema, schema_3d]
    tensor_4d = engine.map_3d_body_to_4d_causal_trajectory(schema_history, time_steps=0.5)
    assert tensor_4d.shape == (2, 16, 4)
    assert tensor_4d[1, 0, 3] == 0.5  # time coordinate

    # 4D causal trajectory fold back to 1D wave
    wave_1d = engine.fold_4d_causal_trajectory_to_1d(tensor_4d)
    assert wave_1d.shape == (2,)
    assert not np.isnan(wave_1d).any()


def test_dual_environment_mirror_loop_discrepancy():
    mirror = DualEnvironmentMirrorLoop()
    base_schema = BodySchemaState(
        nodes_3d=np.random.randn(16, 3) * 0.5,
        joint_rigidity=np.ones(16) * 0.8,
        chromatic_signature=np.array([0.4, 0.4, 0.2]),
        energy_potential=1.0
    )

    env_void = EnvironmentCondition(name="Standard Void", gravity=0.0, viscosity=0.0, surface_friction=0.0)
    env_viscous = EnvironmentCondition(name="Viscous Fluid", gravity=9.81, viscosity=2.5, surface_friction=0.8)

    results = mirror.run_reflection_cycle(base_schema, env_void, env_viscous, steps=5)

    assert "Standard Void" in results
    assert "Viscous Fluid" in results

    proj_void, disc_void = results["Standard Void"]
    proj_viscous, disc_viscous = results["Viscous Fluid"]

    assert len(proj_void) == 5
    assert len(proj_viscous) == 5

    # Viscous fluid should produce significantly higher total discrepancy than standard void
    assert disc_viscous.total_discrepancy > disc_void.total_discrepancy
    assert disc_viscous.topological_friction > disc_void.topological_friction


def test_living_cognitive_schema_engine():
    schema_engine = LivingCognitiveSchemaEngine()
    initial_schema = BodySchemaState(
        nodes_3d=np.zeros((16, 3)),
        joint_rigidity=np.ones(16) * 0.5,
        chromatic_signature=np.array([0.3, 0.3, 0.4]),
        energy_potential=1.0
    )

    discrepancy = TopologicalDiscrepancy(
        positional_drift=0.4,
        phase_decoherence=0.2,
        topological_friction=0.35,
        boundary_deformation=0.15,
        chromatic_mismatch=0.1,
        total_discrepancy=0.36
    )

    living_schema = schema_engine.absorb_discrepancy_and_embodiment(
        env_name="Viscous Medium",
        discrepancy=discrepancy,
        initial_schema=initial_schema
    )

    assert living_schema.schema_id == "SCHEMA_VISCOUS_MEDIUM"
    assert living_schema.metacognitive_awareness > 0.1
    assert len(living_schema.embodied_habit_vector) == 16
    assert np.all(living_schema.embodied_habit_vector >= initial_schema.joint_rigidity)

    # Refine Body Schema
    refined_schema = schema_engine.refine_body_schema(initial_schema, living_schema)
    assert np.array_equal(refined_schema.joint_rigidity, living_schema.embodied_habit_vector)
    assert pytest.approx(np.sum(refined_schema.chromatic_signature)) == 1.0
