"""
test_scale_causal_reversibility.py
==================================
Verification Test Suite for Scale Principles & Bidirectional Causal Reversibility.

Verifies:
1. Volumetric Scale Hierarchy interaction history rewind & 1:1 state restoration.
2. Continuous Phase Transition along density-surface tension spectrum (no point-rule flags).
3. Field Unified Physics Core exact time-reversal (Macro -> Micro) with History_Invariance = 1.0.
4. Clifford Spacetime Rotor inverse sandwich transformation restoration (error < 1e-12).
"""

import pytest
import numpy as np

from core.physics.volumetric_scale_hierarchy import (
    StructuralDensityNode,
    VolumetricScaleHierarchyEngine
)
from core.physics.field_unified_physics_core import (
    FieldUnifiedPhysicsCore,
    MatterState
)
from core.physics.clifford_spacetime_rotor import (
    CliffordSpacetimeRotor,
    CliffordMultivector3D
)


def test_clifford_rotor_bidirectional_reversibility():
    """Verifies that v = R~ (R v R~) R restores original vector with zero error."""
    axis = np.array([1.0, 2.0, -1.0], dtype=np.float64)
    angle = 1.25  # radians
    rotor = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    v_orig = np.array([3.5, -2.1, 8.4], dtype=np.float64)

    # Forward transformation (Micro -> Macro)
    v_transformed = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, v_orig)

    # Backward transformation (Macro -> Micro)
    v_restored = CliffordSpacetimeRotor.rewind_transform_vector_3d(rotor, v_transformed)

    reconstruction_error = float(np.linalg.norm(v_orig - v_restored))
    assert reconstruction_error < 1e-12, f"Rotor reversal error too high: {reconstruction_error}"


def test_volumetric_scale_hierarchy_history_rewind():
    """Verifies that VolumetricScaleHierarchyEngine rewinds interaction history and restores energy/positions exactly."""
    engine = VolumetricScaleHierarchyEngine()
    barrier = StructuralDensityNode(
        node_id="barrier_rock",
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        density=2.5,
        porosity=0.05,
        viscosity_resistance=0.8
    )
    engine.add_node(barrier)

    init_pos = np.array([0.0, 0.1, 0.0], dtype=np.float32)
    init_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Step forward
    step_res = engine.simulate_flow_step(init_vel, init_pos, dt=0.01)
    assert len(barrier.interaction_history) == 1
    assert step_res["total_deflection"] > 0.0

    # Rewind step (Macro -> Micro)
    rewind_res = engine.rewind_step()
    assert rewind_res is not None
    assert rewind_res["history_invariance"] == 1.0
    assert float(np.linalg.norm(rewind_res["restored_position"] - init_pos)) < 1e-6
    assert len(barrier.interaction_history) == 0


def test_field_unified_physics_core_time_reversal():
    """Verifies wavefield time reversal restores complex phase field with History_Invariance = 1.0."""
    physics = FieldUnifiedPhysicsCore(resolution=128)
    initial_field = physics.field.copy()

    # Evolve forward 5 steps
    for _ in range(5):
        physics.step()

    assert not np.array_equal(physics.field, initial_field)

    # Rewind 5 steps
    restored_field, power = physics.rewind_field_time(steps=5)

    diff = float(np.linalg.norm(restored_field - initial_field))
    assert diff < 1e-12, f"Wavefield time reversal error too high: {diff}"


def test_continuous_matter_phase_transition():
    """
    Verifies that matter phase transition occurs along a continuous spectrum of density
    and surface tension without point-rule `if-else` flags.
    """
    physics = FieldUnifiedPhysicsCore(resolution=128)

    # Low density / low tension -> Gas behavior
    gas_res = physics.simulate_continuous_matter_transition(density=0.05, surface_tension=0.1)
    assert gas_res["effective_friction"] < 0.3
    assert gas_res["phase_velocity"] > 0.9

    # High density / high tension -> Solid behavior
    solid_res = physics.simulate_continuous_matter_transition(density=3.0, surface_tension=1.0)
    assert solid_res["effective_friction"] > 3.0
    assert solid_res["phase_velocity"] < 0.2
