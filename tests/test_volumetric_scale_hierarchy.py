"""
test_volumetric_scale_hierarchy.py
==================================
Unit tests verifying the Ontological Volumetric Scale Hierarchy Engine,
Boundary Surface Tension Field, and Causal Flow Deflection Dynamics.
"""

import pytest
import numpy as np
from core.physics.volumetric_scale_hierarchy import (
    StructuralDensityNode,
    BoundarySurfaceTensionField,
    VolumetricScaleHierarchyEngine,
)


def test_structural_density_node_creation_and_surface_tension():
    # Dense rigid rock node
    rock_node = StructuralDensityNode(
        node_id="rock_01",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        density=2.5,
        porosity=0.05,
        viscosity_resistance=0.9,
    )

    # Fluid water node
    water_node = StructuralDensityNode(
        node_id="water_01",
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        density=1.0,
        porosity=0.8,
        viscosity_resistance=0.2,
    )

    # Inherent surface tension should be much higher for dense rigid rock than fluid water
    assert rock_node.surface_tension > water_node.surface_tension
    assert rock_node.calculate_inherent_surface_tension() == pytest.approx(2.5 * 0.95 * 0.9)


def test_boundary_surface_tension_field_gradient():
    field = BoundarySurfaceTensionField()

    rock_node = StructuralDensityNode(
        node_id="rock_01", position=np.array([0.0, 0.0, 0.0]), density=2.5, porosity=0.05
    )
    water_node = StructuralDensityNode(
        node_id="water_01", position=np.array([2.0, 0.0, 0.0]), density=1.0, porosity=0.8
    )

    normal, tension_threshold = field.compute_boundary_gradient(rock_node, water_node)
    assert np.allclose(normal, [1.0, 0.0, 0.0])
    assert tension_threshold > 0.0


def test_causal_flow_deflection_around_dense_barrier():
    field = BoundarySurfaceTensionField()
    barrier = StructuralDensityNode(
        node_id="barrier",
        position=np.array([1.0, 0.0, 0.0]),
        density=2.0,
        porosity=0.1,
        viscosity_resistance=0.5,
    )

    flow_pos = np.array([0.0, 0.1, 0.0], dtype=np.float32)
    flow_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    new_vel, new_pos, deflection = field.deflect_flow(flow_vel, flow_pos, barrier, dt=0.01)

    # Velocity should be deflected vertically (tangential) away from direct approach to barrier
    assert deflection > 0.0
    assert new_vel[1] != 0.0  # Deflected along Y component
    assert len(barrier.interaction_history) == 1
    assert barrier.total_accumulated_energy > 0.0


def test_volumetric_scale_hierarchy_engine_simulation():
    engine = VolumetricScaleHierarchyEngine()

    rock = StructuralDensityNode(
        node_id="rock", position=np.array([1.0, 0.0, 0.0]), density=3.0, porosity=0.01
    )
    engine.add_node(rock)

    flow_pos = np.array([0.0, 0.05, 0.0], dtype=np.float32)
    flow_vel = np.array([2.0, 0.0, 0.0], dtype=np.float32)

    res = engine.simulate_flow_step(flow_vel, flow_pos, dt=0.01)

    assert "flow_position" in res
    assert "flow_velocity" in res
    assert res["total_deflection"] > 0.0
    assert res["energy_retention_ratio"] > 0.0
