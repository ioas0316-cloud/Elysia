import numpy as np
import pytest
from L1_causal_field.core.physics.structural_density_gravity_engine import (
    StructuralDensityGravityEngine,
    PhaseStateSpectrum,
    TopologicalNode,
    RuptureEvent
)


def test_phase_spectrum_classification():
    rock = PhaseStateSpectrum(density=4.0, mobility=0.05, concentration=1.0)
    assert rock.classify_materialization() == "Rock"

    water = PhaseStateSpectrum(density=1.0, mobility=2.0, agitation=2.0)
    assert water.classify_materialization() == "Water"

    gas = PhaseStateSpectrum(density=0.1, mobility=4.0, entropy=3.0)
    assert gas.classify_materialization() == "Gas_Cloud"

    blade = PhaseStateSpectrum(density=3.0, concentration=8.0, tension_limit=0.2)
    assert blade.classify_materialization() == "Blade"


def test_structural_density_and_gravity_gradient():
    engine = StructuralDensityGravityEngine(dimensions=4)

    # Create a dense cluster around pos [10, 0, 0, 0]
    center_pos = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(5):
        pos = center_pos + np.array([i * 0.1, 0, 0, 0], dtype=np.float32)
        engine.add_node(f"dense_{i}", pos, phase=PhaseStateSpectrum(density=3.0, mobility=0.1))

    # Connect dense cluster
    for i in range(4):
        engine.connect_nodes(f"dense_{i}", f"dense_{i+1}", tension_strength=2.0)

    # Calculate structural density
    local_density = engine.compute_local_structural_density(center_pos, radius=2.0)
    assert local_density > 5.0

    # Test gravity gradient from query pos [0, 0, 0, 0]
    query_pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gradient, distortion, c_spec = engine.compute_gravity_field_gradient(query_pos)

    # Gradient should point strongly along positive x-axis towards [10, 0, 0, 0]
    assert gradient[0] > 0.0
    assert distortion > 0.0
    assert c_spec < engine.base_spectral_speed  # Speed compressed due to field distortion


def test_least_resistance_geodesic_flow():
    engine = StructuralDensityGravityEngine(dimensions=4)

    # Path A -> B -> C
    engine.add_node("A", np.array([0, 0, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(mobility=2.0))
    engine.add_node("B", np.array([1, 0, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(mobility=2.0))
    engine.add_node("C", np.array([2, 0, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(mobility=2.0))

    engine.connect_nodes("A", "B", tension_strength=3.0)
    engine.connect_nodes("B", "C", tension_strength=3.0)

    path, cost = engine.compute_least_resistance_geodesic_path("A", "C")
    assert path == ["A", "B", "C"]
    assert cost < float('inf')


def test_cognitive_heat_generation():
    engine = StructuralDensityGravityEngine(dimensions=4)
    engine.add_node("concept_node", np.array([0, 0, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(density=3.0, mobility=0.1))

    in_force = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out_force = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Opposite direction = maximum mismatch

    heat = engine.compute_cognitive_heat_from_tension_differential(in_force, out_force)
    assert heat > 1.0
    assert engine.nodes["concept_node"].phase.agitation > 1.0


def test_thermal_agitation_melting_and_cooling():
    engine = StructuralDensityGravityEngine(dimensions=4)
    engine.add_node("iron", np.array([0, 0, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(density=5.0, mobility=0.05))

    node = engine.nodes["iron"]
    assert node.phase.classify_materialization() == "Rock"

    # Heat up -> Melting
    engine.apply_thermal_agitation_and_phase_transition(heat_impulse=2.0)
    assert node.phase.agitation > 1.0
    assert node.phase.mobility > 0.5
    assert node.phase.classify_materialization() == "Water"

    # Evaporate further
    engine.apply_thermal_agitation_and_phase_transition(heat_impulse=2.0)
    assert node.phase.classify_materialization() == "Gas_Cloud"

    # Cool down
    for _ in range(5):
        engine.apply_thermal_agitation_and_phase_transition(heat_impulse=-1.0)
    assert node.phase.classify_materialization() == "Rock"


def test_condensation_nucleation():
    engine = StructuralDensityGravityEngine(dimensions=4)

    # Add unbound high-entropy gas cloud nodes
    center = np.array([5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    for i in range(4):
        pos = center + np.array([i * 0.2, 0, 0, 0], dtype=np.float32)
        engine.add_node(f"cloud_{i}", pos, phase=PhaseStateSpectrum(density=0.1, entropy=2.5, mobility=3.0))

    condensed_nodes = engine.trigger_condensation_nucleation(center, radius=3.0)
    assert len(condensed_nodes) == 4

    # Check that condensed nodes are bound with tension bonds
    for nid in condensed_nodes:
        assert engine.nodes[nid].phase.entropy < 0.5
        assert len(engine.nodes[nid].connected_edges) > 0


def test_topological_rupture_physical_and_cognitive():
    engine = StructuralDensityGravityEngine(dimensions=4)

    # Setup a target web across plane x = 0
    engine.add_node("target_left", np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=2.0))
    engine.add_node("target_right", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=2.0))
    engine.connect_nodes("target_left", "target_right", tension_strength=1.0)

    # Add blade at x = 0
    engine.add_node("blade", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(concentration=10.0))

    # Physical Slicing
    event_phys = engine.execute_topological_rupture(
        blade_node_id="blade",
        target_region_center=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        target_radius=3.0,
        is_cognitive=False
    )

    assert event_phys.tension_peak > 10.0
    # Edge between target_left and target_right should be severed
    assert "target_right" not in engine.nodes["target_left"].connected_edges

    # Cognitive Insight Slicing
    engine.add_node("noise_node", np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=0.2))
    engine.add_node("core_insight", np.array([-0.5, -0.5, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=5.0))

    event_cog = engine.execute_topological_rupture(
        blade_node_id="blade",
        target_region_center=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        target_radius=3.0,
        is_cognitive=True
    )

    assert event_cog.is_cognitive is True
    assert event_cog.cognitive_insight_extracted is not None
    assert "core_insight" in event_cog.cognitive_insight_extracted
