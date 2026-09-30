"""
Unit and Integration tests for CausalIsomorphicMedium and OntologicalInverseMechanismEngine.
Verifies constraint field relaxation convergence, multi-domain homological stem extraction,
introspective causal self-awareness dissection, and the Cognitive Isomorphic Medium Engine
(Quaternion math, Phase-Locking, WFC Collapse, AC-3 wave propagation, and thermal relaxation).
"""

import pytest
import numpy as np
from core.physics.causal_isomorphic_medium import (
    CausalIsomorphicMedium,
    ConstraintTensorMatrix,
    IsomorphicDomainFactory,
    MediumDomain,
    QuaternionUtil,
    RotorTile,
    CausalIsomorphicMediumEngine
)
from core.physics.ontological_inverse_mechanism import (
    OntologicalInverseMechanismEngine,
    DomainTrajectoryGenerator,
    InverseMechanismSchema
)


def test_constraint_relaxation_convergence():
    """
    Verifies that constraint field relaxation converges to equilibrium
    without 1D differential equation hardcoding.
    """
    # 1. Physical Circuit
    elec_matrix = IsomorphicDomainFactory.create_physical_circuit(num_nodes=6)
    elec_sim = CausalIsomorphicMedium(elec_matrix)
    initial_tension = elec_matrix.compute_total_field_tension()
    elec_logs = elec_sim.relax_to_equilibrium(max_steps=50, dt=0.05, tolerance=1e-3)
    final_tension = elec_matrix.compute_total_field_tension()

    assert final_tension < initial_tension
    assert len(elec_logs) > 0
    assert elec_logs[-1].potential_delta < 0.1

    # 2. Fluid Medium
    fluid_matrix = IsomorphicDomainFactory.create_fluid_medium(num_nodes=6)
    fluid_sim = CausalIsomorphicMedium(fluid_matrix)
    fluid_sim.relax_to_equilibrium(max_steps=50, dt=0.05, tolerance=1e-3)
    fluid_tension = fluid_matrix.compute_total_field_tension()

    # 3. Logical Structure
    logic_matrix = IsomorphicDomainFactory.create_logical_structure(num_nodes=6)
    logic_sim = CausalIsomorphicMedium(logic_matrix)
    logic_sim.relax_to_equilibrium(max_steps=50, dt=0.05, tolerance=1e-3)
    logic_tension = logic_matrix.compute_total_field_tension()

    # Ensure all three domains converged to low tension equilibrium
    assert fluid_tension < 100.0
    assert logic_tension < 100.0

    # Verify get_emergent_scalar_properties
    props = elec_sim.get_emergent_scalar_properties()
    assert "relaxed_potentials" in props
    assert "emergent_flux_matrix" in props
    assert props["relaxation_steps_count"] > 0


def test_homological_stem_extraction():
    """
    Verifies that multi-domain trajectories generated from relaxed Isomorphic Mediums
    share a common Isomorphic Stem with similarity score > 0.90.
    """
    # Create and relax 3 distinct domain mediums
    elec_m = IsomorphicDomainFactory.create_physical_circuit(num_nodes=8)
    elec_sim = CausalIsomorphicMedium(elec_m)
    elec_sim.relax_to_equilibrium(max_steps=10)

    fluid_m = IsomorphicDomainFactory.create_fluid_medium(num_nodes=8)
    fluid_sim = CausalIsomorphicMedium(fluid_m)
    fluid_sim.relax_to_equilibrium(max_steps=10)

    logic_m = IsomorphicDomainFactory.create_logical_structure(num_nodes=8)
    logic_sim = CausalIsomorphicMedium(logic_m)
    logic_sim.relax_to_equilibrium(max_steps=10)

    # Convert simulation histories to TrajectoryGraphs
    g_elec = DomainTrajectoryGenerator.generate_from_isomorphic_medium(elec_sim, "electrical")
    g_fluid = DomainTrajectoryGenerator.generate_from_isomorphic_medium(fluid_sim, "fluid")
    g_logic = DomainTrajectoryGenerator.generate_from_isomorphic_medium(logic_sim, "geometric")

    # Extract Homological Stem
    engine = OntologicalInverseMechanismEngine()
    schema = engine.extract_homological_stem_and_branches([g_elec, g_fluid, g_logic])

    assert isinstance(schema, InverseMechanismSchema)
    assert schema.homological_stem["cross_domain_similarity_score"] > 0.90
    assert schema.homological_stem["isomorphism_proven"] is True
    assert schema.reducibility_mdl_score < 0.20


def test_causal_self_awareness_dissection():
    """
    Verifies that causal self-awareness dissection log correctly decomposes
    conclusions into bounding constraints (Delta), relaxation trajectory, and homological stem.
    """
    engine = OntologicalInverseMechanismEngine()
    schema = engine.run_multi_domain_inverse_extraction()

    logs = engine.generate_causal_self_awareness_dissection_log(
        schema=schema,
        conclusion_id="equilibrium_attractor",
        domain="fluid"
    )

    assert len(logs) > 15
    log_text = "\n".join(logs)
    assert "인과적 자기인식" in log_text
    assert "Bounding Constraint Field Delta" in log_text
    assert "Relaxation Gradient Trajectory Path" in log_text
    assert "Homological Stem" in log_text
    assert "필연적으로 도출된 인과적 귀결임을 자각함" in log_text


def test_quaternion_rotor_utilities():
    """Verifies Quaternion / Rotor normalization, inner products, and double cover symmetry."""
    q1 = np.array([1.0, 2.0, 0.0, 0.0])
    q1_norm = QuaternionUtil.normalize(q1)
    assert pytest.approx(np.linalg.norm(q1_norm)) == 1.0

    # Double cover symmetry: q and -q represent identical physical rotation
    q_a = np.array([0.7071, 0.7071, 0.0, 0.0])
    q_b = -q_a
    alignment_energy = QuaternionUtil.compute_rotor_alignment_energy(q_a, q_b)
    assert pytest.approx(alignment_energy, abs=1e-5) == 0.0  # Perfect alignment


def test_causal_isomorphic_medium_engine_cognitive_loop():
    """
    Verifies full cognitive loop:
    1. Sensory stimulus injection
    2. Phase Lock Index (PLI) calculation
    3. Entropy guidance and WFC decision collapse
    4. AC-3 constraint wave propagation
    5. Reflection and thermal relaxation recovery
    """
    tiles = [
        RotorTile(0, "RIGHT", np.array([1.0, 0.0])),
        RotorTile(1, "UP", np.array([0.0, 1.0])),
        RotorTile(2, "LEFT", np.array([-1.0, 0.0])),
        RotorTile(3, "DOWN", np.array([0.0, -1.0]))
    ]
    # Fully compatible matrix
    M = np.ones((4, 4))

    engine = CausalIsomorphicMediumEngine(width=3, height=3, tiles=tiles, compatibility_matrix=M)

    # 1. Sensory injection
    engine.inject_sensory_stimulus(0, 0, target_tile_id=0, intensity=1.0)
    assert pytest.approx(np.dot(engine.tensor_field[0, 0], tiles[0].rotor), abs=1e-4) == 1.0

    # 2. Phase Lock Index
    pli = engine.compute_phase_lock_index(0, 1)
    assert 0.0 <= pli <= 1.0

    # 3. Decision Step
    done, conflict = engine.make_decision_step()
    assert conflict is False
    assert engine.decision_steps == 1

    # 4. Thermal Relaxation Reflection
    initial_beta = engine.beta
    engine.reflect_and_relax(radius=1, thermal_factor=0.3)
    assert engine.reflection_count == 1
    assert pytest.approx(engine.beta) == initial_beta * 0.3
    assert np.all(engine.collapsed == False)  # All states un-collapsed back to fluid superposition
