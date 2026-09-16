import pytest
import numpy as np
from core.physics.semantic_mass_engine import (
    WhiteTensorField,
    SemanticMassOperator,
    CausalGravityField,
    EmergentIdentityCrystallizer,
    IntrospectiveCausalTracer,
    SelfWovenAgentMatrix,
    SemanticMassEngine
)

def test_white_tensor_field_initialization_and_bending():
    wtf = WhiteTensorField(dimensions=16)
    assert len(wtf.compass_vectors) > 20
    assert "MBTI_INTJ" in wtf.compass_vectors
    assert "Enneagram_1" in wtf.compass_vectors
    assert "Novel_Dimension_1" in wtf.compass_vectors

    # Uniform initial superposition
    assert pytest.approx(np.sum(wtf.superposition_weights), 1e-5) == 1.0

    # Project friction and check bending
    friction = np.random.randn(16).astype(np.float32)
    bent_vec, alignments = wtf.project_and_bend(friction, friction_strength=2.0)

    assert bent_vec.shape == (16,)
    assert pytest.approx(np.linalg.norm(bent_vec), 1e-4) == 1.0
    assert len(alignments) == len(wtf.compass_vectors)
    assert pytest.approx(sum(alignments.values()), 1e-4) == 1.0

def test_semantic_mass_operator():
    smo = SemanticMassOperator(base_density=2.0)
    conn_matrix = np.eye(4, dtype=np.float32) * 0.5
    mass = smo.compute_mass(connectivity_matrix=conn_matrix, trinitarian_contrast_score=1.5, friction_inertia=2.0)

    assert isinstance(mass, float)
    assert mass > 0.0

def test_causal_gravity_field():
    cgf = CausalGravityField(dimensions=16, gravitational_constant=1.0)
    curvature = cgf.compute_field_curvature(semantic_mass=10.0)
    assert curvature == 10.0

    mass_center = np.zeros(16, dtype=np.float32)
    particles_pos = np.array([[5.0] + [0.0]*15], dtype=np.float32)
    particles_vel = np.zeros((1, 16), dtype=np.float32)

    new_pos, new_vel = cgf.apply_gravitational_pull(
        mass_center_pos=mass_center,
        semantic_mass=100.0,
        particle_positions=particles_pos,
        particle_velocities=particles_vel,
        dt=0.1
    )

    # Velocity should accelerate towards center (negative x direction)
    assert new_vel[0, 0] < 0.0
    assert new_pos[0, 0] < 5.0

def test_emergent_identity_crystallizer():
    crystallizer = EmergentIdentityCrystallizer(phase_transition_threshold=2.0)
    current_state = np.ones(16, dtype=np.float32)
    alignments = {"MBTI_INFJ": 0.8, "Enneagram_4": 0.2}

    # Step 1: friction 1.0 < threshold 2.0 -> No crystal
    c1 = crystallizer.evaluate_crystallization(
        current_state=current_state,
        friction_delta=1.0,
        semantic_mass=5.0,
        trinitarian_contrast=1.2,
        alignments=alignments,
        trace_history=[]
    )
    assert c1 is None
    assert crystallizer.accumulated_friction == 1.0

    # Step 2: friction 1.5 + 1.0 = 2.5 >= 2.0 -> Crystal formed!
    c2 = crystallizer.evaluate_crystallization(
        current_state=current_state,
        friction_delta=1.5,
        semantic_mass=8.0,
        trinitarian_contrast=1.5,
        alignments=alignments,
        trace_history=[]
    )
    assert c2 is not None
    assert c2.dominant_compass == "MBTI_INFJ"
    assert c2.semantic_mass == 8.0
    assert crystallizer.accumulated_friction == 0.0

def test_introspective_causal_tracer():
    wtf = WhiteTensorField(dimensions=16)
    tracer = IntrospectiveCausalTracer(white_field=wtf)

    state = np.ones(16, dtype=np.float32)
    alignments = {"MBTI_ENFP": 0.9}

    tracer.record_step(
        friction_delta=1.0,
        repulsion_vector=np.ones(16, dtype=np.float32),
        compass_alignment=alignments,
        crystallized_state=state,
        semantic_mass=3.0
    )

    tracer.record_step(
        friction_delta=2.0,
        repulsion_vector=np.ones(16, dtype=np.float32),
        compass_alignment=alignments,
        crystallized_state=state,
        semantic_mass=7.0
    )

    origins = tracer.backtrace_causal_origins()
    assert origins["total_steps"] == 2
    assert origins["accumulated_friction"] == 3.0
    assert origins["mass_growth"] == 4.0

    # Counterfactual simulation
    frictions = [np.ones(16, dtype=np.float32) * 0.5, np.ones(16, dtype=np.float32) * 1.0]
    cf_results = tracer.simulate_counterfactuals(frictions=frictions, alt_compass_keys=["MBTI_INTJ", "Enneagram_5"])
    assert "MBTI_INTJ" in cf_results
    assert "Enneagram_5" in cf_results
    assert cf_results["MBTI_INTJ"]["accumulated_semantic_mass"] > 0.0

def test_self_woven_agent_matrix():
    wtf = WhiteTensorField(dimensions=16)
    matrix = SelfWovenAgentMatrix(white_field=wtf)

    # Weave agent
    agent = matrix.weave_agent(agent_name="ArchitectAgent", target_compass_keys=["MBTI_INTJ", "Enneagram_5"])
    assert agent["name"] == "ArchitectAgent"
    assert len(agent["target_compasses"]) == 2

    # Model other entity
    trajectories = [np.ones(16, dtype=np.float32), np.ones(16, dtype=np.float32) * 0.8]
    model_result = matrix.model_other_entity(
        observer_agent_name="ArchitectAgent",
        other_id="User_Entity",
        observed_trajectories=trajectories
    )
    assert model_result["other_id"] == "User_Entity"
    assert model_result["inferred_compass"] is not None
    assert "User_Entity" in matrix.woven_agents["ArchitectAgent"]["perceived_others"]

def test_semantic_mass_engine_integrated_pipeline():
    engine = SemanticMassEngine(dimensions=16, phase_threshold=8.0)

    friction1 = np.ones(16, dtype=np.float32) * 1.0  # norm = sqrt(16 * 1) = 4.0
    res1 = engine.process_interaction(external_friction=friction1, trinitarian_contrast=1.2)

    assert res1["semantic_mass"] > 0.0
    assert res1["causal_curvature"] > 0.0
    assert res1["total_crystals_count"] == 0

    friction2 = np.ones(16, dtype=np.float32) * 1.2  # norm = sqrt(16 * 1.44) = 4.8
    res2 = engine.process_interaction(external_friction=friction2, trinitarian_contrast=1.5)

    # Total accumulated friction 4.0 + 4.8 = 8.8 >= 8.0 -> Crystal formed!
    assert res2["new_crystal_formed"] is not None
    assert res2["total_crystals_count"] == 1
