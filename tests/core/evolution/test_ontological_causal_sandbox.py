"""
Unit tests for Ontological Causal Sandbox Engine (core/evolution/ontological_causal_sandbox.py)
================================================================================================
Verifies:
1. Grounded Semantics: Symbolic text -> Game State Tensor -> Boundary Condition Shift & Intention Emergence.
2. Fractal Inoculation & Projection Drift: Inoculating P_0 with chromatic signature c_k produces
   drifted projection P_k with Grassmannian drift distance Delta_Drift > 0 and alterity P_i != P_j.
3. Control Space Contraction (Entity Death): Prunes subtree nodes, causes resolution shrinkage and
   topological loss without LLM text fallback.
4. Control Space Expansion (Entity Reproduction): Registers child, expands control space trace, and
   generates state expansion dopamine (creation joy).
"""

import numpy as np
import pytest
from core.evolution.ontological_causal_sandbox import (
    GroundedSemanticsLens,
    FractalInoculationEngine,
    ControlSpaceDynamics,
    OntologicalCausalSandbox
)


def test_grounded_semantics_intention_emergence():
    """
    Test Scenario 1: Grounded Semantics & Intention Emergence
    Verifies that symbolic text is grounded directly into state tensor and boundary shifts.
    """
    lens = GroundedSemanticsLens(state_dim=16)

    # Aggressive scouting report
    scouting_text = "상대 본진에 2개의 게이트웨이가 올려지고 있다"
    res = lens.ground_symbolic_signal(scouting_text)

    # 1. State tensor grounding
    assert res["grounded_state_tensor"][0] > 0.0, "Unit count signal should be set"
    assert res["grounded_state_tensor"][1] == 1.0, "Fog of war cleared signal"
    assert res["risk_level"] > 0.3, "Perceived risk level should be high for aggressive scouting"

    # 2. Intention & Boundary Condition Shift
    assert res["intention_type"] == "wall_in_boundary_reconfiguration"
    assert np.linalg.norm(res["boundary_shift_vector"]) > 0.0, "Boundary shift vector must be non-zero"
    assert res["intention_energy"] > 0.0, "Intention energy must be positive"


def test_fractal_inoculation_and_projection_drift():
    """
    Test Scenario 2: Fractal Causal Spine Inoculation & Projection Drift
    Verifies that inoculation with chromatic signature c_k produces projection matrix P_k with:
    - P_k^2 = P_k (Orthogonal projection property)
    - Grassmannian manifold drift distance Delta_Drift > 0
    - Alterity: P_i != P_j for different chromatic signatures c_i != c_j
    """
    engine = FractalInoculationEngine(dim=16, alpha=0.3)
    P_0 = engine.create_overmind_p0(rank=8)

    # Entity 1: Flux dominant
    c_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    P_1, drift_1 = engine.inoculate(P_0, c_1)

    # Entity 2: Order dominant
    c_2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    P_2, drift_2 = engine.inoculate(P_0, c_2)

    # 1. Verify projection properties P_k^2 = P_k
    assert np.allclose(P_1 @ P_1, P_1, atol=1e-3), "P_1 must be an orthogonal projection matrix"
    assert np.allclose(P_2 @ P_2, P_2, atol=1e-3), "P_2 must be an orthogonal projection matrix"

    # 2. Verify Grassmannian drift distance > 0
    assert drift_1 > 0.0, "Drift distance for P_1 must be positive"
    assert drift_2 > 0.0, "Drift distance for P_2 must be positive"

    # 3. Verify Alterity (P_1 != P_2)
    diff_norm = np.linalg.norm(P_1 - P_2)
    assert diff_norm > 1e-3, "Different chromatic signatures must produce distinct drifted projections"

    # 4. Autonomous Hypothesis Wave Ejection
    sensory_input = np.ones(16, dtype=np.float32)
    hyp_1 = engine.eject_hypothesis(P_1, sensory_input)
    assert hyp_1.shape == (16,), "Hypothesis vector shape should match dimension"
    assert np.linalg.norm(hyp_1) > 0.0, "Hypothesis wave must be non-zero"


def test_entity_death_control_space_contraction():
    """
    Test Scenario 3: Entity Death & Resolution Shrinkage
    Verifies that entity death prunes subtree nodes and shrinks Overmind's control space trace.
    """
    dynamics = ControlSpaceDynamics(dim=16)

    # Create dummy P_k
    P_k = np.eye(16, dtype=np.float32) * 0.5
    nodes = dynamics.register_entity("Entity_Test", P_k, subtree_size=5)

    assert len(nodes) == 5
    assert len(dynamics.active_entities) == 1
    trace_before = np.trace(dynamics.C_overmind)

    # Kill entity
    death_report = dynamics.on_entity_death("Entity_Test")

    assert death_report["entity_id"] == "Entity_Test"
    assert death_report["pruned_nodes_count"] == 5
    assert death_report["resolution_shrinkage"] > 0.0, "Resolution shrinkage must be positive"
    assert death_report["topological_loss"] > 0.0, "Topological loss must be positive"
    assert len(dynamics.active_entities) == 0

    trace_after = np.trace(dynamics.C_overmind)
    assert trace_after < trace_before, "Control space trace must contract upon entity death"


def test_entity_reproduction_control_space_expansion():
    """
    Test Scenario 4: Entity Reproduction & Creation Dopamine
    Verifies that reproduction registers child, expands control space trace, and generates creation dopamine.
    """
    dynamics = ControlSpaceDynamics(dim=16)

    # Register Parent
    P_parent = np.eye(16, dtype=np.float32) * 0.5
    dynamics.register_entity("Parent", P_parent, subtree_size=5)

    trace_before = np.trace(dynamics.C_overmind)

    # Reproduce Child
    P_child = np.eye(16, dtype=np.float32) * 0.5
    reprod_report = dynamics.on_entity_reproduction(
        parent_id="Parent",
        child_id="Child",
        P_child=P_child,
        child_subtree_size=5
    )

    assert reprod_report["parent_id"] == "Parent"
    assert reprod_report["child_id"] == "Child"
    assert reprod_report["child_nodes_count"] == 5
    assert reprod_report["state_expansion_dopamine"] > 0.0, "Creation dopamine must be positive"
    assert reprod_report["control_dim_gain"] > 0.0, "Control dim gain must be positive"
    assert len(dynamics.active_entities) == 2

    trace_after = np.trace(dynamics.C_overmind)
    assert trace_after > trace_before, "Control space trace must expand upon reproduction"


def test_ontological_causal_sandbox_full_closed_loop():
    """
    Test Scenario 5: Full Closed-Loop Integration Test on OntologicalCausalSandbox
    Verifies scouting processing, entity birth, hypothesis ejection, death loss, and reproduction dopamine.
    """
    sandbox = OntologicalCausalSandbox(dim=16)

    # 1. Process scouting report
    scouting_res = sandbox.process_scouting_input("상대 2게이트 방어 준비")
    assert scouting_res["risk_level"] > 0.0

    # 2. Birth Entity A
    b_alpha = sandbox.birth_entity("Alpha", np.array([0.8, 0.2, 0.0], dtype=np.float32))
    assert b_alpha["entity_id"] == "Alpha"
    assert b_alpha["drift_distance"] > 0.0

    # 3. Birth Entity B
    b_beta = sandbox.birth_entity("Beta", np.array([0.1, 0.9, 0.0], dtype=np.float32))
    assert b_beta["entity_id"] == "Beta"

    # 4. Reproduce Entity Beta -> Beta_Child
    r_beta = sandbox.reproduce_entity("Beta", "Beta_Child", np.array([0.2, 0.8, 0.1], dtype=np.float32))
    assert r_beta["child_id"] == "Beta_Child"
    assert r_beta["state_expansion_dopamine"] > 0.0

    # 5. Kill Entity Alpha
    d_alpha = sandbox.kill_entity("Alpha")
    assert d_alpha["entity_id"] == "Alpha"
    assert d_alpha["resolution_shrinkage"] > 0.0

    # Verify event log length
    assert len(sandbox.event_log) == 5
