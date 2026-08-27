"""
Unit tests for 3-Layer Engine Architecture & Boundary Isolation Enforcement.
Verifies Layer 0, 1, 2 isolation, BoundaryIsolationGuard rule enforcement,
Symbolic Back-tracing, and Multi-Causal Bridge domain transductions.
"""

import math
import pytest
from synaptic_architecture.engine_layering import (
    Layer0GeometricState,
    Layer1TopologicalDynamics,
    Layer2SymbolicCognition,
    CausalLinguisticSymbol,
    SymbolState,
    BoundaryIsolationGuard,
    ObservationSignal,
    ControlDirective,
    CategoryError,
    ReductionViolationError,
    IrreversibleReductionError,
    DomainType,
    MultiCausalBridge,
    ThreeLayerEngine,
)


def test_layer0_geometric_operations():
    l0 = Layer0GeometricState()

    # Dot product
    assert l0.dot_product([1.0, 2.0, 3.0], [4.0, -5.0, 6.0]) == 12.0

    # Distance
    assert math.isclose(l0.euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0)

    # FOV
    assert l0.is_in_field_of_view([1.0, 0.0, 0.0], [1.0, 0.1, 0.0], fov_angle_deg=90.0) is True
    assert l0.is_in_field_of_view([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], fov_angle_deg=90.0) is False

    # Normal vector
    normal = l0.compute_normal_vector([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert normal == [0.0, 0.0, 1.0]


def test_layer1_topological_dynamics_and_c_lens_bandwidth():
    l1 = Layer1TopologicalDynamics(c_lens_bandwidth=10.0, v_critical=5.0)
    sig = ObservationSignal(source_layer=0, signal_type="IMPACT", data={"magnitude": 15.0})

    # Impact magnitude (15.0) is clamped to c_lens_bandwidth (10.0)
    res = l1.process_observation(sig)
    assert res["status"] == "SEALED_EXCESS_TENSION"
    assert res["sealed_count"] == 1
    assert len(l1.sealed_attractors) == 1
    assert l1.sealed_attractors[0].reason == "Friction tension exceeded critical limit V_critical"


def test_boundary_isolation_guard_violations():
    guard = BoundaryIsolationGuard()

    # Upward signal validation
    with pytest.raises(CategoryError):
        guard.validate_upward_signal({"invalid": "raw_dict"})

    # Downward directive validation
    with pytest.raises(CategoryError):
        guard.validate_downward_directive({"invalid": "raw_dict"})

    # No downward reduction
    with pytest.raises(ReductionViolationError):
        guard.enforce_no_downward_reduction("layer0_formula", "Layer2Decision")

    with pytest.raises(ReductionViolationError):
        guard.enforce_no_downward_reduction("cognition_is_fov_check", "Layer0Operation")

    # Level-bounded isomorphism
    with pytest.raises(CategoryError):
        guard.check_level_isomorphism(domain_a_level=0, domain_b_level=2)

    # Information Irreversibility Protection
    with pytest.raises(IrreversibleReductionError):
        guard.check_loss_reduction(loss_value=0.42)


def test_symbolic_back_tracing_and_narrative_traceback():
    l2 = Layer2SymbolicCognition()

    # 1. Register narrative chain
    sym_root = CausalLinguisticSymbol(
        symbol="조직의 관성",
        causal_tension=4.0,
        required_context_depth=2.0,
        parents=()
    )
    sym_child = CausalLinguisticSymbol(
        symbol="조직의 동맥경화",
        causal_tension=8.0,
        required_context_depth=3.0,
        parents=("조직의 관성",)
    )

    l2.reason_narrative(sym_root)
    l2.reason_narrative(sym_child)

    # 2. Issue linguistic feedback targeting '조직의 동맥경화'
    feedback = CausalLinguisticSymbol(
        symbol="해한(解恨)_피드백",
        causal_tension=5.0,
        required_context_depth=1.0,
        parents=(),
        metadata={"target_symbol": "조직의 동맥경화"}
    )

    # Combined tension (8.0 + 5.0 = 13.0) exceeds v_critical (10.0), so origin node gets SEALED
    res = l2.process_symbolic_feedback(feedback)
    assert res["status"] == "SUCCESS"
    assert res["resolved_origin"] == "조직의 동맥경화"
    assert res["action"] == "SEALED_ORIGIN_NODE"
    assert "조직의 동맥경화" in [s.attractor_id for s in l2.acceptance_interface.sealed_symbols] or \
           l2.acceptance_interface.symbolic_registry["조직의 동맥경화"] == SymbolState.SEALED


def test_multi_causal_bridge_cross_domain_transduction():
    bridge = MultiCausalBridge()

    # Trigger physical collision in Spatial domain
    spatial_domain = bridge.domains[DomainType.SPATIAL_DYNAMICS]
    invariant = spatial_domain.trigger_physical_collision(
        obj_a="EntityAlpha",
        obj_b="BoundaryBeta",
        vector="(-0.92, 0.38, -0.05)"
    )
    assert invariant.invariant_id == "SPATIAL_COLLISION_INVARIANT"

    # Transduce generated collision causal invariant across domains
    trans_res = bridge.transduce_causal_invariant(invariant)
    assert trans_res["source_domain"] == "SPATIAL_DYNAMICS"

    # Verify Axiomatic Logic Domain reaction
    axiomatic_domain = bridge.domains[DomainType.AXIOMATIC_LOGIC]
    assert "Axiom_Discontinuous_Impulse_Transition" in axiomatic_domain.active_axioms
    assert "Axiom_Continuous_Motion" not in axiomatic_domain.active_axioms

    # Verify Symbolic Narrative Domain reaction
    narrative_domain = bridge.domains[DomainType.SYMBOLIC_NARRATIVE]
    assert len(narrative_domain.symbolic_conflicts) == 1
    assert narrative_domain.symbolic_conflicts[0]["archetype"] == "외부 충격에 의한 자아/경계막 파열"

    # Cross-domain reversible backtrace
    trace_map = bridge.cross_domain_reversible_backtrace("BoundaryBreachNode")
    assert "SPATIAL_DYNAMICS" in trace_map
    assert "AXIOMATIC_LOGIC" in trace_map
    assert "SYMBOLIC_NARRATIVE" in trace_map


def test_three_layer_engine_integration():
    engine = ThreeLayerEngine()

    # Upward observation processing
    up_res = engine.process_upward({"magnitude": 3.0, "vector": [1.0, 0.0, 0.0]})
    assert isinstance(up_res["layer0_signal"], ObservationSignal)
    assert up_res["layer1_response"]["status"] == "STABLE"

    # Downward directive processing
    directive = engine.process_downward_directive(
        target_layer=0,
        action="ADJUST_SENSOR_ORIENTATION",
        params={"angle": 15.0}
    )
    assert isinstance(directive, ControlDirective)
    assert directive.target_layer == 0
