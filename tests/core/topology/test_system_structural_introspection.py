"""
Unit tests for System Structural Introspection Engine & Dynamic Causal Feeder
"""

import pytest
import numpy as np

from core.topology.system_structural_introspection import SystemStructuralIntrospectionEngine
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def test_scan_codebase_ast():
    engine = SystemStructuralIntrospectionEngine()
    scan_res = engine.scan_codebase_ast()

    assert "total_discovered_modules" in scan_res
    assert scan_res["total_discovered_modules"] > 0
    assert "core.topology.system_structural_introspection" in scan_res["modules"]


def test_generate_isomorphic_nexus_nodes():
    engine = SystemStructuralIntrospectionEngine(max_meta_depth=3, friction_threshold=0.5)
    isomorphic_res = engine.generate_isomorphic_nexus_nodes(depth=1)

    assert isomorphic_res["total_modules"] > 0
    assert isomorphic_res["introspected_modules_count"] > 0
    assert isomorphic_res["introspection_coverage"] == 1.0
    assert "nexus_nodes_created" in isomorphic_res
    assert isomorphic_res["nexus_nodes_created"] > 0


def test_max_meta_depth_recursion_limit():
    engine = SystemStructuralIntrospectionEngine(max_meta_depth=2)
    res = engine.generate_isomorphic_nexus_nodes(depth=5)

    assert res["status"] == "MAX_META_DEPTH_REACHED"


def test_compute_system_causal_field_feedback():
    engine = SystemStructuralIntrospectionEngine()
    engine.generate_isomorphic_nexus_nodes(depth=1)
    feedback_res = engine.compute_system_causal_field_feedback()

    assert "feedback_tensor" in feedback_res
    assert len(feedback_res["feedback_tensor"]) == 4
    assert feedback_res["introspection_status"] in [
        "HIGHLY_SELF_AWARE",
        "HIGH_FRICTION_RECONFIGURATION_NEEDED"
    ]


def test_self_referential_architecture_integration():
    arch_engine = SelfReferentialArchitectureEngine()
    stimulus = {
        "voltage_intent": np.array([1.0, -0.5, 2.0]),
        "introspection_depth": 1
    }
    cycle_res = arch_engine.run_full_self_referential_cycle(stimulus)

    assert "introspection_scan" in cycle_res
    assert "isomorphic_mapping" in cycle_res
    assert "causal_structural_feedback" in cycle_res
    assert cycle_res["isomorphic_mapping"]["introspection_coverage"] == 1.0
