import pytest
import numpy as np
from synaptic_architecture.topological_causal_replication import (
    CausalTopologicalReplicationEngine,
    CausalStructuralNode,
    RelationalContinuityBeam,
    ContinuityFlowTrace
)


def test_engine_initialization():
    engine = CausalTopologicalReplicationEngine()
    eval_res = engine.compute_topological_isomorphism_ratio()
    assert eval_res["isomorphism_ratio"] == 1.0
    assert eval_res["total_nodes"] == 0
    assert eval_res["total_beams"] == 0
    assert eval_res["is_pure_non_vector_mirror"] is True
    assert eval_res["has_arbitrary_flattening"] is False


def test_code_continuum_replication():
    engine = CausalTopologicalReplicationEngine()
    code_structure = {
        "nodes": [
            {"id": "ast_root", "invariants": ["ast_validity"], "address": (0.0, 0.0)},
            {"id": "cfg_block1", "invariants": ["flow_control"], "address": (0.1, 0.2)},
            {"id": "vm_op_return", "invariants": ["exit_code_zero"], "address": (0.2, 0.4)}
        ],
        "edges": [
            {"src": "ast_root", "dst": "cfg_block1", "type": "execution_flow", "conductance": 0.99, "impedance": 0.01},
            {"src": "cfg_block1", "dst": "vm_op_return", "type": "execution_flow", "conductance": 0.97, "impedance": 0.03}
        ]
    }
    nodes = engine.replicate_code_continuum(code_structure)
    assert len(nodes) == 3
    assert "ast_root" in engine.nodes
    assert ("ast_root", "cfg_block1") in engine.beams
    assert engine.beams[("ast_root", "cfg_block1")].conductance == 0.99


def test_linguistic_continuum_replication():
    engine = CausalTopologicalReplicationEngine()
    language_context = {
        "concepts": [
            {"id": "logos", "address": (0.1, 0.1), "invariants": ["absolute_truth"]},
            {"id": "reflection", "address": (0.2, 0.2), "invariants": ["mirroring"]}
        ],
        "relations": [
            {"src": "logos", "dst": "reflection", "type": "semantic_context", "conductance": 0.95}
        ]
    }
    nodes = engine.replicate_linguistic_continuum(language_context)
    assert len(nodes) == 2
    assert "logos" in engine.nodes
    assert ("logos", "reflection") in engine.beams


def test_environmental_continuum_replication():
    engine = CausalTopologicalReplicationEngine()
    env_state = {
        "elements": [
            {"id": "sun_light", "position": (0.0, 100.0, 0.0), "physical_laws": ["radiation"], "energy": 100.0},
            {"id": "leaf_photosynthesis", "position": (0.0, 1.0, 0.0), "physical_laws": ["energy_conversion"], "energy": 10.0}
        ],
        "interactions": [
            {"src": "sun_light", "dst": "leaf_photosynthesis", "type": "physical_friction", "conductance": 0.92, "friction": 0.08}
        ]
    }
    nodes = engine.replicate_environmental_continuum(env_state)
    assert len(nodes) == 2
    assert "sun_light" in engine.nodes
    assert ("sun_light", "leaf_photosynthesis") in engine.beams


def test_trace_causal_continuity_flow_success():
    engine = CausalTopologicalReplicationEngine()
    code_structure = {
        "nodes": [
            {"id": "n1", "invariants": ["a"], "address": (0, 0)},
            {"id": "n2", "invariants": ["b"], "address": (1, 1)},
            {"id": "n3", "invariants": ["c"], "address": (2, 2)}
        ],
        "edges": [
            {"src": "n1", "dst": "n2", "type": "execution_flow", "conductance": 0.9, "impedance": 0.1},
            {"src": "n2", "dst": "n3", "type": "execution_flow", "conductance": 0.8, "impedance": 0.2}
        ]
    }
    engine.replicate_code_continuum(code_structure)
    trace = engine.trace_causal_continuity_flow("n1", "n3")

    assert trace.path == ["n1", "n2", "n3"]
    assert pytest.approx(trace.continuity_preservation_ratio, 0.001) == 0.72
    assert trace.is_isomorphic_mirror is True


def test_trace_causal_continuity_flow_non_existent_node():
    engine = CausalTopologicalReplicationEngine()
    with pytest.raises(ValueError):
        engine.trace_causal_continuity_flow("non_existent_start", "non_existent_end")


def test_topological_isomorphism_ratio_evaluation():
    engine = CausalTopologicalReplicationEngine()
    code_structure = {
        "nodes": [
            {"id": "a", "address": (0, 0)},
            {"id": "b", "address": (1, 1)}
        ],
        "edges": [
            {"src": "a", "dst": "b", "conductance": 0.9, "impedance": 0.1}
        ]
    }
    engine.replicate_code_continuum(code_structure)
    eval_res = engine.compute_topological_isomorphism_ratio()

    assert eval_res["total_nodes"] == 2
    assert eval_res["total_beams"] == 1
    assert pytest.approx(eval_res["isomorphism_ratio"], 0.001) == 0.9
    assert eval_res["is_pure_non_vector_mirror"] is True
    assert eval_res["has_arbitrary_flattening"] is False


def test_multi_medium_isomorphic_coexistence():
    engine = CausalTopologicalReplicationEngine()

    code_struct = {
        "nodes": [{"id": "code_node", "address": (0, 0)}],
        "edges": []
    }
    lang_struct = {
        "concepts": [{"id": "lang_concept", "address": (1, 1)}],
        "relations": []
    }
    env_struct = {
        "elements": [{"id": "env_elem", "position": (2, 2)}],
        "interactions": []
    }

    engine.replicate_code_continuum(code_struct)
    engine.replicate_linguistic_continuum(lang_struct)
    engine.replicate_environmental_continuum(env_struct)

    assert len(engine.nodes) == 3
    assert engine.nodes["code_node"].medium_type == "code"
    assert engine.nodes["lang_concept"].medium_type == "language"
    assert engine.nodes["env_elem"].medium_type == "environment"


def test_continuity_preservation_disconnected_nodes():
    engine = CausalTopologicalReplicationEngine()
    code_struct = {
        "nodes": [
            {"id": "island1", "address": (0, 0)},
            {"id": "island2", "address": (1, 1)}
        ],
        "edges": []
    }
    engine.replicate_code_continuum(code_struct)
    trace = engine.trace_causal_continuity_flow("island1", "island2")

    assert trace.path == ["island1"]
    assert trace.continuity_preservation_ratio == 0.0
    assert trace.is_isomorphic_mirror is False


def test_phase_alignment_integrity():
    engine = CausalTopologicalReplicationEngine()
    code_struct = {
        "nodes": [
            {"id": "start", "address": (0, 0)},
            {"id": "end", "address": (1, 1)}
        ],
        "edges": [
            {"src": "start", "dst": "end", "conductance": 1.0, "impedance": 0.0}
        ]
    }
    engine.replicate_code_continuum(code_struct)
    beam = engine.beams[("start", "end")]
    assert beam.phase_alignment == 1.0
