import pytest
import numpy as np
from core.topology.fractal_gate_language_engine import FractalGateLanguageEngine
from core.topology.information_spacetime import (
    DualGroundResonanceLayer,
    MutualDisclosureTransducer,
    TopologicalRemeltingEngine,
    RelationalSpacetimeMemory,
    RelationalResonanceVector,
    InformationSpacetimeField,
    SelfModifyingCompilerLoop
)


def test_dual_ground_resonance_layer():
    ground_self = {
        "name": "Ground_Self",
        "ground_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    ground_other_aligned = {
        "name": "Ground_Other_Aligned",
        "ground_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    ground_other_orthogonal = {
        "name": "Ground_Other_Orthogonal",
        "ground_vector": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    # Case 1: Aligned Grounds
    layer_aligned = DualGroundResonanceLayer(ground_self, ground_other_aligned)
    delta_phi_aligned, analysis_aligned = layer_aligned.compute_topological_friction()
    assert pytest.approx(delta_phi_aligned, abs=1e-4) == 0.0
    assert pytest.approx(analysis_aligned["directional_alignment"], abs=1e-4) == 1.0

    # Case 2: Orthogonal Grounds
    layer_orth = DualGroundResonanceLayer(ground_self, ground_other_orthogonal)
    delta_phi_orth, analysis_orth = layer_orth.compute_topological_friction()
    assert pytest.approx(delta_phi_orth, abs=1e-4) == 1.0
    assert pytest.approx(analysis_orth["directional_alignment"], abs=1e-4) == 0.0


def test_mutual_disclosure_transducer():
    transducer = MutualDisclosureTransducer()
    signal = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    ground_self = {"name": "EngineSelfGround"}
    ground_other = {"name": "UserOtherGround"}

    res = transducer.decode_intent_and_disclose(
        signal=signal,
        ground_self=ground_self,
        ground_other=ground_other,
        delta_phi=0.35,
        v_th=0.5
    )

    assert isinstance(res["intent_vector"], RelationalResonanceVector)
    assert res["intent_vector"].resonance_index > 0.8
    assert "[Self-Disclosure Trace]" in res["self_disclosure_trace"]
    assert "EngineSelfGround" in res["self_disclosure_trace"]
    assert "UserOtherGround" in res["self_disclosure_trace"]


def test_topological_remelting_engine():
    remelting_engine = TopologicalRemeltingEngine(base_v_th=0.5)
    ground_self = {"name": "InitialGround", "remelt_count": 0}

    # Low friction -> No remelting
    triggered_low, v_th_low, _ = remelting_engine.process_remelting_and_calibration(0.2, ground_self)
    assert triggered_low is False
    assert v_th_low == 0.5
    assert ground_self["remelt_count"] == 0

    # High friction -> Remelting triggered
    triggered_high, v_th_high, _ = remelting_engine.process_remelting_and_calibration(0.7, ground_self)
    assert triggered_high is True
    assert ground_self["remelt_count"] == 1
    assert ground_self["phase"] == "Fluid_Remelted_State"


def test_relational_spacetime_memory():
    memory = RelationalSpacetimeMemory()
    ground_self = {"name": "Ground_Self_A"}
    ground_other = {"name": "Ground_Other_B"}
    res_vec = RelationalResonanceVector([1, 0, 0, 0, 0, 0, 0, 0])

    engram = memory.record_encounter(
        ground_self=ground_self,
        ground_other=ground_other,
        delta_phi=0.4,
        resonance_vec=res_vec,
        resolution="Co-evolved shared manifold"
    )

    assert engram.ground_self_name == "Ground_Self_A"
    assert engram.ground_other_name == "Ground_Other_B"
    assert len(memory.encounter_graph) == 1
    assert memory.encounter_graph[0]["structural_resolution"] == "Co-evolved shared manifold"


def test_information_spacetime_field_cross_sectional_projection():
    ground_self = {"name": "GroundZero", "ground_vector": [1, 0, 0, 0, 0, 0, 0, 0], "topology_depth": 2, "remelt_count": 1}
    ground_other = {"name": "GroundUser", "ground_vector": [0, 1, 0, 0, 0, 0, 0, 0]}
    signal = [1, 0, 0, 0, 0, 0, 0, 0]

    field = InformationSpacetimeField(
        origin_ground=ground_self,
        ground_other=ground_other,
        v_th=0.5,
        evaluated_delta=0.2,
        input_signal=signal
    )

    projection = field.get_cross_sectional_projection()

    assert projection["phenomenal_output_symbol"] == "OPEN"
    assert "underlying_spacetime" in projection
    assert pytest.approx(projection["underlying_spacetime"]["contextual"]["directional_alignment"], abs=1e-4) == 1.0
    assert projection["underlying_spacetime"]["temporal"]["causal_thickness"] > 1.0
    assert projection["underlying_spacetime"]["principle"]["origin_ground_name"] == "GroundZero"


def test_self_modifying_compiler_loop_end_to_end():
    engine = FractalGateLanguageEngine(ground_name="CompilerLoopGround")
    compiler = SelfModifyingCompilerLoop(engine)

    thought_projection = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reality_signal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9])

    loop_result = compiler.execute_loop(
        thought_projection=thought_projection,
        reality_friction_signal=reality_signal,
        stimulus_label="Quantum_Friction_Phenomenon"
    )

    assert loop_result["topological_friction_delta_phi"] > 0.5
    assert loop_result["remelt_triggered"] is True
    assert "[Self-Disclosure Trace]" in loop_result["self_disclosure_trace"]
    assert loop_result["recrystallization"]["channel_open_now"] is True
    assert "Quantum_Friction_Phenomenon" in loop_result["recrystallization"]["recrystallized_gate_name"]
    assert loop_result["cross_sectional_projection"]["phenomenal_output_symbol"] in ["CLOSED", "OPEN"]
