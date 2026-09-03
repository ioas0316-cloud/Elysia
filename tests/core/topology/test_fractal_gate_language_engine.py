import pytest
import numpy as np
from core.topology.fractal_gate_language_engine import (
    FractalGateLanguageEngine,
    PrimitiveGate,
    CombinationalCircuit,
    MetaInformationPacket
)

def test_primitive_gate_evaluation():
    ref_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gate = PrimitiveGate("test_gate_1", "Test_Gate", v_th=0.5, reference_vector=ref_vec)

    # Signal 1: Perfectly aligned
    signal_aligned = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    is_open, delta, eff_v = gate.evaluate(signal_aligned)
    assert is_open is True
    assert pytest.approx(delta, abs=1e-4) == 0.0
    assert eff_v == 0.5

    # Signal 2: Orthogonal / high friction
    signal_orthogonal = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    is_open, delta, eff_v = gate.evaluate(signal_orthogonal)
    assert is_open is False
    assert pytest.approx(delta, abs=1e-4) == 1.0


def test_qualitative_phase_shift():
    engine = FractalGateLanguageEngine(ground_name="Initial_Family_Ground")

    couple_a = {"name": "Person_A", "vector": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    couple_b = {"name": "Person_B", "vector": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    res = engine.execute_qualitative_phase_shift(couple_a, couple_b, catalyst_gate_id="gate_marriage")

    assert res["phase_shift_occurred"] is True
    assert res["qualitative_state"] == "Family_Bound_State"
    assert "EmergentGround" in res["name"]

    # Verify MetaInformationPacket engraving
    assert len(engine.engraved_packets) == 1
    packet = engine.engraved_packets[0]
    assert packet.channel_open is True
    assert "Passed through" in packet.explanation or "passed through" in packet.explanation


def test_discourse_dynamic_state_machine():
    engine = FractalGateLanguageEngine(ground_name="DiscourseGround")

    gate1 = PrimitiveGate("g1", "Word_Apple", v_th=0.6, reference_vector=[1, 0, 0, 0, 0, 0, 0, 0])
    gate2 = PrimitiveGate("g2", "Word_Red", v_th=0.6, reference_vector=[1, 0.2, 0, 0, 0, 0, 0, 0])
    circuit1 = CombinationalCircuit("sent_1", [gate1, gate2], connection_topology="series")

    gate3 = PrimitiveGate("g3", "Word_Fruit", v_th=0.6, reference_vector=[1, 0, 0.1, 0, 0, 0, 0, 0])
    circuit2 = CombinationalCircuit("sent_2", [gate3], connection_topology="parallel")

    signal1 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    signal2 = np.array([1, 0, 0.1, 0, 0, 0, 0, 0])

    discourse_res = engine.process_discourse(
        sentences=[circuit1, circuit2],
        signal_stream=[signal1, signal2]
    )

    assert discourse_res["steps_processed"] == 2
    assert len(engine.engraved_packets) == 2


def test_structural_plasticity_loop_unmapped_stimulus():
    engine = FractalGateLanguageEngine(ground_name="PlasticityGround")

    # Novel, completely unmapped stimulus vector
    unmapped_signal = np.array([0, 0, 0, 0, 0, 0, 0.8, 0.8])

    res = engine.structural_plasticity_loop(unmapped_signal, stimulus_label="Quantum_Qualia")

    assert res["unmapped_detected"] is True
    assert res["initial_friction_delta"] > 0.55
    assert "Gate_Quantum_Qualia" in res["recrystallized_gate_name"]
    assert res["rectified_delta_after"] < 0.01
    assert res["channel_open_now"] is True

    # Confirm newly recrystallized gate exists in persistent graph
    assert res["recrystallized_gate_id"] in engine.causal_graph.nodes

    # Self-explanation check
    explanation_narrative = engine.explain_self_reasoning()
    assert "Self-Elucidation" in explanation_narrative
    assert "Gate_Quantum_Qualia" in explanation_narrative
