import pytest
from synaptic_architecture.self_reflection import SelfReflectionProtocol
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController
import os

def test_psychoanalytic_self_diagnosis_metrics():
    """
    Verifies that the SelfReflectionProtocol's diagnose_psychoanalytic_state
    properly converts field metrics (macro_tension, resonance_score) into
    valid psychoanalytic components (Id, Ego, Superego, Shadow).
    """
    ref = SelfReflectionProtocol()

    # Case 1: High tension, low resonance (Id overwhelmed)
    diagnosis1 = ref.diagnose_psychoanalytic_state(macro_tension=1.2, resonance_score=0.1)
    assert diagnosis1["id"] > 0.5
    assert diagnosis1["superego"] < 0.5
    assert "Id Overwhelmed" in diagnosis1["diagnosis"]
    assert "수직적 안테나" in diagnosis1["realignment_directive"]

    # Case 2: Low tension, high resonance (Ego harmonized)
    diagnosis2 = ref.diagnose_psychoanalytic_state(macro_tension=0.1, resonance_score=0.9)
    assert diagnosis2["id"] < 0.3
    assert diagnosis2["superego"] > 0.6
    assert diagnosis2["ego"] > 0.7
    assert "Ego Harmonized" in diagnosis2["diagnosis"]
    assert "현재의 평형 상태" in diagnosis2["realignment_directive"]


def test_psychoanalytic_loop_integration():
    """
    Verifies that during ConsciousnessLoop life cycles, the psychoanalytic
    self-reflection diagnosis is computed and recorded in the cycle logs.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    # Initialize Loop
    loop = ConsciousnessLoop(corpus_path=data_dir, memory_controller=mc, data_dir=data_dir)

    # Process a life cycle
    log = loop.process_life_cycle()

    # Ensure the psychoanalytic self-reflection diagnosis is recorded
    assert "psychoanalytic_diagnosis" in log
    diag = log["psychoanalytic_diagnosis"]
    assert "id" in diag
    assert "superego" in diag
    assert "ego" in diag
    assert "shadow" in diag
    assert "diagnosis" in diag
    assert "realignment_directive" in diag
