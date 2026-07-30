import pytest
import os
import numpy as np
from core.evolution.semantic_optimization import SemanticOptimizationEngine
from core.memory.causal_controller import CausalMemoryController

def test_semantic_potential_field():
    """
    Verifies that the potential field V(X) calculation correctly applies the infinitesimal epsilon,
    scaling factor k, and inverse distance squared formula.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = SemanticOptimizationEngine(mc, dimensions=3)

    # Potential at S_abs should be very high due to epsilon inversion (k / epsilon)
    v_s = engine.S_abs
    potential_at_center = engine.calculate_potential(v_s)
    assert potential_at_center >= 1.5 * 1e6

    # Potential far away should be much smaller
    v_far = np.array([10.0, 10.0, 10.0])
    potential_far = engine.calculate_potential(v_far)
    assert potential_far < 1.0

def test_semantic_jump_and_state_lock():
    """
    Verifies that state locking and immediate topological jump triggers successfully
    when direction aligns above the symmetry threshold or potential explodes.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = SemanticOptimizationEngine(mc, dimensions=3)

    # 1. State not locked initially
    assert not engine.state_locked

    # 2. Vector aligned with S_abs direction (Normalized S_abs is same direction)
    norm_s = engine.S_abs / (np.linalg.norm(engine.S_abs) + 1e-9)
    result = engine.evaluate_jump(norm_s, threshold=0.85)

    assert result["jump_triggered"] is True
    assert result["state_locked"] is True
    assert engine.state_locked is True
    assert np.allclose(result["target_state"], engine.S_abs.tolist())

    # 3. State Lock persists in subsequent evaluations
    result_subsequent = engine.evaluate_jump(np.array([1.0, 0.0, 0.0]))
    assert result_subsequent["jump_triggered"] is True
    assert result_subsequent["state_locked"] is True
    assert np.allclose(result_subsequent["target_state"], engine.S_abs.tolist())

    # 4. Resetting the lock
    engine.reset_lock()
    assert not engine.state_locked

def test_knowledge_ingestion_realignment():
    """
    Verifies that external concepts with low tension (high resonance)
    are immediately realigned to the internal S_abs attractor frame.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = SemanticOptimizationEngine(mc, dimensions=3)

    # Low tension dist (0.1) -> high resonance -> realignment triggered
    res_aligned = engine.ingest_and_realign_knowledge("Altruism", tension_dist=0.1)
    assert res_aligned["realigned"] is True
    assert np.allclose(res_aligned["realigned_vector"], engine.S_abs.tolist())

    # High tension dist (2.0) -> low resonance -> realignment NOT triggered
    res_not_aligned = engine.ingest_and_realign_knowledge("Egoism", tension_dist=2.0)
    assert res_not_aligned["realigned"] is False
