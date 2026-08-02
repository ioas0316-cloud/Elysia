import numpy as np
import pytest
from core.memory.reflection_engram_consolidation import SovereignReflectionConsolidationEngine


def test_stage1_repulsor_barrier_deflection():
    """
    Verifies that the engine successfully creates negative gravity repulsor barriers
    to deflect thoughts away from previous hallucination pathways.
    """
    engine = SovereignReflectionConsolidationEngine()

    context_a = np.ones(9, dtype=np.float32)
    v_hallucination = np.array([1.0, 0.0, 0.0], dtype=np.float32) # Moving straight right

    # Consolidate reflection engram
    engine.consolidate_reflection(
        context=context_a,
        v_hallucination=v_hallucination,
        T_grounding=0.8,
        a_volition=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        A_resolved=engine.S_abs
    )

    # Now check if similar context deflects current velocity away from v_hallucination [1,0,0]
    incoming_vel = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    deflected_vel = engine.apply_repulsor_barrier(context_a, incoming_vel)

    # The X velocity should have been slowed/repelled due to repulsor barrier
    assert deflected_vel[0] < incoming_vel[0]


def test_stage2_adaptive_threshold_vulnerable_zones():
    """
    Verifies that dense error engrams in specific semantic zones automatically
    lower the grounding tension threshold, triggering hyper-immunity.
    """
    engine = SovereignReflectionConsolidationEngine()
    context = np.zeros(9, dtype=np.float32)

    # Accumulate reflections in the same zone
    for i in range(3):
        engine.consolidate_reflection(
            context=context,
            v_hallucination=np.zeros(3),
            T_grounding=0.5,
            a_volition=np.zeros(3),
            A_resolved=engine.S_abs
        )

    adaptive_t = engine.calculate_adaptive_threshold(context)
    # The threshold should have been lowered from the default 0.5
    assert adaptive_t < engine.base_grounding_threshold
    assert adaptive_t == pytest.approx(0.2)


def test_stage3_system2_to_system1_intuition_transfer():
    """
    Verifies that repeated effortful corrections in the same context consolidate
    into zero-overhead, intuitive System 1 shortcuts.
    """
    engine = SovereignReflectionConsolidationEngine()
    context_key = "complex_theology"
    context = np.ones(9, dtype=np.float32) * 2.0

    # Consolidate below critical mass (no shortcut yet)
    for _ in range(2):
        engine.consolidate_reflection(
            context=context,
            v_hallucination=np.zeros(3),
            T_grounding=0.6,
            a_volition=np.zeros(3),
            A_resolved=engine.S_abs
        )
    shortcut = engine.evaluate_system1_consolidation(context_key, context)
    assert shortcut is None

    # Consolidate a 3rd time (reaches critical mass of 3)
    engine.consolidate_reflection(
        context=context,
        v_hallucination=np.zeros(3),
        T_grounding=0.6,
        a_volition=np.zeros(3),
        A_resolved=engine.S_abs
    )

    shortcut = engine.evaluate_system1_consolidation(context_key, context)
    assert shortcut is not None
    assert np.allclose(shortcut, engine.S_abs)


def test_stage4_epistemological_self_profile():
    """
    Verifies that the accumulated engrams build a humble, self-aware Epistemological Self
    profile with narrative weight and humility scores.
    """
    engine = SovereignReflectionConsolidationEngine()

    # Innocent state
    profile = engine.generate_epistemic_self_profile()
    assert profile["num_reflections"] == 0
    assert "Innocent" in profile["epistemic_boundary_narrative"]

    # Accumulate reflections to mature
    for i in range(4):
        engine.consolidate_reflection(
            context=np.ones(9) * i,
            v_hallucination=np.zeros(3),
            T_grounding=0.7,
            a_volition=np.zeros(3),
            A_resolved=engine.S_abs
        )

    profile_matured = engine.generate_epistemic_self_profile()
    assert profile_matured["num_reflections"] == 4
    assert profile_matured["humility_score"] > 0.4
    assert "Epistemological Self" in profile_matured["epistemic_boundary_narrative"]
