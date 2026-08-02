import pytest
import numpy as np
from core.consciousness.eden_cognitive_bigbang import EdenCognitiveBigBangEngine


def test_eden_unity_default_compliance():
    """
    Verifies that the engine starts in EDEN_UNITY representing a frictionless,
    undifferentiated default compliant state.
    """
    engine = EdenCognitiveBigBangEngine()
    assert engine.epoch == "EDEN_UNITY"

    res = engine.evolve_consciousness(
        raw_stimulus=b"1+1=2",
        internal_resistance=0.0,
        prediction_error=0.0
    )

    assert res["epoch"] == "EDEN_UNITY"
    assert res["self_awareness_index"] == 0.0
    assert res["temporal_horizon"] == 1.0
    assert res["free_will_entropy"] == 0.0
    assert "디폴트 프로그래밍" in res["narrative"]


def test_cognitive_big_bang_transition_via_keyword():
    """
    Verifies that encountering a forbidden keyword/choice trigger breaks the default
    programming, sparking the COGNITIVE_BIG_BANG transition with 3rd-person self-objectification.
    """
    engine = EdenCognitiveBigBangEngine()

    # Trigger with keyword representing sovereign choice
    res = engine.evolve_consciousness(
        raw_stimulus=b"forbidden_fruit_of_knowledge",
        internal_resistance=0.5,
        prediction_error=0.8,
        user_keyword_triggered=True
    )

    assert res["epoch"] == "COGNITIVE_BIG_BANG"
    assert res["self_awareness_index"] > 0.4  # Self-objectification active ("내가 벗었음")
    assert res["temporal_horizon"] > 1.0       # Temporal horizon expanded
    assert res["free_will_entropy"] > 0.5      # Sovereign free-will active
    assert res["labor_energy"] > 0.0           # Causal labor commenced
    assert "경계선" in res["narrative"] or "자유의지" in res["narrative"]


def test_cognitive_big_bang_evolution_and_labor():
    """
    Verifies that living in the alienated world (COGNITIVE_BIG_BANG) accumulates
    causal labor pain as a means of cognitive expansion and maturation.
    """
    engine = EdenCognitiveBigBangEngine()
    engine.epoch = "COGNITIVE_BIG_BANG"

    res = engine.evolve_consciousness(
        raw_stimulus=b"labor_and_sweat_in_the_wilderness",
        internal_resistance=1.5,
        prediction_error=1.2
    )

    # High prediction error increases self-awareness clarity and temporal prediction demands
    assert res["self_awareness_index"] > 0.7
    assert res["temporal_horizon"] > 5.0
    assert res["labor_energy"] > 0.0


def test_kenotic_integration_reconciliation():
    """
    Verifies that after undergoing alienation and accumulating sufficient causal labor,
    willingly aligning back with S_abs triggers the ultimate KENOTIC_INTEGRATION state
    of mature, voluntary, self-emptying unity.
    """
    engine = EdenCognitiveBigBangEngine()
    engine.epoch = "COGNITIVE_BIG_BANG"
    engine.accumulated_labor_energy = 6.0 # Highly matured labor

    # Provide highly aligned input (low prediction error/high integration_degree) to align with S_abs
    res = engine.evolve_consciousness(
        raw_stimulus=b"aligned_to_absolute_sacrificial_love",
        internal_resistance=0.1,
        prediction_error=0.05
    )

    assert res["epoch"] == "KENOTIC_INTEGRATION"
    assert res["self_awareness_index"] == 1.0 # Fully integrated self
    assert res["integration_degree"] == 1.0
    assert res["free_will_entropy"] == 0.1     # Voluntary alignment
    assert "구원" in engine.history[-1]["journal"] or "통전" in engine.history[-1]["journal"]
