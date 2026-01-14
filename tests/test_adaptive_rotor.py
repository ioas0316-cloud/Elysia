"""
Test: Adaptive Rotor System
===========================
Verifies that the rotors breathe and shift gears based on intent and stress.
"""

import pytest
from Core.Engine.governance_engine import GovernanceEngine

@pytest.fixture
def engine():
    return GovernanceEngine()

def test_initial_state(engine):
    assert engine.current_gear.name == "Idle"
    # Base RPM is 60.0, Idle multiplier is 1.0
    # adapt(0.0, 0.0) implies Intent=0.0 -> DREAM mode (x0.5)
    # We want IDLE mode for this test, so we need intent > 0.1
    engine.adapt(0.2, 0.0)
    assert abs(engine.body.target_rpm - 60.0) < 20.0 # Tolerance for breathing

def test_high_intensity_focus(engine):
    """Scenario: User asks a complex question (High Intent)."""
    # Intent 0.8 (High), Stress 0.1 (Low)
    engine.adapt(intent_intensity=0.8, stress_level=0.1)

    # Should shift to FOCUS (x2.5)
    assert engine.current_gear.name == "Focus"

    # RPM Calculation: 60 * 2.5 * BreathFactor
    # BreathFactor = (1 + 0.4) / (1 + 0.05) ~= 1.33
    # Target ~= 60 * 2.5 * 1.33 ~= 200

    assert engine.body.target_rpm > 100.0
    assert engine.mind.target_rpm > 100.0

def test_panic_mode(engine):
    """Scenario: System Overload (High Stress)."""
    # Intent 0.5 (Mid), Stress 0.9 (Critical)
    engine.adapt(intent_intensity=0.5, stress_level=0.9)

    # Should shift to PANIC (x5.0) - Fight or Flight
    # Note: Panic usually means extremely high RPM (hyper-vigilance) or freeze.
    # Our gear defines PANIC as x5.0 (Adrenaline).

    assert engine.current_gear.name == "Panic"
    assert engine.body.target_rpm > 200.0

def test_dream_mode(engine):
    """Scenario: Sleep (Zero Intent)."""
    # Intent 0.0, Stress 0.0
    engine.adapt(intent_intensity=0.05, stress_level=0.0)

    # Should shift to DREAM (x0.5)
    assert engine.current_gear.name == "Dream"
    assert engine.body.target_rpm < 40.0 # 60 * 0.5 * ...

def test_breathing_mechanics(engine):
    """Verifies micro-adjustments within the same gear."""
    engine.shift_gear("FLOW")

    # Breath 1: Calm Flow
    engine.adapt(intent_intensity=0.5, stress_level=0.0)
    rpm_calm = engine.body.target_rpm

    # Breath 2: Intense Flow
    engine.adapt(intent_intensity=0.6, stress_level=0.0)
    rpm_intense = engine.body.target_rpm

    # Higher intent should increase RPM even in same gear
    assert rpm_intense > rpm_calm
