"""
test_digital_universe_active_intervention.py
============================================
Unit tests for DigitalUniverseActiveInterventionEngine.
Verifies micro-causal physical inputs, 3x3x3 fractal phase grid coupling,
and macro semantic mass / causal gravity alignment.
"""

import numpy as np
import pytest
from core.embodied.digital_universe_active_intervention_engine import (
    DigitalUniverseActiveInterventionEngine,
    MicroCausalPhysicsFrame,
)


def test_digital_universe_active_intervention_basic_flow():
    engine = DigitalUniverseActiveInterventionEngine(
        reach_radius=3.0, boundary_rigidity=0.8, causal_gravity_constant=0.75
    )

    frame = MicroCausalPhysicsFrame(
        position=np.array([1.0, 1.0, 1.0]),
        raycast_sight_vector=np.array([0.0, 0.0, 1.0]),
        contact_friction=0.4,
        collision_normal_force=0.2,
        action_reach_limit=3.0,
        frame_temporal_dt=0.016,
        bit_entropy_signature=0.05,
    )

    action = np.array([0.5, 0.0, 1.0])
    result = engine.process_micro_interaction(frame, action)

    assert result["net_intervention_power"] >= 0.0
    assert 0.0 <= result["3x3x3_coherence_score"] <= 1.0
    assert result["semantic_mass"] > 1.0
    assert "narrative" in result
    assert len(engine.intervention_history) == 1


def test_physical_friction_resistance_and_semantic_mass_accumulation():
    engine = DigitalUniverseActiveInterventionEngine(reach_radius=5.0)

    # Frame with high friction/resistance
    high_friction_frame = MicroCausalPhysicsFrame(
        position=np.array([1.0, 1.0, 1.0]),
        raycast_sight_vector=np.array([1.0, 0.0, 0.0]),
        contact_friction=1.5,
        collision_normal_force=2.0,
        bit_entropy_signature=0.3,
    )

    action = np.array([1.0, 1.0, 1.0])
    initial_mass = engine.semantic_mass

    for _ in range(5):
        engine.process_micro_interaction(high_friction_frame, action)

    # High friction & active intervention should increase semantic mass
    assert engine.semantic_mass > initial_mass
    assert len(engine.intervention_history) == 5


def test_out_of_reach_attenuation():
    engine = DigitalUniverseActiveInterventionEngine(reach_radius=2.0)

    # Target far out of reach
    distant_frame = MicroCausalPhysicsFrame(
        position=np.array([10.0, 10.0, 10.0]),
        raycast_sight_vector=np.array([1.0, 0.0, 0.0]),
        contact_friction=0.1,
    )

    action = np.array([1.0, 0.0, 0.0])
    result = engine.process_micro_interaction(distant_frame, action)

    # Influence should be significantly attenuated due to reach boundary
    assert result["effective_reach_influence"] < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
