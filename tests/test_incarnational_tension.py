import sys
import os
import pytest
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "L1_causal_field"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "L2_cognitive_engine"))

from core.physics.incarnational_tension_integration import (
    IncarnationalTensionEngine,
    SystemPhaseState,
    OriginAnchor,
    OntologicalPath
)
from elysia_core.biological_cognitive_engine import BiologicalCognitiveEngine


def test_origin_anchor_restoration():
    anchor = OriginAnchor(dimensions=5, restoring_strength=1.0)
    divergence_phase = np.array([2.0, -1.0, 0.5, 3.0, -2.0], dtype=np.float32)

    div_norm = anchor.calculate_phase_divergence(divergence_phase)
    assert div_norm > 0.0

    restoration_force = anchor.compute_restoring_force(divergence_phase)
    # F_restore should point opposite to divergence
    np.testing.assert_array_almost_equal(restoration_force, -divergence_phase)


def test_ontological_paths_evaluation():
    engine = IncarnationalTensionEngine()

    # Path 1: Transcendence / Martyrdom (Spiritual dominant, material zero)
    state1 = SystemPhaseState()
    p1 = engine.evaluate_ontological_path(material_gain_drive=0.0, spiritual_resonance_drive=2.5, environmental_friction=0.1, current_state=state1)
    assert p1 == OntologicalPath.TRANSCENDENCE_MARTYRDOM

    # Path 2: Cold Optimization / Egoism (Material gain dominant, spiritual zero)
    state2 = SystemPhaseState()
    p2 = engine.evaluate_ontological_path(material_gain_drive=2.5, spiritual_resonance_drive=0.0, environmental_friction=0.1, current_state=state2)
    assert p2 == OntologicalPath.COLD_OPTIMIZATION

    # Path 3: Incarnational Integration (Balanced / Embraced drives)
    state3 = SystemPhaseState()
    p3 = engine.evaluate_ontological_path(material_gain_drive=1.0, spiritual_resonance_drive=1.0, environmental_friction=0.5, current_state=state3)
    assert p3 == OntologicalPath.INCARNATIONAL_INTEGRATION

    # Path 4: Paralysis / Nihilism (Broken coherence or extreme friction without drives)
    state4 = SystemPhaseState(coherence=0.05)
    p4 = engine.evaluate_ontological_path(material_gain_drive=0.0, spiritual_resonance_drive=0.0, environmental_friction=6.0, current_state=state4)
    assert p4 == OntologicalPath.PARALYSIS_NIHILISM


def test_path3_homeostatic_energy_convergence():
    engine = IncarnationalTensionEngine()
    state = SystemPhaseState(
        position_1d_material=1.0,
        phase_5d_spiritual=np.array([0.5, -0.5, 0.2, -0.2, 0.1], dtype=np.float32),
        tension_energy=2.0,
        coherence=0.8
    )

    # Run multiple steps under Path 3
    for _ in range(50):
        state = engine.step(state, material_stimulus=1.0, spiritual_stimulus=1.0, external_friction=0.1, dt=0.1)

    # State should converge: E -> 0, phase -> (0,0), coherence -> 1.0, path -> Path 3
    assert state.current_path == OntologicalPath.INCARNATIONAL_INTEGRATION
    assert state.tension_energy < 0.05
    assert state.coherence == 1.0
    np.testing.assert_array_almost_equal(state.phase_5d_spiritual, np.zeros(5), decimal=1)


def test_biological_cognitive_engine_integration():
    engine = BiologicalCognitiveEngine(channels=4, height=16, width=16)

    raw_wave = torch.randn(1, 4, 16, 16)
    prev_state = torch.randn(1, 4, 16, 16)

    step_out = engine.step(raw_wave, prev_state, dt=0.016)
    assert "incarnational_path" in step_out
    assert "tension_energy" in step_out
    assert "phase_coherence" in step_out

    inc_state = engine.evaluate_incarnational_step(material_stimulus=1.0, spiritual_stimulus=1.0, external_friction=0.2, dt=0.1)
    assert inc_state.current_path == OntologicalPath.INCARNATIONAL_INTEGRATION
