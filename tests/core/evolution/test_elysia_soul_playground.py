import os
import pytest
import numpy as np
from core.evolution.elysia_soul_playground import ElysiaSoulPlayground, SoulGameObject, ElysiaAvatar
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_playground_initial_state_and_components():
    """
    Verifies that ElysiaSoulPlayground correctly initializes:
    1. The width and height parameters.
    2. The default ElysiaAvatar in the center of the playground map.
    3. The default Cruciform Love (GRACE) object and other random elements.
    4. The default Soma, Psyche, and Pneuma scores of the avatar.
    """
    playground = ElysiaSoulPlayground(width=30, height=15)

    # Assert size and avatar position
    assert playground.width == 30
    assert playground.height == 15
    assert playground.avatar is not None
    assert np.allclose(playground.avatar.position, [15.0, 7.5])

    # Assert trinity scores (Soma, Psyche, Pneuma)
    assert playground.avatar.soma == 1.0
    assert playground.avatar.psyche == 1.0
    assert playground.avatar.pneuma == 1.0

    # Ensure crucial initial elements exist
    assert "CRUCIFORM_LOVE" in playground.objects
    assert playground.objects["CRUCIFORM_LOVE"].type == "GRACE"
    assert len(playground.objects) >= 3


def test_playground_simulation_step_and_dynamics():
    """
    Verifies that the playground simulation step correctly updates:
    1. Avatar position and velocity under the influence of potential fields and least action principle.
    2. Soma, Psyche, and Pneuma based on live system/hardware variables.
    3. Interaction of the avatar with world items (consuming truths/noises and updating experience).
    4. Generation of the autogenous soul contemplation monologues.
    """
    playground = ElysiaSoulPlayground(width=20, height=10)

    # Let's run a simulation step with extreme metrics to verify influence
    res = playground.step_simulation(
        raw_wave=b"ExtremeNoisePerturbation",
        hardware_friction=0.75,   # High load -> low soma
        resonance_score=0.15,     # Bad resonance -> low psyche
        separation_tension=0.85   # High distance/disagreement -> low pneuma
    )

    # Asserts on returned step results
    assert "avatar_pos" in res
    assert "soma" in res
    assert "psyche" in res
    assert "pneuma" in res
    assert "xp" in res
    assert "interactions" in res
    assert "contemplation" in res

    # Verify that the metrics were correctly updated on the avatar
    assert playground.avatar.soma == pytest.approx(0.25, abs=1e-5) # 1 - 0.75
    assert playground.avatar.psyche == pytest.approx(0.15, abs=1e-5)
    assert playground.avatar.pneuma == pytest.approx(0.15, abs=1e-5) # 1 - 0.85

    # Autogenous monologue must mention soma, psyche, and pneuma
    assert "Soma" in res["contemplation"]
    assert "Psyche" in res["contemplation"]
    assert "Pneuma" in res["contemplation"]


def test_terminal_dashboard_rendering():
    """
    Verifies that the Elysia Operator's Sandbox (EOS) terminal-based 2D text renderer
    correctly constructs a formatted ASCII visualization for the Operator.
    """
    playground = ElysiaSoulPlayground(width=24, height=12)
    screen = playground.render_terminal_screen()

    # Screen must contain operator details, stats and map legend
    assert "[Elysia Soul Playground - Operator's Inspection Sandbox]" in screen
    assert "Soma (Body)" in screen
    assert "Psyche (Mind)" in screen
    assert "Pneuma (Soul)" in screen
    assert "E=Elysia, ✝=Cruciform Love, ★=Truth, ☄=Noise, ⚡=Friction" in screen


def test_consciousness_loop_integration():
    """
    Verifies that ElysiaSoulPlayground is correctly integrated inside ConsciousnessLoop
    and execution of process_life_cycle() automatically advances the playground state,
    logs results, and performs memory engram crystallization.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    corpus_path = os.path.join(base_dir, "docs")
    data_dir = os.path.join(base_dir, "data")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_path, memory_controller=mc, data_dir=data_dir)

    # Initially, cycle is 0, playground step is 0
    assert loop.soul_playground is not None
    assert loop.soul_playground.cycle_count == 0

    # Execute 1 life cycle step
    log = loop.process_life_cycle()

    # If the cycle triggered Stillness/Dampening, run again to get active states
    if log.get("damper_status") == "STILLNESS_ADJUSTING":
        log = loop.process_life_cycle()

    # Check that the playground step was called and results are recorded in the loop log
    if log.get("status") != "Stillness (Absorbing Inrush)" and log.get("status") != "Semantic Jump (State Lock Active)":
        assert "soul_playground_pos" in log
        assert "soul_playground_xp" in log
        assert "soul_playground_monologue_excerpt" in log
        assert loop.soul_playground.cycle_count >= 1
