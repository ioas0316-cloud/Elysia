import pytest
import numpy as np
from synaptic_architecture.machine_internal_world import MachineInternalWorld
from synaptic_architecture.scale_lens_engine import ScaleLensEngine, EmergentMacroAxiom
from synaptic_architecture.structural_valence import StructuralValence, StructuralCategory
from synaptic_architecture.language_protocol_bridge import LanguageProtocolBridge, IsomorphicGroundingPair

def test_machine_internal_world_primitive_operators():
    world = MachineInternalWorld(state_dim=2, base_reluctance=0.5)
    assert np.allclose(world.state, np.zeros(2))

    # Primitive Operator 1: Push against resistance
    drive = np.array([1.0, -1.0])
    res = world.push_against_resistance(drive, dt=0.1)

    assert "state" in res
    assert "velocity" in res
    assert res["instant_friction"] > 0.0
    assert res["impedance"] > 0.0

    # Primitive Operator 2: Tune frequency
    freq_res = world.tune_frequency(delta_freq=0.5, dt=0.05)
    assert freq_res["frequency"] == 1.5

    # Primitive Operator 3: Probe friction
    probe = world.probe_friction()
    assert probe["accumulated_friction"] > 0.0

def test_scale_lens_coarse_graining_and_top_down_constraint():
    world = MachineInternalWorld(state_dim=2, base_reluctance=0.5)
    lens = ScaleLensEngine(world, damping_factor=0.8, window_size=5)

    # Drive with high force and sharp trajectory angle changes to induce impedance & curvature
    drives = [np.array([5.0, 0.0]), np.array([0.0, 5.0]), np.array([-5.0, 0.0]), np.array([0.0, -5.0])]

    for d in drives:
        micro_res = world.push_against_resistance(d, dt=0.1)
        lens_res = lens.observe_and_coarse_grain(micro_res)

    assert lens_res["damped_friction"] > 0.0
    assert lens_res["damped_impedance"] > 0.0
    assert len(lens.emergent_axioms) > 0

    # Check top-down constraint effect on reluctance field
    assert np.all(world.reluctance_field > 0.5)

def test_structural_valence_and_category_differentiation():
    valence_engine = StructuralValence(initial_dim=2, differentiation_threshold=1.5)

    # Test positive flow valence
    flow_res = valence_engine.evaluate_valence(
        current_state=np.array([0.1, 0.1]),
        current_velocity=np.array([1.0, 1.0]),
        damped_friction=0.2,
        impedance=0.3
    )
    assert flow_res["valence"] > 0.0
    assert flow_res["state_label"] == "Flow / Resonance"

    # Test high friction triggering category differentiation
    friction_res = valence_engine.evaluate_valence(
        current_state=np.array([3.0, 3.0]),
        current_velocity=np.array([0.1, 0.1]),
        damped_friction=2.0,
        impedance=2.0
    )
    assert friction_res["valence"] < 0.0
    assert friction_res["state_label"] == "Friction / Noise"
    assert friction_res["category_differentiated"] is True
    assert len(valence_engine.categories) > 1

def test_language_protocol_bridge_grounding():
    world = MachineInternalWorld(state_dim=2)
    lens = ScaleLensEngine(world)
    valence = StructuralValence(initial_dim=2)
    bridge = LanguageProtocolBridge(world, lens, valence)

    # Force creation of an emergent macro axiom
    lens.emergent_axioms.append(
        EmergentMacroAxiom("MacroConstraint_ImpedanceCap_1", curvature_threshold=0.785, reluctance_modifier=1.25, boundary_cap=4.5)
    )
    lens.damped_impedance = 1.5

    groundings = bridge.search_isomorphic_grounding()
    assert len(groundings) > 0
    assert groundings[0].is_grounded is True

    translation = bridge.translate_internal_state_to_symbol()
    assert len(translation["grounded_symbols"]) > 0
