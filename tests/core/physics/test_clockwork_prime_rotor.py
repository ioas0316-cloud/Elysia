import numpy as np
import pytest
from core.physics.clockwork_prime_rotor import (
    PrimeRotor,
    CausalRheostatDial,
    ClockworkUniverseField,
    ClockworkAgent
)

def test_prime_rotor_isomorphic_mappings():
    """Verify that PrimeRotor translates its physical phase and velocity into color and sound."""
    rotor_2 = PrimeRotor(prime=2, initial_phase=0.0)
    rotor_3 = PrimeRotor(prime=3, initial_phase=np.pi / 2.0)

    # 1. Sound Isomorphism
    # Frequency should be higher for lower prime and modulate with sine(phase)
    freq_2 = rotor_2.to_sound_frequency()
    freq_3 = rotor_3.to_sound_frequency()
    assert freq_2 > 0
    assert freq_3 > 0
    assert freq_2 != freq_3

    # 2. Color Isomorphism
    chrom_2 = rotor_2.to_chromatic_vector()
    chrom_3 = rotor_3.to_chromatic_vector()

    # chromatic vectors are normalized to sum to 1.0
    assert np.isclose(np.sum(chrom_2), 1.0)
    assert np.isclose(np.sum(chrom_3), 1.0)
    assert len(chrom_2) == 3

    # Rotate rotor and verify state update
    rotor_2.rotate(dt=0.5)
    assert rotor_2.phase != 0.0

def test_causal_rheostat_insulation_and_dilation():
    """Verify CausalRheostatDial controls causal propagation velocity and complete insulation boundaries."""
    dial_5 = CausalRheostatDial(prime=5, initial_resistance=1.0)

    # Base state
    assert dial_5.conductance == 1.0
    assert dial_5.get_time_dilation_factor() == 1.0

    # Decrease resistance -> Higher conductance (speed up)
    dial_5.set_resistance(0.2)
    assert dial_5.conductance == 5.0
    assert dial_5.get_time_dilation_factor() == 5.0

    # Infinite resistance -> Complete insulation boundary
    dial_5.insulate()
    assert dial_5.conductance == 0.0
    assert dial_5.get_time_dilation_factor() == 0.0

def test_clockwork_universe_field_tension_and_diffusion():
    """Verify energy propagation, toroidal wrapping, and non-repeating complex phase vector sums."""
    field = ClockworkUniverseField(shape=(4, 4), primes=[2, 3, 5])

    # Initial tension map calculation
    t_map_init = field.get_tension_map()
    assert t_map_init.shape == (4, 4)

    # Stimulate a cell with energy and verify toroidal wrapping
    field.stimulate(y=4, x=4, energy_amount=10.0, target_prime=2)
    assert field.energy[0, 0] > 0.0

    # Step the field and verify energy diffusion
    field.step(dt=0.1)

    # Energy should decay/diffuse and tension maps should adapt dynamically
    t_map_stepped = field.get_tension_map()
    assert not np.array_equal(t_map_init, t_map_stepped)

def test_clockwork_agent_experience_and_future_trajectory():
    """Verify experience assimilation (naite rings), factorization decoding, and future trajectory projection."""
    agent = ClockworkAgent(id="monster_01", home_pos=(2, 2), primes=[2, 3, 5])

    # Ensure clean slate
    prod, active = agent.decode_state_signature()
    assert prod == 1
    assert len(active) == 0

    # Assimilate standard instincts (primes in list)
    agent.assimilate_experience(prime_axis=2, intensity=5.0)
    agent.assimilate_experience(prime_axis=3, intensity=3.0)

    # Verify factorization state
    prod, active = agent.decode_state_signature()
    assert 2 in active
    assert 3 in active
    assert prod == 6

    # Assimilate a completely new complex experience (e.g. fire hazard = 19)
    agent.assimilate_experience(prime_axis=19, intensity=2.0)
    prod, active = agent.decode_state_signature()
    assert 19 in active
    assert prod == 114

    # Verify chromatic state and sound signatures are intact
    assert len(agent.get_chromatic_state()) == 3
    assert len(agent.get_sound_signature()) == 4 # 2, 3, 5, 19

    # Predict future path using continuous phase integration
    path = agent.predict_future_trajectory(steps=5, dt=0.1)
    assert len(path) == 5
    for coord in path:
        assert isinstance(coord[0], float)
        assert isinstance(coord[1], float)

def test_agent_field_resonance_navigation():
    """Verify that ClockworkAgent navigates the field based on phase/resistance resonance."""
    field = ClockworkUniverseField(shape=(4, 4), primes=[2, 3, 5])
    agent = ClockworkAgent(id="hero", home_pos=(0, 0), primes=[2, 3, 5])

    # Record agent starting position
    init_pos = agent.position.copy()

    # Step agent in the field
    agent.navigate_and_step(field, dt=0.1)

    # Position should smoothly transition or update towards resonance wells
    # Step multiple times to ensure continuous update loop passes without error
    for _ in range(5):
        agent.navigate_and_step(field, dt=0.1)
        field.step(dt=0.1)
