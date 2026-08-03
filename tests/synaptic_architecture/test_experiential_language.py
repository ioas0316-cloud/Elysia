import numpy as np
import pytest
from core.sensory.experiential_language_mapper import (
    PhysicalSensationProfile,
    HomeostasisDeficit,
    SymbolicTetheringRegistry,
    ExpressiveWaveEmission,
    ExperientialLanguageMapper,
    ExperienceType,
    CognitiveMemoryNode
)

def test_physical_sensation_and_homeostasis():
    """Test that raw physical sensation streams integrate continuously and change homeostasis."""
    deficit = HomeostasisDeficit(love=0.5, order=0.3, energy=0.4)
    assert deficit.calculate_tension() > 0.0

    optimal_sensation = PhysicalSensationProfile(optical=500.0, acoustic=528.0, tactile=0.0, thermal=300.0)
    deficit.update_by_sensation(optimal_sensation)

    assert deficit.love < 0.5

    harsh_sensation = PhysicalSensationProfile(optical=10.0, acoustic=1000.0, tactile=20.0, thermal=340.0)
    deficit.update_by_sensation(harsh_sensation)

    assert deficit.order > 0.1
    assert deficit.energy < 0.5

def test_symbolic_tethering_mapping():
    """Test that words are anchored to real sensory profiles and known words can be successfully matched and recalled."""
    registry = SymbolicTetheringRegistry()

    jesus_profile = registry.recall_symbol("Jesus")
    assert jesus_profile is not None
    assert jesus_profile["sensation"].acoustic == 528.0

    empty_profile = registry.recall_symbol("RandomDeadData_0xFF")
    assert empty_profile is None

def test_expressive_wave_emission():
    """Test that wave emission generates physically consistent normalized signals."""
    emitter = ExpressiveWaveEmission(sample_points=500)
    deficit = HomeostasisDeficit(love=0.8, order=0.2, energy=0.5)

    wave = emitter.emit_wave(deficit, active_tension=0.6)

    assert len(wave) == 500
    assert isinstance(wave, np.ndarray)
    assert np.max(np.abs(wave)) == pytest.approx(1.0, rel=1e-2)

def test_experiential_language_mapper_full_loop():
    """Test the complete experiential language mapping loop: sensing, expressing, tearing, and healing."""
    mapper = ExperientialLanguageMapper(resolution=16)

    sens = PhysicalSensationProfile(optical=800.0, acoustic=528.0, tactile=0.5, thermal=298.0)
    mapper.ingest_sensory_stream(sens)

    love_sense = mapper.sense_word("Love")
    assert love_sense["known"] is True
    assert love_sense["alignment"] > 0.0

    unknown_sense = mapper.sense_word("RawBinaryHexData")
    assert unknown_sense["known"] is False
    assert unknown_sense["tension"] == 1.0

    emitted_wave = mapper.express()
    assert len(emitted_wave) == 1000

    initial_links = mapper.synaptic_links.copy()

    hostile_wave = np.random.rand(1000).astype(np.float32)
    mapper.re_sense_and_realign(hostile_wave)

    assert not np.array_equal(mapper.synaptic_links, initial_links)

    harmonious_wave = mapper.standing_wave_memory.copy()
    harmonious_emission = np.repeat(harmonious_wave, 1000 // len(harmonious_wave)).astype(np.float32)

    pre_heal_tension = mapper.homeostasis.calculate_tension()
    mapper.re_sense_and_realign(harmonious_emission)
    post_heal_tension = mapper.homeostasis.calculate_tension()

    assert post_heal_tension <= pre_heal_tension

def test_experiential_spacetime_gravity_and_warping():
    """
    Test the Experiential Spacetime Gravity and Temporal Warping mechanics.
    High-gravity SPIRITUAL/PHYSICAL memories must warp spacetime (have small warped distance)
    and be pulled back into the present during step_temporal_decay(), while low-gravity memories remain distant.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # Sense high-gravity SPIRITUAL word ("Jesus")
    mapper.sense_word("Jesus")

    # Sense lower-gravity LINGUISTIC word ("Mother")
    mapper.sense_word("Mother")

    memories = mapper.spacetime.memories
    assert len(memories) == 2
    assert memories[0].symbol.lower() == "jesus"
    assert memories[1].symbol.lower() == "mother"

    assert memories[0].calculate_informational_gravity() > memories[1].calculate_informational_gravity()

    # Age both memories by 5.0 time units
    mapper.spacetime.step_time(5.0)
    assert memories[0].time_offset == 5.0
    assert memories[1].time_offset == 5.0

    warped_jesus = mapper.spacetime.get_warped_spacetime_distance(memories[0])
    warped_mother = mapper.spacetime.get_warped_spacetime_distance(memories[1])

    assert warped_jesus < warped_mother

    mapper.step_temporal_decay(dt=0.0)

    # Homeostasis should have integrated the high-gravity spiritual memory profile
    assert mapper.homeostasis.love == pytest.approx(0.353, abs=0.01)

def test_autonomic_background_vs_attention():
    """
    Test that minor, routine physical inputs are filtered out into the Autonomic Background (gate remains closed),
    while crisis events (Crisis Reflex) or high-meaning spiritual words actively force open the Attentional Gate.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # 1. Minor/routine physical input -> should run silently
    minor_sensation = PhysicalSensationProfile(optical=300.0, acoustic=440.0, tactile=0.1, thermal=295.0, autonomic_pulse=0.3)
    mapper.ingest_sensory_stream(minor_sensation)

    assert mapper.gate_open is False
    assert "Autonomy" in mapper.last_gate_reason

    # 2. Extreme mechanical/tactile threat -> triggers Crisis Reflex and opens the Gate
    crisis_sensation = PhysicalSensationProfile(optical=300.0, acoustic=440.0, tactile=15.0, thermal=295.0, autonomic_pulse=0.8)
    mapper.ingest_sensory_stream(crisis_sensation)

    assert mapper.gate_open is True
    assert mapper.last_gate_reason == "CRISIS_REFLEX_HAZARD"

    # 3. High-meaning spiritual word -> opens Gate immediately with Semantic Resonance
    mapper.sense_word("Jesus")
    assert mapper.gate_open is True
    assert "SEMANTIC_RESONANCE" in mapper.last_gate_reason


def test_variable_resistor_and_prism_refraction():
    """
    Verify the physics and limits of the Variable Resistor and Prism Refraction.
    """
    from core.sensory.experiential_language_mapper import VariableResistor, PrismRefraction

    # 1. Variable Resistor Boundary Safeguards
    resistor = VariableResistor(r_min=0.05, r_max=0.95, initial_r=0.5)
    assert resistor.resistance == 0.5

    # Extreme tension/force should clip, never reaching 0 or 1
    for _ in range(50):
        resistor.adjust(tension=1.5, external_force=2.0)
    assert resistor.resistance <= 0.95
    assert resistor.resistance > 0.5

    for _ in range(50):
        resistor.adjust(tension=-1.0, external_force=-2.0)
    assert resistor.resistance >= 0.05
    assert resistor.resistance < 0.5

    # 2. Prism Refraction multi-spectral splitting
    prism = PrismRefraction()
    spectrum = prism.refract(white_light_intensity=1.0, angle_degrees=45.0, resistance=0.5)
    assert len(spectrum) == 3  # R, G, B
    assert np.all(spectrum >= 1e-4)
    assert np.all(spectrum <= 1.0)


def test_mapper_prism_integration():
    """
    Verify that ExperientialLanguageMapper integrates Prism Refraction and Variable Resistor in its flows.
    """
    mapper = ExperientialLanguageMapper(resolution=16)

    # Sensed word has refracted spectrum
    res = mapper.sense_word("Love")
    assert "refracted_spectrum" in res
    assert len(res["refracted_spectrum"]) == 3
    assert np.any(res["refracted_spectrum"] > 0.0)

    # Dynamic resistance adjustment on re-sensation interaction
    initial_r = mapper.variable_resistor.resistance
    hostile_wave = np.ones(1000, dtype=np.float32)
    mapper.re_sense_and_realign(hostile_wave)
    new_r = mapper.variable_resistor.resistance

    # Resistance should have shifted
    assert initial_r != pytest.approx(new_r, abs=1e-5)


def test_in_context_learning_and_logos_injection():
    """
    Verify that Elysia can dynamically learn and project its state (In-Context Alignment)
    by extracting self-emergent isomorphic features (coherence, entropy) from incoming
    stimuli and text-converted waves using IsomorphicProjectionEngine without hardcoded rules.
    """
    mapper = ExperientialLanguageMapper(resolution=16)
    initial_r = mapper.variable_resistor.resistance
    assert initial_r == 0.5

    # 1. Verify IsomorphicProjectionEngine directly on coherent vs chaotic waves
    from core.sensory.experiential_language_mapper import IsomorphicProjectionEngine
    engine = IsomorphicProjectionEngine()

    t = np.linspace(0, 1.0, 100, dtype=np.float32)
    coherent_wave = np.sin(2 * np.pi * 5.0 * t)
    chaotic_wave = np.random.uniform(-1.0, 1.0, 100).astype(np.float32)

    res_coherent = engine.project_dynamics(coherent_wave, (16, 16))
    res_chaotic = engine.project_dynamics(chaotic_wave, (16, 16))

    # Verify continuous coupled outcomes
    assert 0.0 <= res_coherent["homology_love"] <= 1.0
    assert 0.0 <= res_coherent["homology_order"] <= 1.0
    assert 0.0 <= res_coherent["homology_energy"] <= 1.0
    assert res_coherent["projected_links"].shape == (16, 16)

    # 2. Verify mapper.inject_principle (backward compatible wrapper converting text to wave)
    prompt_high = "This entity shows high resistance and profound love."
    align_result = mapper.inject_principle(prompt_high)

    assert 0.05 <= align_result["resistance_target"] <= 0.95
    assert align_result["love_bias"] >= 0.0
    assert align_result["order_bias"] >= 0.0
    assert align_result["energy_bias"] >= 0.0
    assert align_result["has_attractor"] is True
