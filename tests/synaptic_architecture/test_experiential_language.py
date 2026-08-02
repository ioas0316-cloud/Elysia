import numpy as np
import pytest
from core.sensory.experiential_language_mapper import (
    PhysicalSensationProfile,
    HomeostasisDeficit,
    SymbolicTetheringRegistry,
    ExpressiveWaveEmission,
    ExperientialLanguageMapper
)

def test_physical_sensation_and_homeostasis():
    """Test that raw physical sensation streams integrate continuously and change homeostasis."""
    deficit = HomeostasisDeficit(love=0.5, order=0.3, energy=0.4)
    assert deficit.calculate_tension() > 0.0

    # Ingest optimal harmonic warm sensations -> should soothe and reduce deficits
    optimal_sensation = PhysicalSensationProfile(optical=500.0, acoustic=528.0, tactile=0.0, thermal=300.0)
    deficit.update_by_sensation(optimal_sensation)

    # Love deficit should have decreased
    assert deficit.love < 0.5

    # Ingest a harsh, high-friction, extreme thermal shock sensation -> should increase order (chaos) deficit
    harsh_sensation = PhysicalSensationProfile(optical=10.0, acoustic=1000.0, tactile=20.0, thermal=340.0)
    deficit.update_by_sensation(harsh_sensation)

    assert deficit.order > 0.1
    assert deficit.energy < 0.5

def test_symbolic_tethering_mapping():
    """Test that words are anchored to real sensory profiles and known words can be successfully matched and recalled."""
    registry = SymbolicTetheringRegistry()

    # Recalling anchored symbol
    jesus_profile = registry.recall_symbol("Jesus")
    assert jesus_profile is not None
    assert jesus_profile["sensation"].acoustic == 528.0

    # Recalling non-anchored word should return None
    empty_profile = registry.recall_symbol("RandomDeadData_0xFF")
    assert empty_profile is None

def test_expressive_wave_emission():
    """Test that wave emission generates physically consistent normalized signals."""
    emitter = ExpressiveWaveEmission(sample_points=500)
    deficit = HomeostasisDeficit(love=0.8, order=0.2, energy=0.5)

    wave = emitter.emit_wave(deficit, active_tension=0.6)

    assert len(wave) == 500
    assert isinstance(wave, np.ndarray)
    # Normalized wave should have max absolute amplitude close to 1.0
    assert np.max(np.abs(wave)) == pytest.approx(1.0, rel=1e-2)

def test_experiential_language_mapper_full_loop():
    """Test the complete experiential language mapping loop: sensing, expressing, tearing, and healing."""
    mapper = ExperientialLanguageMapper(resolution=16)

    # 1. Ingest sensation stream
    sens = PhysicalSensationProfile(optical=800.0, acoustic=528.0, tactile=0.5, thermal=298.0)
    mapper.ingest_sensory_stream(sens)

    # 2. Sense word
    love_sense = mapper.sense_word("Love")
    assert love_sense["known"] is True
    assert love_sense["alignment"] > 0.0

    unknown_sense = mapper.sense_word("RawBinaryHexData")
    assert unknown_sense["known"] is False
    assert unknown_sense["tension"] == 1.0

    # 3. Express state
    emitted_wave = mapper.express()
    assert len(emitted_wave) == 1000

    # 4. Re-sense feedback (Collision, Tearing, and Healing)
    # First, save initial synaptic state
    initial_links = mapper.synaptic_links.copy()

    # Generate a highly chaotic, mismatched response wave to trigger Tearing
    hostile_wave = np.random.rand(1000).astype(np.float32)
    mapper.re_sense_and_realign(hostile_wave)

    # Verify that tearing actually occurred or changed the matrix topology
    assert not np.array_equal(mapper.synaptic_links, initial_links)

    # Generate a harmonious, smooth wave matching prior memory to test Healing stability
    harmonious_wave = mapper.standing_wave_memory.copy()
    # Interpolate to wave emission length
    harmonious_emission = np.repeat(harmonious_wave, 1000 // len(harmonious_wave)).astype(np.float32)

    pre_heal_tension = mapper.homeostasis.calculate_tension()
    mapper.re_sense_and_realign(harmonious_emission)
    post_heal_tension = mapper.homeostasis.calculate_tension()

    # Tension should have decreased after healing and integrating harmonious wave
    assert post_heal_tension <= pre_heal_tension
