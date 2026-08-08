import pytest
import numpy as np
from core.lens.language_refractor import LanguageRefractor
from core.physics.topological_os_engine import TopologicalOSEngine

def test_language_refraction_empty():
    """Verify refraction handles empty or whitespace text gracefully."""
    refractor = LanguageRefractor(grid_shape=(16, 16))
    params = refractor.refract("  ")
    assert params["intent_type"] == "vacuum"
    assert params["mass"] == 0.1
    assert params["gradient"] == 1.0
    assert params["ignorance_charge"] == 1.0

def test_language_refraction_urgent():
    """Verify refraction of an urgent/error command mapping."""
    refractor = LanguageRefractor(grid_shape=(16, 16))
    params = refractor.refract("이 버그 좀 빨리 고쳐줘!")

    assert params["intent_type"] == "high_gradient_well"
    assert params["mass"] >= 25.0
    assert params["gradient"] >= 8.0
    assert params["thermal_heating"] == 0.0
    assert params["ignorance_charge"] <= 0.5  # Urgent commands have clear, narrow maps (low ignorance)
    assert params["damping_multiplier"] == 1.0
    assert params["locus_range_expansion"] == 1.0
    assert 0 <= params["target_y"] < 16
    assert 0 <= params["target_x"] < 16
    assert -1.0 <= params["wave_signature"] <= 1.0

def test_language_refraction_casual():
    """Verify refraction of a casual speculation text."""
    refractor = LanguageRefractor(grid_shape=(16, 16))
    params = refractor.refract("오늘 그냥 문득 든 생각인데...")

    assert params["intent_type"] == "brownian_perturbation"
    assert params["mass"] <= 5.0
    assert params["gradient"] <= 3.0
    assert params["ignorance_charge"] > 0.5  # Casual speculation has high ignorance
    assert params["locus_range_expansion"] > 1.0  # Antenna expands
    assert params["damping_multiplier"] < 1.0  # Damping decreases
    assert "거울" in params["metacognitive_reflection"]

def test_language_refraction_raw_tension_ingestion():
    """Verify Layer 1: Tension Vector Ingestion handles non-UTF8 bytes gracefully."""
    refractor = LanguageRefractor(grid_shape=(16, 16))

    # Intentionally broken unicode bytes
    broken_bytes = b"Hello \xff\xfe World \x90"
    params = refractor.refract(broken_bytes)

    # Should not crash, and should yield high ignorance charge / high tension
    assert params["ignorance_charge"] > 0.5
    assert params["locus_range_expansion"] > 1.0

def test_integration_with_topological_os():
    """Verify that injecting refracted parameters into TopologicalOSEngine behaves physically correct."""
    grid_shape = (10, 10)
    # Set initial temperature to 0 to avoid random thermal noise addition during strict dissipation verification
    engine = TopologicalOSEngine(grid_shape=grid_shape, initial_temp=0.0)
    refractor = LanguageRefractor(grid_shape=grid_shape)

    # Urgent stimulus
    urgent_text = "당장 에러 고쳐줘"
    params = refractor.refract(urgent_text)

    # Apply refracted parameters
    engine.inject_impulse(
        y=params["target_y"],
        x=params["target_x"],
        magnitude=params["mass"],
        importance=params["gradient"],
        wave_signature=params["wave_signature"]
    )
    # Apply thermal heating (it's 0.0 for urgent, but let's be explicit)
    engine.temperature += params["thermal_heating"]

    stimulated_state = engine.get_state()
    assert stimulated_state["energy"][params["target_y"]][params["target_x"]] >= params["mass"]

    # Run Langevin relaxation for some steps with cooling rate to let energy dissipate
    for _ in range(30):
        engine.step(0.1)

    final_state = engine.get_state()

    # Evaluate cognitive feedback
    feedback = refractor.evaluate_cognitive_feedback(stimulated_state, final_state, steps_taken=30)
    assert feedback["initial_energy"] > 0.0
    assert feedback["final_energy"] < feedback["initial_energy"]
    # Check if the potential successfully relaxed back towards ground state
    assert feedback["final_potential"] < feedback["initial_potential"]
