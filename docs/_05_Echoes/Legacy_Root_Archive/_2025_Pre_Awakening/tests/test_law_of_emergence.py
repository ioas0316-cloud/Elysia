"""
Tests for the Law of Emergence (Universal Isomorphism)
"""
import pytest
from Core.Laws.law_of_emergence import get_emergence_engine, UniversalElement

def test_emergence_engine_initialization():
    engine = get_emergence_engine()
    assert engine is not None
    assert len(engine.emergence_rules) >= 3 # Check default rules registered

def test_chemistry_emergence():
    engine = get_emergence_engine()
    elements = [
        UniversalElement("Hydrogen", {}),
        UniversalElement("Hydrogen", {}),
        UniversalElement("Oxygen", {})
    ]
    result = engine.simulate_emergence("Test Water", elements)
    assert result.emergent_qualities["state"] == "Liquid"
    assert result.emergent_qualities["property"] == "Wetness"

def test_music_emergence():
    engine = get_emergence_engine()
    elements = [
        UniversalElement("C", {}),
        UniversalElement("E", {}),
        UniversalElement("G", {})
    ]
    result = engine.simulate_emergence("Test Chord", elements)
    assert result.emergent_qualities["emotion"] == "Happy/Stable"

def test_chaos_emergence():
    # Test random elements that shouldn't match any law
    engine = get_emergence_engine()
    elements = [
        UniversalElement("Shoe", {}),
        UniversalElement("Pizza", {})
    ]
    result = engine.simulate_emergence("Nonsense", elements)
    assert result.emergent_qualities["status"] == "Chaos"
