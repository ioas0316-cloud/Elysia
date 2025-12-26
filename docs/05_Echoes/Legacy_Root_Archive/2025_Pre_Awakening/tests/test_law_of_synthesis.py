"""
Tests for the Law of Synthesis (Attribute Combination)
"""
import pytest
from Core.Laws.law_of_synthesis import get_synthesis_engine, Attribute, ConceptDefinition

def test_synthesis_initialization():
    synth = get_synthesis_engine()
    assert synth is not None
    assert synth.knowledge_base is not None

def test_observation_parsing():
    synth = get_synthesis_engine()
    synth.knowledge_base.clear() # Reset state

    # Test "is a" (Class)
    synth.observe("Apple is a fruit")
    assert "Apple" in synth.knowledge_base
    attrs = synth.knowledge_base["Apple"].attributes
    assert len(attrs) == 1
    assert attrs[0].category == "class"
    assert attrs[0].name == "fruit"

    # Test "is" (Color/Property)
    synth.observe("Apple is red")
    attrs = synth.knowledge_base["Apple"].attributes
    assert len(attrs) == 2
    # Find the 'red' attribute
    red_attr = next((a for a in attrs if a.name == "red"), None)
    assert red_attr is not None
    assert red_attr.category == "color"

def test_definition_synthesis():
    synth = get_synthesis_engine()
    synth.knowledge_base.clear()

    synth.observe("Ball is round")
    synth.observe("Ball is red")
    synth.observe("Ball is a toy")

    definition = synth.derive_definition("Ball")
    # Expected: "Ball is a round, red toy." OR "Ball is a red, round toy." (order may vary slightly based on insertion but logic tries to sort)
    assert "Ball" in definition
    assert "round" in definition
    assert "red" in definition
    assert "toy" in definition
    assert "is a" in definition

def test_unknown_concept():
    synth = get_synthesis_engine()
    result = synth.derive_definition("Ghost")
    assert "no knowledge" in result
