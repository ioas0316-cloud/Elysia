"""
Tests for the Law of Alchemy (Transmutation of Causality)
"""
import pytest
from Core.Laws.law_of_alchemy import get_alchemy_engine, NarrativeEvent, TensionLevel, Archetype

def test_alchemy_initialization():
    alchemy = get_alchemy_engine()
    assert alchemy is not None
    assert alchemy.known_archetypes == {}

def test_event_abstraction_logic():
    alchemy = get_alchemy_engine()

    # Test: Negligence
    event_negligence = NarrativeEvent("The strong Hare decides to sleep", TensionLevel.RELAXATION)
    node = alchemy._abstract_event(event_negligence)
    assert node.function_name == "Negligence of Power"
    assert node.abstract_principle == "Strength creates Arrogance"
    assert "RELAXATION" in node.tension_change

    # Test: Triumph
    event_win = NarrativeEvent("The Tortoise passes and wins the race", TensionLevel.CLIMAX)
    node = alchemy._abstract_event(event_win)
    assert node.function_name == "Triumph of the Persistent"
    assert "Paradigm Shift" in node.tension_change

    # Test: Demonstration
    event_run = NarrativeEvent("The Hare runs fast leaving dust", TensionLevel.BUILDUP)
    node = alchemy._abstract_event(event_run)
    assert node.function_name == "Demonstration of Power"

    # Test: Persistence
    event_crawl = NarrativeEvent("The Tortoise keeps crawling steadily", TensionLevel.BUILDUP)
    node = alchemy._abstract_event(event_crawl)
    assert node.function_name == "Persistence of Weakness"

def test_full_extraction_flow():
    alchemy = get_alchemy_engine()
    events = [
        NarrativeEvent("The Hare challenges", TensionLevel.BUILDUP),
        NarrativeEvent("The Tortoise wins", TensionLevel.CLIMAX)
    ]
    archetype = alchemy.extract_archetype("Test Story", events)

    assert isinstance(archetype, Archetype)
    assert archetype.name == "Archetype of Test Story"
    assert len(archetype.structure) == 2
    # Verify the structure contains the abstract nodes
    assert archetype.structure[1].function_name == "Triumph of the Persistent"

def test_transmutation_flow():
    alchemy = get_alchemy_engine()
    events = [NarrativeEvent("The Tortoise wins", TensionLevel.CLIMAX)]
    archetype = alchemy.extract_archetype("Simple Win", events)

    # Transmute to Space War
    new_story = alchemy.transmute(archetype, "Space War")
    assert len(new_story) == 1
    # Check if "Rebel" (mapped from weak/tortoise/win) appears
    assert "Rebel Alliance" in new_story[0]
    assert "destroys" in new_story[0]

    # Transmute to Business
    new_story_biz = alchemy.transmute(archetype, "Business")
    assert "Garage Startup" in new_story_biz[0]
    assert "disrupts" in new_story_biz[0]
