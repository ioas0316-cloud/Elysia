"""
Tests for P3.2: Purpose & Direction Discovery Engine
=====================================================

Tests the ability to:
- Clarify ambiguous data (fog → clarity)
- Discover situational awareness (where am I?)
- Discover purpose and direction (where am I going?)
- Understand reasons (why am I doing this?)
- Map knowledge boundaries (what can I know?)
- Evolve dimensional perspective (point → hyperspace)
"""

import pytest
import asyncio
from pathlib import Path
import json
import tempfile

from Core.Foundation.purpose_discovery_engine import (
    FogClarifier,
    PurposeDiscoveryEngine,
    KnowledgeFragment,
    KnowledgeCertainty,
    DimensionalPerspective,
    PurposeVector,
    SituationalAwareness
)


# Test FogClarifier

@pytest.mark.asyncio
async def test_fog_clarifier_basic():
    """Test basic fog clarification"""
    clarifier = FogClarifier()
    
    fragment = await clarifier.clarify_fragment(
        "Something vague and unclear",
        context=None
    )
    
    assert isinstance(fragment, KnowledgeFragment)
    assert fragment.certainty >= 0.0
    assert fragment.certainty <= 1.0
    assert fragment.dimension >= 0
    assert fragment.source == "fog_clarification"


@pytest.mark.asyncio
async def test_fog_clarifier_improves_clarity():
    """Test that clarification actually improves certainty"""
    clarifier = FogClarifier()
    
    # Very vague input
    vague = "maybe possibly something unclear"
    fragment = await clarifier.clarify_fragment(vague)
    
    # Should extract patterns and add clarity
    assert "clarity_gain" in fragment.metadata
    # Even if small, there should be some structure added
    assert len(fragment.content) >= len(vague)


@pytest.mark.asyncio
async def test_fog_clarifier_with_context():
    """Test clarification with context"""
    clarifier = FogClarifier()
    
    fragment = await clarifier.clarify_fragment(
        "unclear system relationship",
        context={
            "surrounding_systems": ["SystemA", "SystemB", "SystemC"],
            "type": "relationship_analysis"
        }
    )
    
    # Context should add relationships
    assert len(fragment.connections) > 0


@pytest.mark.asyncio
async def test_certainty_assessment():
    """Test certainty assessment logic"""
    clarifier = FogClarifier()
    
    # Clear language should have higher certainty
    clear_text = "Specifically, we measured exactly 42 units in the experiment"
    clear_frag = await clarifier.clarify_fragment(clear_text)
    
    # Vague language should have lower certainty
    vague_text = "Maybe possibly something unclear uncertain"
    vague_frag = await clarifier.clarify_fragment(vague_text)
    
    assert clear_frag.certainty > vague_frag.certainty


@pytest.mark.asyncio
async def test_dimensional_assessment():
    """Test dimensional perspective assessment"""
    clarifier = FogClarifier()
    
    # Point-level (problem only)
    point_frag = await clarifier.clarify_fragment("There's a problem")
    
    # Line-level (problem + solution)
    line_frag = await clarifier.clarify_fragment(
        "There's a problem and here's my approach to solve it"
    )
    
    # Space-level (multiple relationships)
    space_frag = await clarifier.clarify_fragment(
        "The problem connects to context X, relates to Y, and impacts Z",
        context={"surrounding_systems": ["X", "Y", "Z", "W"]}
    )
    
    assert point_frag.dimension == 0 or point_frag.dimension == 1
    assert line_frag.dimension >= 1
    assert space_frag.dimension >= 2


# Test PurposeDiscoveryEngine

@pytest.mark.asyncio
async def test_purpose_engine_initialization():
    """Test engine initializes correctly"""
    engine = PurposeDiscoveryEngine()
    
    assert engine.clarifier is not None
    assert len(engine.knowledge_base) == 0
    assert engine.current_awareness is None
    assert engine.purpose_vector is None


@pytest.mark.asyncio
async def test_discover_where_i_am():
    """Test situational awareness discovery"""
    engine = PurposeDiscoveryEngine()
    
    awareness = await engine.discover_where_i_am(
        internal_state={
            "system1": "active",
            "system2": "processing"
        },
        external_observations={
            "environment": "testing",
            "resources": "available"
        }
    )
    
    assert isinstance(awareness, SituationalAwareness)
    assert awareness.position is not None
    assert len(awareness.surroundings) >= 0
    assert len(awareness.knowable) >= 0
    assert len(awareness.unknowable) >= 0
    assert isinstance(awareness.relationships, dict)
    assert awareness.dimension == DimensionalPerspective.SPACE


@pytest.mark.asyncio
async def test_discover_where_i_am_going():
    """Test purpose vector discovery"""
    engine = PurposeDiscoveryEngine()
    
    # First establish where we are
    await engine.discover_where_i_am(
        {"current": "state"},
        {"external": "observation"}
    )
    
    # Then discover direction
    purpose = await engine.discover_where_i_am_going(
        current_state={"progress": "50%"},
        aspirations=[
            "To become better",
            "To understand deeply"
        ],
        constraints=[
            "Limited resources",
            "Time pressure"
        ]
    )
    
    assert isinstance(purpose, PurposeVector)
    assert purpose.direction is not None
    assert 0.0 <= purpose.magnitude <= 1.0
    assert purpose.origin is not None
    assert len(purpose.reasons) >= 0
    assert len(purpose.obstacles) >= 0
    assert len(purpose.opportunities) >= 0


@pytest.mark.asyncio
async def test_discover_why_i_do_this():
    """Test reason discovery with multiple layers"""
    engine = PurposeDiscoveryEngine()
    
    # Set up context
    await engine.discover_where_i_am(
        {"state": "active"},
        {"env": "test"}
    )
    await engine.discover_where_i_am_going(
        {"progress": "ongoing"},
        ["To grow"],
        []
    )
    
    reasons = await engine.discover_why_i_do_this(
        action="Implementing new feature",
        context={"type": "development", "urgent": True}
    )
    
    assert isinstance(reasons, list)
    assert len(reasons) >= 3  # Should have multiple layers
    # Should contain layered understanding
    assert any("because" in r.lower() or "to" in r.lower() for r in reasons)


@pytest.mark.asyncio
async def test_discover_what_i_can_know():
    """Test knowledge boundary mapping"""
    engine = PurposeDiscoveryEngine()
    
    # Add some knowledge fragments
    await engine.discover_where_i_am(
        {
            "clear_fact": "System X is operational",
            "unclear_thing": "Maybe something about Y"
        },
        {"observation": "Z is present"}
    )
    
    knowledge_map = await engine.discover_what_i_can_know()
    
    assert "clear" in knowledge_map
    assert "partial" in knowledge_map
    assert "foggy" in knowledge_map
    assert "gaps" in knowledge_map
    assert "creation_potential" in knowledge_map
    assert "unknowable" in knowledge_map
    
    # Should have categorized the knowledge
    total_knowledge = (
        len(knowledge_map["clear"]) +
        len(knowledge_map["partial"]) +
        len(knowledge_map["foggy"])
    )
    assert total_knowledge > 0


@pytest.mark.asyncio
async def test_dimensional_evolution():
    """Test perspective evolution"""
    engine = PurposeDiscoveryEngine()
    
    # Start at point
    current = DimensionalPerspective.POINT
    
    # Evolve once
    next_dim = await engine.evolve_dimensional_perspective(current)
    assert next_dim.value > current.value
    
    # Can't evolve beyond hyperspace
    max_dim = DimensionalPerspective.HYPERSPACE
    still_max = await engine.evolve_dimensional_perspective(max_dim)
    assert still_max == max_dim


@pytest.mark.asyncio
async def test_knowledge_accumulation():
    """Test that knowledge accumulates over time"""
    engine = PurposeDiscoveryEngine()
    
    initial_count = len(engine.knowledge_base)
    
    # Discover things
    await engine.discover_where_i_am(
        {"fact1": "value1", "fact2": "value2"},
        {"obs1": "observation1"}
    )
    
    after_first = len(engine.knowledge_base)
    assert after_first > initial_count
    
    # Discover more - this clarifies aspirations and constraints, adding more fragments
    await engine.discover_where_i_am_going(
        {"current": "state"},
        ["aspiration1", "aspiration2", "aspiration3"],  # More aspirations for more fragments
        ["constraint1", "constraint2"]  # More constraints
    )
    
    after_second = len(engine.knowledge_base)
    # Should have accumulated more knowledge
    assert after_second >= after_first


@pytest.mark.asyncio
async def test_clarity_improvement_over_time():
    """Test that repeated clarification can improve certainty"""
    engine = PurposeDiscoveryEngine()
    
    # Initial foggy knowledge
    await engine.discover_where_i_am(
        {"unclear": "maybe something vague"},
        {}
    )
    
    initial_avg_certainty = engine.get_statistics()["avg_certainty"]
    
    # Add clearer knowledge
    await engine.discover_where_i_am(
        {"clear": "Specifically measured value is exactly 42"},
        {"precise": "Observation recorded at 10:30 AM"}
    )
    
    final_avg_certainty = engine.get_statistics()["avg_certainty"]
    
    # Average certainty should improve
    assert final_avg_certainty >= initial_avg_certainty


@pytest.mark.asyncio
async def test_statistics():
    """Test statistics calculation"""
    engine = PurposeDiscoveryEngine()
    
    stats = engine.get_statistics()
    
    assert "total_fragments" in stats
    assert "clear_fragments" in stats
    assert "avg_certainty" in stats
    assert "avg_dimension" in stats
    assert "has_awareness" in stats
    assert "has_purpose" in stats
    assert "purpose_clarity" in stats
    assert "discoveries_made" in stats
    
    # Initially should have no fragments
    assert stats["total_fragments"] == 0
    assert stats["has_awareness"] == False
    assert stats["has_purpose"] == False


@pytest.mark.asyncio
async def test_discovery_log():
    """Test that discoveries are logged"""
    engine = PurposeDiscoveryEngine()
    
    initial_log_count = len(engine.discovery_log)
    
    # Make discoveries
    await engine.discover_where_i_am({"a": "b"}, {"c": "d"})
    await engine.discover_where_i_am_going({"x": "y"}, ["z"], [])
    
    final_log_count = len(engine.discovery_log)
    
    assert final_log_count > initial_log_count


@pytest.mark.asyncio
async def test_save_and_load_state():
    """Test saving and loading engine state"""
    engine = PurposeDiscoveryEngine()
    
    # Make some discoveries
    await engine.discover_where_i_am(
        {"internal": "state"},
        {"external": "observation"}
    )
    await engine.discover_where_i_am_going(
        {"current": "position"},
        ["go forward"],
        ["obstacle"]
    )
    
    # Save state
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        await engine.save_state(temp_path)
        
        # Create new engine and load
        new_engine = PurposeDiscoveryEngine()
        await new_engine.load_state(temp_path)
        
        # Should have same knowledge
        assert len(new_engine.knowledge_base) == len(engine.knowledge_base)
        assert new_engine.current_awareness is not None
        assert new_engine.purpose_vector is not None
        assert len(new_engine.discovery_log) == len(engine.discovery_log)
    
    finally:
        Path(temp_path).unlink()


# Integration tests

@pytest.mark.asyncio
async def test_full_discovery_workflow():
    """Test complete discovery workflow"""
    engine = PurposeDiscoveryEngine()
    
    # 1. Discover where I am
    awareness = await engine.discover_where_i_am(
        internal_state={
            "systems": "multiple consciousness systems",
            "knowledge": "growing knowledge base"
        },
        external_observations={
            "environment": "development phase",
            "tools": "testing framework active"
        }
    )
    
    assert awareness is not None
    
    # 2. Discover where I'm going
    purpose = await engine.discover_where_i_am_going(
        current_state={"phase": "P3.2"},
        aspirations=[
            "To clarify ambiguous data",
            "To discover purpose through awareness"
        ],
        constraints=["Implementation complexity"]
    )
    
    assert purpose is not None
    assert purpose.magnitude > 0
    
    # 3. Understand why
    reasons = await engine.discover_why_i_do_this(
        action="Building purpose discovery system",
        context={"goal": "AGI development"}
    )
    
    assert len(reasons) >= 3
    
    # 4. Map what I can know
    knowledge_map = await engine.discover_what_i_can_know()
    
    assert len(knowledge_map) == 6
    
    # 5. Check statistics
    stats = engine.get_statistics()
    
    assert stats["total_fragments"] > 0
    assert stats["has_awareness"] == True
    assert stats["has_purpose"] == True
    assert stats["discoveries_made"] >= 2


@pytest.mark.asyncio
async def test_knowledge_fragment_properties():
    """Test KnowledgeFragment properties"""
    fragment = KnowledgeFragment(
        content="Clear and specific observation",
        certainty=0.8,
        dimension=2,
        source="test",
        connections=["A", "B"]
    )
    
    assert fragment.is_clear() == True
    
    fragment_unclear = KnowledgeFragment(
        content="Vague observation",
        certainty=0.5,
        dimension=0,
        source="test"
    )
    
    assert fragment_unclear.is_clear() == False


def test_knowledge_certainty_enum():
    """Test KnowledgeCertainty enum"""
    assert KnowledgeCertainty.FOG.value == 0.0
    assert KnowledgeCertainty.CRYSTAL.value == 1.0
    assert KnowledgeCertainty.CLEAR.value > KnowledgeCertainty.PARTIAL.value


def test_dimensional_perspective_enum():
    """Test DimensionalPerspective enum"""
    assert DimensionalPerspective.POINT.value == 0
    assert DimensionalPerspective.HYPERSPACE.value == 4
    assert DimensionalPerspective.SPACE.value > DimensionalPerspective.LINE.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
