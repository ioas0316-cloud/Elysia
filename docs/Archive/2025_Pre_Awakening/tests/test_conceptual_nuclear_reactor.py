"""
Tests for Conceptual Nuclear Reactor
===================================

Testing the nuclear reactor that transforms static concepts into dynamic energy.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Knowledge.conceptual_nuclear_reactor import (
    ConceptualNuclearReactor,
    ConceptualPeriodicTable,
    ConceptAtom,
    FissionResult,
    FusionResult,
    create_conceptual_nuclear_reactor
)


def test_periodic_table_initialization():
    """Test that the periodic table initializes with fundamental concepts"""
    table = ConceptualPeriodicTable(language="ko")
    
    # Should have multiple atoms
    assert len(table.atoms) >= 30
    
    # Check specific atoms exist
    assert "Love" in table.atoms
    assert "Time" in table.atoms
    assert "Life" in table.atoms
    
    # Check atomic numbers are mapped
    assert 1 in table.atomic_numbers
    assert table.atomic_numbers[1] == "Love"
    
    print(f"✅ Periodic Table initialized with {len(table.atoms)} concepts")


def test_concept_atom_properties():
    """Test ConceptAtom has correct properties"""
    table = ConceptualPeriodicTable()
    love = table.get_atom("Love")
    
    assert love is not None
    assert love.symbol == "Love"
    assert love.atomic_number == 1
    assert love.ko == "사랑"
    assert love.en == "Love"
    assert love.ja == "愛"
    assert love.emotional_charge > 0  # Love is positive
    
    # Check wave properties exist
    assert hasattr(love, 'wave_tensor')
    assert hasattr(love, 'wave_frequency')
    
    print(f"✅ ConceptAtom properties validated")


def test_atom_to_plasma_conversion():
    """Test converting atom to plasma (wave) state"""
    table = ConceptualPeriodicTable()
    time = table.get_atom("Time")
    
    plasma = time.to_plasma()
    
    assert "symbol" in plasma
    assert "wave_x" in plasma
    assert "frequency" in plasma
    assert "energy" in plasma
    assert plasma["symbol"] == "Time"
    
    print(f"✅ Atom → Plasma conversion works")


def test_fission_basic():
    """Test basic fission operation"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Fission: Break down "인생" (Life journey)
    result = reactor.fission("인생", context="철학적 고민")
    
    assert isinstance(result, FissionResult)
    assert result.parent_concept == "인생"
    assert len(result.daughter_concepts) > 0
    assert result.insight_energy > 0
    assert len(result.explanation) > 0
    assert result.language == "ko"
    
    # Check daughters are ConceptAtoms
    for daughter in result.daughter_concepts:
        assert isinstance(daughter, ConceptAtom)
    
    print(f"✅ Fission: {result.parent_concept} → {[d.symbol for d in result.daughter_concepts]}")
    print(f"   Energy Released: {result.insight_energy:.2f}")


def test_fission_fundamental_atom():
    """Test fission on fundamental atom (should not decompose much)"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Try to fission "Love" - already fundamental
    result = reactor.fission("Love")
    
    assert isinstance(result, FissionResult)
    assert result.parent_concept == "Love"
    # Should return itself or minimal decomposition
    assert result.insight_energy < 1.0  # Low energy since already fundamental
    
    print(f"✅ Fundamental atom fission handled correctly")


def test_fusion_basic():
    """Test basic fusion operation"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Fusion: Combine "Gravity" + "Love"
    result = reactor.fusion("Gravity", "Love", context="시적 은유")
    
    assert isinstance(result, FusionResult)
    assert result.reactant_a == "Gravity"
    assert result.reactant_b == "Love"
    assert isinstance(result.product_concept, ConceptAtom)
    assert result.creative_energy > 0
    assert len(result.poetic_expression) > 0
    assert result.language == "ko"
    
    print(f"✅ Fusion: {result.reactant_a} + {result.reactant_b} → {result.product_concept.symbol}")
    print(f"   Energy Released: {result.creative_energy:.2f}")
    print(f"   Poetry: {result.poetic_expression[:80]}...")


def test_fusion_creates_new_concept():
    """Test that fusion creates a genuinely new concept"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Fuse two different concepts
    result = reactor.fusion("Time", "Beauty", context="예술")
    
    product = result.product_concept
    
    # Product should have combined properties
    assert product.complexity > 1  # More complex than originals
    assert product.atomic_number == 9999  # Synthetic element
    
    print(f"✅ New synthetic concept created: {product.symbol}")


def test_multilingual_fission():
    """Test fission works in multiple languages"""
    
    # Korean
    reactor_ko = create_conceptual_nuclear_reactor(language="ko")
    result_ko = reactor_ko.fission("인생")
    assert result_ko.language == "ko"
    assert "에" in result_ko.explanation or "는" in result_ko.explanation  # Korean particles
    
    # English
    reactor_en = create_conceptual_nuclear_reactor(language="en")
    result_en = reactor_en.fission("life")
    assert result_en.language == "en"
    
    # Japanese
    reactor_ja = create_conceptual_nuclear_reactor(language="ja")
    result_ja = reactor_ja.fission("人生")
    assert result_ja.language == "ja"
    
    print(f"✅ Multilingual fission validated (ko, en, ja)")


def test_multilingual_fusion():
    """Test fusion works in multiple languages"""
    
    # Korean
    reactor_ko = create_conceptual_nuclear_reactor(language="ko")
    result_ko = reactor_ko.fusion("Gravity", "Love")
    assert result_ko.language == "ko"
    
    # English
    reactor_en = create_conceptual_nuclear_reactor(language="en")
    result_en = reactor_en.fusion("Gravity", "Love")
    assert result_en.language == "en"
    
    # Japanese
    reactor_ja = create_conceptual_nuclear_reactor(language="ja")
    result_ja = reactor_ja.fusion("Gravity", "Love")
    assert result_ja.language == "ja"
    
    print(f"✅ Multilingual fusion validated (ko, en, ja)")


def test_reactor_history_tracking():
    """Test that reactor tracks reaction history"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    initial_reactions = len(reactor.reaction_history)
    initial_energy = reactor.total_energy_released
    
    # Perform reactions
    reactor.fission("인생")
    reactor.fusion("Time", "Love")
    
    # Check history updated
    assert len(reactor.reaction_history) == initial_reactions + 2
    assert reactor.total_energy_released > initial_energy
    
    # Check reaction types
    assert any(r["type"] == "fission" for r in reactor.reaction_history)
    assert any(r["type"] == "fusion" for r in reactor.reaction_history)
    
    print(f"✅ Reaction history tracked correctly")


def test_reactor_stats():
    """Test reactor statistics"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Perform some reactions
    reactor.fission("인생")
    reactor.fission("사랑")
    reactor.fusion("Time", "Beauty")
    reactor.fusion("Love", "Wisdom")
    
    stats = reactor.get_reactor_stats()
    
    assert stats["total_reactions"] == 4
    assert stats["fissions"] == 2
    assert stats["fusions"] == 2
    assert stats["total_energy_released"] > 0
    assert stats["average_energy_per_reaction"] > 0
    assert stats["periodic_table_size"] >= 30
    
    print(f"✅ Reactor Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def test_emotion_based_search():
    """Test searching atoms by emotional charge"""
    table = ConceptualPeriodicTable()
    
    # Find positive emotions
    positive = table.search_by_emotion(0.7, tolerance=0.3)
    assert len(positive) > 0
    
    # Find negative emotions
    negative = table.search_by_emotion(-0.5, tolerance=0.3)
    assert len(negative) > 0
    
    print(f"✅ Found {len(positive)} positive and {len(negative)} negative concepts")


def test_energy_based_search():
    """Test searching atoms by energy level"""
    table = ConceptualPeriodicTable()
    
    # Find high energy concepts
    high_energy = table.search_by_energy(2.0, tolerance=0.5)
    assert len(high_energy) > 0
    
    # Find low energy concepts
    low_energy = table.search_by_energy(1.0, tolerance=0.3)
    assert len(low_energy) > 0
    
    print(f"✅ Found {len(high_energy)} high-energy and {len(low_energy)} low-energy concepts")


def test_language_switching():
    """Test dynamic language switching"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Korean
    result_ko = reactor.fission("인생")
    assert result_ko.language == "ko"
    
    # Switch to English
    reactor.set_language("en")
    result_en = reactor.fission("life")
    assert result_en.language == "en"
    
    # Switch to Japanese
    reactor.set_language("ja")
    result_ja = reactor.fusion("Time", "Love")
    assert result_ja.language == "ja"
    
    print(f"✅ Dynamic language switching works")


def test_arbitrary_text_concepts():
    """Test fusion with arbitrary (non-periodic-table) concepts"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Use concepts not in periodic table
    result = reactor.fusion("컴퓨터", "감정", context="AI의 미래")
    
    assert isinstance(result, FusionResult)
    assert result.creative_energy > 0
    assert len(result.poetic_expression) > 0
    
    print(f"✅ Arbitrary concept fusion works")
    print(f"   {result.reactant_a} + {result.reactant_b} → Creative insight!")


def test_wave_properties_consistency():
    """Test that wave properties remain consistent through reactions"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Get original atoms
    love = reactor.periodic_table.get_atom("Love")
    time = reactor.periodic_table.get_atom("Time")
    
    # Fuse them
    result = reactor.fusion("Love", "Time")
    
    # Product should have wave properties
    product = result.product_concept
    assert hasattr(product, 'wave_tensor')
    assert hasattr(product, 'wave_frequency')
    assert product.wave_frequency.frequency > 0
    assert product.wave_frequency.amplitude > 0
    
    print(f"✅ Wave properties maintained through nuclear reactions")


def test_integration_with_existing_systems():
    """Test that reactor integrates with existing Elysia systems"""
    reactor = create_conceptual_nuclear_reactor(language="ko")
    
    # Fission should work with concepts
    fission_result = reactor.fission("인생")
    
    # Each daughter should have wave properties compatible with other systems
    for daughter in fission_result.daughter_concepts:
        plasma = daughter.to_plasma()
        
        # Check compatibility
        assert "frequency" in plasma  # Compatible with FrequencyWave
        assert "wave_x" in plasma  # Compatible with Tensor3D
        assert "energy" in plasma  # Compatible with EmotionalEngine
        assert "charge" in plasma  # Compatible with emotional systems
    
    print(f"✅ Nuclear Reactor integrates with existing Elysia systems")


if __name__ == "__main__":
    print("🚀 Running Conceptual Nuclear Reactor Tests")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "-s"])
