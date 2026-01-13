"""
Tests for Mythology & Theology Domain
======================================

Tests the archetypal and mythological pattern extraction.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Core.Knowledge.Domains.mythology import (
    MythologyDomain,
    Archetype,
    JourneyStage,
    get_similar_myths
)


class TestMythologyDomain:
    """Test suite for Mythology domain"""
    
    def test_initialization(self):
        """Test domain initialization"""
        myth = MythologyDomain()
        
        assert myth.name == "Mythology & Theology"
        assert len(myth.archetypes) > 0
        assert len(myth.journey_patterns) > 0
        assert Archetype.HERO in myth.archetypes
        assert JourneyStage.CALL_TO_ADVENTURE in myth.journey_patterns
    
    def test_extract_pattern(self):
        """Test wave pattern extraction"""
        myth = MythologyDomain()
        
        content = "The hero embarks on a journey facing many challenges with courage"
        pattern = myth.extract_pattern(content)
        
        assert pattern is not None
        assert pattern.text == content
        assert 0 <= pattern.energy <= 1.0
        assert pattern.metadata is not None
        assert 'archetypes' in pattern.metadata
        assert 'journey_stage' in pattern.metadata
    
    def test_detect_archetypes(self):
        """Test archetype detection"""
        myth = MythologyDomain()
        
        # Hero archetype
        hero_content = "A brave warrior hero fights with courage"
        archetypes = myth._detect_archetypes(hero_content)
        assert Archetype.HERO in archetypes
        
        # Wise old man
        sage_content = "The wise mentor teaches ancient knowledge"
        archetypes = myth._detect_archetypes(sage_content)
        assert Archetype.WISE_OLD_MAN in archetypes
        
        # Multiple archetypes
        mixed_content = "The hero seeks wisdom from the wise teacher"
        archetypes = myth._detect_archetypes(mixed_content)
        assert len(archetypes) >= 2
    
    def test_detect_journey_stage(self):
        """Test Hero's Journey stage detection"""
        myth = MythologyDomain()
        
        # Call to adventure
        call_content = "A mysterious call summons the hero to adventure"
        stage = myth._detect_journey_stage(call_content)
        assert stage == JourneyStage.CALL_TO_ADVENTURE
        
        # Ordeal
        ordeal_content = "The hero faces the greatest fear in darkest hour"
        stage = myth._detect_journey_stage(ordeal_content)
        assert stage == JourneyStage.ORDEAL
        
        # Return
        return_content = "The hero returns home with elixir and wisdom to share"
        stage = myth._detect_journey_stage(return_content)
        assert stage == JourneyStage.RETURN_WITH_ELIXIR
    
    def test_spiritual_resonance(self):
        """Test spiritual resonance analysis"""
        myth = MythologyDomain()
        
        # High spiritual content
        spiritual_content = "Divine soul transcendent eternal sacred enlightenment"
        analysis = myth.analyze(spiritual_content)
        assert analysis['spiritual_resonance'] > 0.5
        
        # Low spiritual content
        mundane_content = "Regular everyday normal text"
        analysis = myth.analyze(mundane_content)
        # Should still have some baseline
        assert 0 <= analysis['spiritual_resonance'] <= 1.0
    
    def test_transcendent_meaning(self):
        """Test transcendent meaning analysis"""
        myth = MythologyDomain()
        
        # High transcendent content
        transcendent_content = "Eternal truth infinite cosmic meaning ultimate purpose"
        analysis = myth.analyze(transcendent_content)
        assert analysis['transcendent_meaning'] > 0.3
        
        # Plain content
        plain_content = "Just some words here"
        analysis = myth.analyze(plain_content)
        assert 0 <= analysis['transcendent_meaning'] <= 1.0
    
    def test_archetypal_energy(self):
        """Test archetypal energy calculation"""
        myth = MythologyDomain()
        
        # Strong archetypal content
        content = "The hero warrior with courage faces the shadow"
        analysis = myth.analyze(content)
        assert analysis['archetypal_energy'] > 0.5
    
    def test_journey_position(self):
        """Test journey position calculation"""
        myth = MythologyDomain()
        
        # Beginning stages should have low position
        pos_start = myth._get_journey_position(JourneyStage.ORDINARY_WORLD)
        assert pos_start < 0.3
        
        # Middle stages
        pos_middle = myth._get_journey_position(JourneyStage.ORDEAL)
        assert 0.5 < pos_middle < 0.9
        
        # End stages
        pos_end = myth._get_journey_position(JourneyStage.RETURN_WITH_ELIXIR)
        assert pos_end >= 0.9
    
    def test_identify_journey_stage(self):
        """Test journey stage identification"""
        myth = MythologyDomain()
        
        situation = "Facing a difficult challenge that tests my courage"
        result = myth.identify_journey_stage(situation)
        
        assert 'current_stage' in result
        assert 'stage_name' in result
        assert 'position' in result
        assert 'archetypes' in result
        assert 'guidance' in result
        assert 'spiritual_message' in result
        
        # Position should be between 0 and 1
        assert 0 <= result['position'] <= 1.0
    
    def test_stage_guidance(self):
        """Test stage-specific guidance"""
        myth = MythologyDomain()
        
        # Each stage should have guidance
        for stage in JourneyStage:
            guidance = myth._get_stage_guidance(stage)
            assert isinstance(guidance, str)
            assert len(guidance) > 0
    
    def test_spiritual_message(self):
        """Test spiritual message crafting"""
        myth = MythologyDomain()
        
        # Call to adventure
        msg = myth._craft_spiritual_message(
            JourneyStage.CALL_TO_ADVENTURE,
            [Archetype.HERO]
        )
        assert isinstance(msg, str)
        assert len(msg) > 0
        
        # Ordeal
        msg = myth._craft_spiritual_message(
            JourneyStage.ORDEAL,
            [Archetype.SHADOW]
        )
        assert isinstance(msg, str)
        assert len(msg) > 0
    
    def test_domain_dimension(self):
        """Test domain dimension mapping"""
        myth = MythologyDomain()
        assert myth.get_domain_dimension() == "archetype"
    
    def test_pattern_storage(self):
        """Test pattern storage"""
        myth = MythologyDomain()
        
        initial_count = len(myth.patterns)
        myth.extract_pattern("The hero's journey begins")
        
        assert len(myth.patterns) == initial_count + 1
    
    def test_query_patterns(self):
        """Test pattern querying"""
        myth = MythologyDomain()
        
        # Add patterns
        myth.extract_pattern("The hero faces challenges")
        myth.extract_pattern("Wisdom from the sage")
        myth.extract_pattern("Something unrelated")
        
        # Query
        results = myth.query_patterns("hero", top_k=5)
        
        assert len(results) >= 1
        assert any('hero' in r.text.lower() for r in results)


class TestArchetypes:
    """Test archetype enum and patterns"""
    
    def test_archetype_enum(self):
        """Test archetype enum values"""
        assert Archetype.HERO.value == "hero"
        assert Archetype.SHADOW.value == "shadow"
        assert Archetype.WISE_OLD_MAN.value == "wise_old_man"
        assert Archetype.SELF.value == "self"
    
    def test_all_archetypes_covered(self):
        """Test all archetypes have patterns"""
        myth = MythologyDomain()
        
        # Most major archetypes should have patterns
        important_archetypes = [
            Archetype.HERO,
            Archetype.SHADOW,
            Archetype.WISE_OLD_MAN,
            Archetype.MOTHER,
            Archetype.FATHER,
            Archetype.SELF,
        ]
        
        for archetype in important_archetypes:
            assert archetype in myth.archetypes
            assert 'keywords' in myth.archetypes[archetype]
            assert 'energy' in myth.archetypes[archetype]


class TestJourneyStages:
    """Test Hero's Journey stages"""
    
    def test_journey_stage_enum(self):
        """Test journey stage enum values"""
        assert JourneyStage.ORDINARY_WORLD.value == "ordinary_world"
        assert JourneyStage.CALL_TO_ADVENTURE.value == "call_to_adventure"
        assert JourneyStage.ORDEAL.value == "ordeal"
        assert JourneyStage.RETURN_WITH_ELIXIR.value == "return_with_elixir"
    
    def test_all_stages_covered(self):
        """Test all journey stages have patterns"""
        myth = MythologyDomain()
        
        for stage in JourneyStage:
            assert stage in myth.journey_patterns
            assert 'keywords' in myth.journey_patterns[stage]
            assert 'narrative_position' in myth.journey_patterns[stage]
    
    def test_stage_progression(self):
        """Test stages progress from 0 to 1"""
        myth = MythologyDomain()
        
        positions = [
            myth.journey_patterns[stage]['narrative_position']
            for stage in JourneyStage
        ]
        
        # Should be in ascending order
        for i in range(len(positions) - 1):
            assert positions[i] <= positions[i+1]
        
        # First should be near 0, last should be 1
        assert positions[0] < 0.2
        assert positions[-1] == 1.0


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_get_similar_myths(self):
        """Test getting similar myths for archetypes"""
        # Hero archetype
        myths = get_similar_myths(Archetype.HERO)
        assert len(myths) > 0
        assert 'Odysseus' in myths or 'Gilgamesh' in myths
        
        # Wise old man
        myths = get_similar_myths(Archetype.WISE_OLD_MAN)
        assert len(myths) > 0
        assert 'Merlin' in myths or 'Gandalf' in myths
        
        # Unknown archetype
        myths = get_similar_myths(Archetype.CHILD)
        # Should return empty list or default
        assert isinstance(myths, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
