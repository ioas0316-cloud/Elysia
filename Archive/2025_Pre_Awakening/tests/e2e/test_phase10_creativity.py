"""
Tests for Phase 10: Creativity & Art Generation

Tests the creative systems:
- Story Generation
- Music Composition
- Visual Art Creation
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Foundation.story_generator import StoryGenerator, StoryStyle
from Core.Foundation.music_composer import MusicComposer, MusicEmotion
from Core.Foundation.visual_artist import VisualArtist, ArtStyle


class TestStoryGenerator:
    """Test story generation system"""
    
    @pytest.fixture
    def generator(self):
        return StoryGenerator()
    
    @pytest.mark.asyncio
    async def test_story_generation_fantasy(self, generator):
        """Test fantasy story generation"""
        story = await generator.generate_story(
            prompt="A hero's quest",
            style="fantasy",
            length="short"
        )
        
        assert story is not None
        assert "world" in story
        assert "characters" in story
        assert "plot" in story
        assert "full_story" in story
        assert "meta" in story
        
        # Check world
        assert story["world"]["name"]
        assert len(story["world"]["rules"]) > 0
        
        # Check characters
        assert len(story["characters"]) >= 2
        assert any(c["role"] == "protagonist" for c in story["characters"])
        
        # Check plot
        assert len(story["plot"]) >= 3
        
        # Check metadata
        assert story["meta"]["title"]
        assert len(story["meta"]["themes"]) > 0
    
    @pytest.mark.asyncio
    async def test_story_generation_scifi(self, generator):
        """Test science fiction story generation"""
        story = await generator.generate_story(
            prompt="Space exploration",
            style="science_fiction",
            length="short"
        )
        
        assert story is not None
        assert story["world"]["technology_level"] == "advanced"
        assert len(story["characters"]) >= 2
    
    @pytest.mark.asyncio
    async def test_world_building(self, generator):
        """Test world building"""
        world = await generator.build_world("Magic kingdom", "fantasy")
        
        assert world.name
        assert world.description
        assert len(world.locations) > 0
        assert world.technology_level
    
    @pytest.mark.asyncio
    async def test_character_creation(self, generator):
        """Test character creation"""
        world = await generator.build_world("Test world", "fantasy")
        characters = await generator.create_characters(world, "Test prompt")
        
        assert len(characters) >= 2
        
        # Check for protagonist
        protagonist = next((c for c in characters if c.role == "protagonist"), None)
        assert protagonist is not None
        assert protagonist.name
        assert len(protagonist.personality) > 0
    
    @pytest.mark.asyncio
    async def test_plot_construction(self, generator):
        """Test plot construction"""
        world = await generator.build_world("Test world", "fantasy")
        characters = await generator.create_characters(world, "Test prompt")
        plot = await generator.construct_plot(world, characters, "Test prompt", "short")
        
        assert len(plot) >= 3
        assert plot[0].sequence == 1
        assert all(p.characters_involved for p in plot)
    
    def test_theme_extraction(self, generator):
        """Test theme extraction"""
        from Core.Foundation.story_generator import Story, World, PlotPoint, EmotionType
        
        # Create mock story
        world = World("Test", "Test world")
        plot = [
            PlotPoint(1, "discover the truth", [], "loc", EmotionType.HOPE, 0.8),
            PlotPoint(2, "overcome challenges", [], "loc", EmotionType.TENSION, 0.9)
        ]
        
        from dataclasses import dataclass, field
        @dataclass
        class MockStory:
            world: World
            plot: list
        
        story = MockStory(world=world, plot=plot)
        themes = generator.extract_themes(story)
        
        assert len(themes) > 0
        assert isinstance(themes, list)


class TestMusicComposer:
    """Test music composition system"""
    
    @pytest.fixture
    def composer(self):
        return MusicComposer()
    
    @pytest.mark.asyncio
    async def test_music_composition_joyful(self, composer):
        """Test joyful music composition"""
        composition = await composer.compose_music(
            emotion="joyful",
            style="classical",
            duration_bars=4
        )
        
        assert composition is not None
        assert "composition" in composition
        assert "analysis" in composition
        
        # Check analysis
        assert composition["analysis"]["key"]
        assert composition["analysis"]["tempo"] > 0
        assert 0.0 <= composition["analysis"]["emotion_match"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_music_composition_melancholic(self, composer):
        """Test melancholic music composition"""
        composition = await composer.compose_music(
            emotion="melancholic",
            style="jazz",
            duration_bars=8
        )
        
        assert composition is not None
        assert "minor" in composition["analysis"]["scale"]
        assert composition["analysis"]["tempo"] < 100  # Slower tempo for melancholic
    
    @pytest.mark.asyncio
    async def test_melody_generation(self, composer):
        """Test melody generation"""
        key_data = {
            "root_midi": 60,
            "note_name": "C",
            "scale_type": "major",
            "mode": "major"
        }
        
        melody = await composer.generate_melody(key_data, MusicEmotion.JOYFUL, 4)
        
        assert len(melody.notes) > 0
        assert melody.key == "C"
        assert all(note.pitch >= 0 and note.pitch <= 127 for note in melody.notes)
    
    @pytest.mark.asyncio
    async def test_harmony_generation(self, composer):
        """Test harmony generation"""
        from Core.Foundation.music_composer import Melody, Note
        
        key_data = {
            "root_midi": 60,
            "note_name": "C",
            "scale_type": "major",
            "mode": "major"
        }
        
        melody = Melody(notes=[Note(60, 1.0)], key="C", scale_type="major")
        harmony = await composer.generate_harmony(melody, key_data)
        
        assert len(harmony.chords) > 0
        assert all(chord.root >= 0 for chord in harmony.chords)
    
    def test_emotion_mapping(self, composer):
        """Test emotion to music parameter mapping"""
        key_data = composer.select_key_for_emotion(MusicEmotion.JOYFUL)
        assert key_data["note_name"]
        
        tempo = composer.select_tempo_for_emotion(MusicEmotion.JOYFUL)
        assert tempo > 100  # Joyful should be faster
        
        tempo_sad = composer.select_tempo_for_emotion(MusicEmotion.MELANCHOLIC)
        assert tempo_sad < 90  # Melancholic should be slower


class TestVisualArtist:
    """Test visual art generation system"""
    
    @pytest.fixture
    def artist(self):
        return VisualArtist()
    
    @pytest.mark.asyncio
    async def test_artwork_creation_abstract(self, artist):
        """Test abstract artwork creation"""
        artwork = await artist.create_artwork(
            concept="Flowing energy",
            style="abstract",
            size=(800, 600)
        )
        
        assert artwork is not None
        assert "artwork" in artwork
        assert "concept" in artwork
        assert "palette" in artwork
        assert "evaluation" in artwork
        
        # Check concept
        assert artwork["concept"]["theme"]
        assert artwork["concept"]["mood"]
        
        # Check palette
        assert len(artwork["palette"]["colors"]) > 0
        
        # Check evaluation
        assert 0.0 <= artwork["evaluation"]["overall_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_artwork_creation_realistic(self, artist):
        """Test realistic artwork creation"""
        artwork = await artist.create_artwork(
            concept="Mountain landscape",
            style="realistic",
            size=(1024, 768)
        )
        
        assert artwork is not None
        assert artwork["concept"]["mood"]
    
    @pytest.mark.asyncio
    async def test_conceptualization(self, artist):
        """Test concept visualization"""
        concept = await artist.conceptualize("Peaceful nature", ArtStyle.IMPRESSIONIST)
        
        assert concept.theme == "Peaceful nature"
        assert concept.mood in ["peaceful", "neutral", "joyful", "energetic", "melancholic", "mysterious"]
        assert len(concept.elements) > 0
    
    @pytest.mark.asyncio
    async def test_color_palette_selection(self, artist):
        """Test color palette selection"""
        from Core.Foundation.visual_artist import VisualConcept
        
        concept = VisualConcept(
            theme="Test",
            mood="peaceful",
            elements=["water"]
        )
        
        palette = await artist.select_color_palette(concept, ArtStyle.ABSTRACT)
        
        assert len(palette.colors) > 0
        assert palette.scheme
        assert all(0 <= c.r <= 255 and 0 <= c.g <= 255 and 0 <= c.b <= 255 for c in palette.colors)
    
    @pytest.mark.asyncio
    async def test_composition_design(self, artist):
        """Test composition design"""
        from Core.Foundation.visual_artist import VisualConcept
        
        concept = VisualConcept(
            theme="Test",
            mood="peaceful",
            elements=["form"]
        )
        
        composition = await artist.design_composition(concept, ArtStyle.MINIMALIST)
        
        assert composition.layout
        assert len(composition.focal_points) > 0
        assert composition.balance in ["symmetrical", "asymmetrical"]
        assert composition.depth_layers >= 2
    
    @pytest.mark.asyncio
    async def test_evaluation(self, artist):
        """Test artwork evaluation"""
        from Core.Foundation.visual_artist import (
            Artwork, VisualConcept, ColorPalette, Composition, Color
        )
        
        concept = VisualConcept(theme="Test", mood="peaceful", elements=["water"])
        palette = ColorPalette(colors=[Color(0, 0, 255, "Blue")], name="Test")
        composition = Composition(layout="centered", focal_points=[(0.5, 0.5)])
        
        artwork = Artwork(
            concept=concept,
            palette=palette,
            composition=composition,
            style=ArtStyle.ABSTRACT
        )
        
        evaluation = await artist.evaluate_artwork(artwork, concept)
        
        assert 0.0 <= evaluation.overall_score <= 1.0
        assert 0.0 <= evaluation.color_harmony <= 1.0
        assert 0.0 <= evaluation.composition_balance <= 1.0
        assert isinstance(evaluation.notes, list)


class TestIntegration:
    """Test integrated creative workflows"""
    
    @pytest.mark.asyncio
    async def test_multi_system_creation(self):
        """Test creating story, music, and art together"""
        theme = "A magical journey"
        
        # Generate story
        generator = StoryGenerator()
        story = await generator.generate_story(
            prompt=theme,
            style="fantasy",
            length="short"
        )
        assert story is not None
        
        # Compose music
        composer = MusicComposer()
        music = await composer.compose_music(
            emotion="peaceful",
            style="classical",
            duration_bars=4
        )
        assert music is not None
        
        # Create artwork
        artist = VisualArtist()
        artwork = await artist.create_artwork(
            concept=theme,
            style="impressionist",
            size=(800, 600)
        )
        assert artwork is not None
        
        # Verify all systems produced output
        assert story["meta"]["title"]
        assert music["analysis"]["tempo"] > 0
        assert artwork["evaluation"]["overall_score"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
