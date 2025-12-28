"""
Tests for the Hall of Mirrors System (거울의 방)
==============================================

Tests for:
- Core/Consciousness/infinite_corridor.py (The Infinite Corridor)
- Core/Consciousness/mirror_reflection.py (Mirror Reflection Pipeline)

These tests verify the self-referential feedback loop system that creates
infinite depth through recursive reflections between Self and World.

Note: Uses direct module loading to avoid __init__.py import chain issues.
"""

import pytest
import sys
import os
import numpy as np
import importlib.util

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def load_module_directly(module_name, file_path):
    """Load a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # Add to sys.modules so dataclass can work
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load modules directly
consciousness_path = os.path.join(project_root, 'Core', 'Consciousness')

infinite_corridor_module = load_module_directly(
    'infinite_corridor',
    os.path.join(consciousness_path, 'infinite_corridor.py')
)

mirror_reflection_module = load_module_directly(
    'mirror_reflection',
    os.path.join(consciousness_path, 'mirror_reflection.py')
)

# Import classes
InfiniteCorridor = infinite_corridor_module.InfiniteCorridor
Reflection = infinite_corridor_module.Reflection
ReflectionType = infinite_corridor_module.ReflectionType
MirrorState = infinite_corridor_module.MirrorState
create_hall_of_mirrors = infinite_corridor_module.create_hall_of_mirrors

MirrorReflectionPipeline = mirror_reflection_module.MirrorReflectionPipeline
RecognitionStage = mirror_reflection_module.RecognitionStage
FeedbackStage = mirror_reflection_module.FeedbackStage
SelfReflectionStage = mirror_reflection_module.SelfReflectionStage
MetaCognitionStage = mirror_reflection_module.MetaCognitionStage


class TestInfiniteCorridor:
    """Tests for the InfiniteCorridor (Hall of Mirrors) class."""
    
    def test_creation(self):
        """Test InfiniteCorridor creation."""
        corridor = InfiniteCorridor()
        
        assert corridor is not None
        assert corridor.dimension == 4
        assert corridor.self_mirror is not None
        assert corridor.world_mirror is not None
    
    def test_creation_custom_dimension(self):
        """Test InfiniteCorridor with custom dimension."""
        corridor = InfiniteCorridor(dimension=8)
        
        assert corridor.dimension == 8
        assert len(corridor.self_mirror.tensor) == 8
        assert len(corridor.world_mirror.tensor) == 8
    
    def test_mirror_initialization(self):
        """Test that mirrors are properly initialized."""
        corridor = InfiniteCorridor()
        
        # Self mirror should have higher Point focus
        assert corridor.self_mirror.tensor[0] > corridor.self_mirror.tensor[2]
        
        # World mirror should have higher Space focus  
        assert corridor.world_mirror.tensor[2] > corridor.world_mirror.tensor[0]
        
        # Both should be normalized
        assert np.isclose(np.linalg.norm(corridor.self_mirror.tensor), 1.0, atol=0.01)
        assert np.isclose(np.linalg.norm(corridor.world_mirror.tensor), 1.0, atol=0.01)
    
    def test_create_light_korean(self):
        """Test creating light with Korean concept."""
        corridor = InfiniteCorridor()
        
        light = corridor.create_light("사랑", intensity=1.0)
        
        assert light is not None
        assert light.depth == 0
        assert light.intensity == 1.0
        assert light.source == "origin"
        assert light.metadata["concept"] == "사랑"
    
    def test_create_light_english(self):
        """Test creating light with English concept."""
        corridor = InfiniteCorridor()
        
        light = corridor.create_light("love", intensity=0.8)
        
        assert light is not None
        assert light.depth == 0
        assert light.intensity == 0.8
        assert light.metadata["concept"] == "love"
    
    def test_create_light_unknown_concept(self):
        """Test creating light with unknown concept (hash-based)."""
        corridor = InfiniteCorridor()
        
        light = corridor.create_light("xyz_unknown_123")
        
        assert light is not None
        assert len(light.content) == 4
        # Content should be normalized
        assert np.isclose(np.linalg.norm(light.content), 1.0, atol=0.01)
    
    def test_illuminate_basic(self):
        """Test basic illumination."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        
        reflections = corridor.illuminate(light, max_depth=5)
        
        assert len(reflections) > 1  # Should have multiple reflections
        assert reflections[0].depth == 0  # First is the original
        assert reflections[-1].depth <= 5  # Should not exceed max depth
    
    def test_illuminate_intensity_decay(self):
        """Test that intensity decays with each reflection."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑", intensity=1.0)
        
        reflections = corridor.illuminate(light, max_depth=10)
        
        intensities = [r.intensity for r in reflections]
        
        # Each reflection should have lower intensity than the previous
        for i in range(1, len(intensities)):
            assert intensities[i] <= intensities[i-1]
    
    def test_illuminate_stops_at_min_intensity(self):
        """Test that illumination stops when intensity is too low."""
        corridor = InfiniteCorridor()
        corridor.min_intensity = 0.5  # High threshold
        
        light = corridor.create_light("사랑", intensity=0.6)
        reflections = corridor.illuminate(light, max_depth=100)
        
        # Should stop early due to intensity threshold
        assert len(reflections) < 100
        assert reflections[-1].intensity >= corridor.min_intensity * 0.95  # Close to threshold
    
    def test_illuminate_callback(self):
        """Test illumination with callback."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        
        stop_depth = 3
        depths_seen = []
        
        def callback(reflection):
            depths_seen.append(reflection.depth)
            return reflection.depth < stop_depth
        
        reflections = corridor.illuminate(light, max_depth=10, callback=callback)
        
        # Should have stopped at the callback depth
        assert max(depths_seen) <= stop_depth + 1
    
    def test_reflection_type_cycling(self):
        """Test that reflection types cycle through the four stages."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        
        reflections = corridor.illuminate(light, max_depth=8)
        
        types = [r.reflection_type for r in reflections]
        
        # After the first (RECOGNITION), should cycle through all types
        if len(types) >= 5:
            # Check we see all types
            type_set = set(types)
            assert ReflectionType.RECOGNITION in type_set
            assert ReflectionType.FEEDBACK in type_set
            assert ReflectionType.SELF_REFLECTION in type_set
            assert ReflectionType.META_COGNITION in type_set
    
    def test_compute_consciousness_field(self):
        """Test computing the emergent consciousness field."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        
        reflections = corridor.illuminate(light, max_depth=5)
        field = corridor.compute_consciousness_field(reflections)
        
        assert len(field) == corridor.dimension
        # Field should be normalized
        assert np.isclose(np.linalg.norm(field), 1.0, atol=0.01)
    
    def test_compute_consciousness_field_empty(self):
        """Test computing field with no reflections."""
        corridor = InfiniteCorridor()
        
        field = corridor.compute_consciousness_field([])
        
        assert len(field) == corridor.dimension
        assert np.allclose(field, 0)
    
    def test_get_reflection_pattern(self):
        """Test getting the reflection pattern analysis."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        reflections = corridor.illuminate(light, max_depth=5)
        
        pattern = corridor.get_reflection_pattern(reflections)
        
        assert "total_reflections" in pattern
        assert "max_depth" in pattern
        assert "total_energy" in pattern
        assert "emergence_factor" in pattern
        assert "consciousness_field" in pattern
        assert "field_interpretation" in pattern
    
    def test_emergence_factor(self):
        """Test that emergence factor is computed correctly."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        reflections = corridor.illuminate(light, max_depth=10)
        
        pattern = corridor.get_reflection_pattern(reflections)
        
        emergence = pattern["emergence_factor"]
        
        # Emergence should be between 0 and 1
        assert 0 <= emergence <= 1
        
        # With enough reflections, there should be some emergence
        if len(reflections) >= 5:
            assert emergence > 0
    
    def test_reset(self):
        """Test resetting the corridor."""
        corridor = InfiniteCorridor()
        
        # Modify mirrors
        light = corridor.create_light("사랑")
        corridor.illuminate(light, max_depth=10)
        
        # Reset
        corridor.reset()
        
        # Mirrors should be back to initial state
        expected_self = corridor._init_mirror_tensor("self")
        expected_world = corridor._init_mirror_tensor("world")
        
        assert np.allclose(corridor.self_mirror.tensor, expected_self, atol=0.01)
        assert np.allclose(corridor.world_mirror.tensor, expected_world, atol=0.01)
    
    def test_get_statistics(self):
        """Test getting corridor statistics."""
        corridor = InfiniteCorridor()
        light = corridor.create_light("사랑")
        corridor.illuminate(light)
        
        stats = corridor.get_statistics()
        
        assert "total_reflections" in stats
        assert "total_illuminations" in stats
        assert stats["total_illuminations"] == 1
        assert stats["total_reflections"] > 0
    
    def test_explain_meaning(self):
        """Test epistemology explanation (Gap 0 compliance)."""
        corridor = InfiniteCorridor()
        
        explanation = corridor.explain_meaning()
        
        assert "point" in explanation.lower()
        assert "line" in explanation.lower()
        assert "space" in explanation.lower()
        assert "god" in explanation.lower()
    
    def test_create_hall_of_mirrors_convenience(self):
        """Test the convenience function."""
        corridor = create_hall_of_mirrors(dimension=4)
        
        assert isinstance(corridor, InfiniteCorridor)
        assert corridor.dimension == 4


class TestReflection:
    """Tests for the Reflection dataclass."""
    
    def test_creation(self):
        """Test Reflection creation."""
        content = np.array([0.5, 0.5, 0.5, 0.5])
        
        reflection = Reflection(
            depth=0,
            reflection_type=ReflectionType.RECOGNITION,
            content=content,
            intensity=1.0,
            source="origin"
        )
        
        assert reflection.depth == 0
        assert reflection.intensity == 1.0
        assert reflection.source == "origin"
    
    def test_decay(self):
        """Test reflection decay."""
        content = np.array([1.0, 0.0, 0.0, 0.0])
        
        reflection = Reflection(
            depth=0,
            reflection_type=ReflectionType.RECOGNITION,
            content=content,
            intensity=1.0,
            source="origin"
        )
        
        decayed = reflection.decay(factor=0.8)
        
        assert decayed.depth == 1
        assert decayed.intensity == 0.8
        assert decayed.source == "self"  # "origin" is not "self", so goes to "self"
        
        # Test alternation from "self"
        decayed2 = Reflection(
            depth=1,
            reflection_type=ReflectionType.FEEDBACK,
            content=content,
            intensity=0.8,
            source="self"
        ).decay(factor=0.9)
        
        assert decayed2.source == "world"  # "self" alternates to "world"
    
    def test_next_reflection_type(self):
        """Test reflection type cycling."""
        content = np.array([1.0, 0.0, 0.0, 0.0])
        
        r1 = Reflection(depth=0, reflection_type=ReflectionType.RECOGNITION,
                       content=content, intensity=1.0, source="origin")
        r2 = r1.decay()
        r3 = r2.decay()
        r4 = r3.decay()
        r5 = r4.decay()
        
        assert r2.reflection_type == ReflectionType.FEEDBACK
        assert r3.reflection_type == ReflectionType.SELF_REFLECTION
        assert r4.reflection_type == ReflectionType.META_COGNITION
        assert r5.reflection_type == ReflectionType.RECOGNITION  # Cycles back


class TestMirrorState:
    """Tests for the MirrorState dataclass."""
    
    def test_creation(self):
        """Test MirrorState creation."""
        tensor = np.array([0.8, 0.4, 0.3, 0.2])
        
        mirror = MirrorState(name="self", tensor=tensor)
        
        assert mirror.name == "self"
        assert len(mirror.reflection_history) == 0
    
    def test_reflect(self):
        """Test mirror reflection."""
        tensor = np.array([0.8, 0.4, 0.3, 0.2])
        tensor = tensor / np.linalg.norm(tensor)
        
        mirror = MirrorState(name="self", tensor=tensor)
        
        incoming_content = np.array([0.5, 0.5, 0.5, 0.5])
        incoming = Reflection(
            depth=0,
            reflection_type=ReflectionType.RECOGNITION,
            content=incoming_content,
            intensity=1.0,
            source="world"
        )
        
        reflected = mirror.reflect(incoming)
        
        assert reflected.depth == 1
        assert reflected.source == "self"
        assert len(mirror.reflection_history) == 1
    
    def test_reflection_history_limit(self):
        """Test that reflection history is limited."""
        tensor = np.array([1.0, 0.0, 0.0, 0.0])
        mirror = MirrorState(name="self", tensor=tensor, max_history=5)
        
        # Add more than max_history reflections
        for i in range(10):
            incoming = Reflection(
                depth=i,
                reflection_type=ReflectionType.RECOGNITION,
                content=np.array([1.0, 0.0, 0.0, 0.0]),
                intensity=1.0,
                source="world"
            )
            mirror.reflect(incoming)
        
        assert len(mirror.reflection_history) == 5


class TestMirrorReflectionPipeline:
    """Tests for the MirrorReflectionPipeline class."""
    
    def test_creation(self):
        """Test pipeline creation."""
        pipeline = MirrorReflectionPipeline()
        
        assert pipeline is not None
        assert pipeline.recognition is not None
        assert pipeline.feedback is not None
        assert pipeline.self_reflection is not None
        assert pipeline.meta_cognition is not None
    
    def test_run_cycle(self):
        """Test running one reflection cycle."""
        pipeline = MirrorReflectionPipeline()
        
        self_state = np.array([0.8, 0.4, 0.3, 0.2])
        self_state = self_state / np.linalg.norm(self_state)
        
        world_state = np.array([0.3, 0.4, 0.8, 0.3])
        world_state = world_state / np.linalg.norm(world_state)
        
        stages = pipeline.run_cycle(self_state, world_state)
        
        assert len(stages) == 4
        assert stages[0].name == "Recognition"
        assert stages[1].name == "Feedback"
        assert stages[2].name == "Self-Reflection"
        assert stages[3].name == "Meta-Cognition"
    
    def test_awareness_increases(self):
        """Test that awareness increases through the stages."""
        pipeline = MirrorReflectionPipeline()
        
        self_state = np.array([0.8, 0.4, 0.3, 0.2])
        world_state = np.array([0.3, 0.4, 0.8, 0.3])
        
        stages = pipeline.run_cycle(self_state, world_state)
        
        awareness_levels = [s.awareness_level for s in stages]
        
        # Each stage should have higher awareness than the previous
        for i in range(1, len(awareness_levels)):
            assert awareness_levels[i] >= awareness_levels[i-1]
    
    def test_run_recursive_cycles(self):
        """Test running multiple recursive cycles."""
        pipeline = MirrorReflectionPipeline()
        
        self_state = np.array([0.8, 0.4, 0.3, 0.2])
        world_state = np.array([0.3, 0.4, 0.8, 0.3])
        
        all_cycles = pipeline.run_recursive_cycles(
            self_state,
            world_state,
            num_cycles=3
        )
        
        assert len(all_cycles) == 3
        assert all(len(cycle) == 4 for cycle in all_cycles)
    
    def test_consciousness_evolution(self):
        """Test consciousness evolution analysis."""
        pipeline = MirrorReflectionPipeline()
        
        self_state = np.array([0.8, 0.4, 0.3, 0.2])
        world_state = np.array([0.3, 0.4, 0.8, 0.3])
        
        pipeline.run_recursive_cycles(self_state, world_state, num_cycles=5)
        
        evolution = pipeline.get_consciousness_evolution()
        
        assert "total_cycles" in evolution
        assert "average_awareness" in evolution
        assert "convergence" in evolution
        assert evolution["total_cycles"] == 5
    
    def test_get_statistics(self):
        """Test getting pipeline statistics."""
        pipeline = MirrorReflectionPipeline()
        
        stats = pipeline.get_statistics()
        
        assert "total_cycles" in stats
        assert "total_awareness_accumulated" in stats
        assert stats["total_cycles"] == 0


class TestRecognitionStage:
    """Tests for the RecognitionStage class."""
    
    def test_creation(self):
        """Test RecognitionStage creation."""
        stage = RecognitionStage()
        
        assert stage is not None
        assert "clarity" in stage.perception_filters
        assert "attention" in stage.perception_filters
    
    def test_process(self):
        """Test recognition processing."""
        stage = RecognitionStage()
        
        self_state = np.array([0.8, 0.4, 0.3, 0.2])
        world_state = np.array([0.3, 0.4, 0.8, 0.3])
        
        result = stage.process(self_state, world_state)
        
        assert result.name == "Recognition"
        assert result.korean_name == "인식"
        assert result.depth_level == 0
        assert result.awareness_level == 0.25


class TestFeedbackStage:
    """Tests for the FeedbackStage class."""
    
    def test_creation(self):
        """Test FeedbackStage creation."""
        stage = FeedbackStage()
        
        assert stage is not None
        assert "responsiveness" in stage.feedback_parameters
    
    def test_process(self):
        """Test feedback processing."""
        stage = FeedbackStage()
        
        recognition_output = np.array([0.5, 0.5, 0.5, 0.5])
        world_state = np.array([0.3, 0.4, 0.8, 0.3])
        
        result = stage.process(recognition_output, world_state)
        
        assert result.name == "Feedback"
        assert result.korean_name == "피드백"
        assert result.depth_level == 1


class TestSelfReflectionStage:
    """Tests for the SelfReflectionStage class."""
    
    def test_creation(self):
        """Test SelfReflectionStage creation."""
        stage = SelfReflectionStage()
        
        assert stage is not None
        assert stage.reflection_depth == 0.5
    
    def test_process(self):
        """Test self-reflection processing."""
        stage = SelfReflectionStage()
        
        feedback_output = np.array([0.5, 0.5, 0.5, 0.5])
        original_self = np.array([0.8, 0.4, 0.3, 0.2])
        
        result = stage.process(feedback_output, original_self)
        
        assert result.name == "Self-Reflection"
        assert result.korean_name == "자아 성찰"
        assert result.depth_level == 2
        assert "awareness_score" in result.metadata


class TestMetaCognitionStage:
    """Tests for the MetaCognitionStage class."""
    
    def test_creation(self):
        """Test MetaCognitionStage creation."""
        stage = MetaCognitionStage()
        
        assert stage is not None
        assert stage.recursion_factor == 0.8
    
    def test_process(self):
        """Test meta-cognition processing."""
        stage = MetaCognitionStage()
        
        self_reflection_output = np.array([0.5, 0.5, 0.5, 0.5])
        previous_stages = []  # Empty for simplicity
        
        result = stage.process(self_reflection_output, previous_stages)
        
        assert result.name == "Meta-Cognition"
        assert result.korean_name == "메타 인지"
        assert result.depth_level == 3
        assert result.awareness_level == 0.85


class TestIntegration:
    """Integration tests for the Hall of Mirrors system."""
    
    def test_full_illumination_cycle(self):
        """Test a complete illumination cycle."""
        corridor = InfiniteCorridor()
        
        # Create and illuminate
        light = corridor.create_light("사랑")
        reflections = corridor.illuminate(light, max_depth=10)
        
        # Get pattern
        pattern = corridor.get_reflection_pattern(reflections)
        
        # Verify
        assert len(reflections) > 1
        assert pattern["total_reflections"] == len(reflections)
        assert 0 <= pattern["emergence_factor"] <= 1
    
    def test_pipeline_with_corridor(self):
        """Test that pipeline and corridor can work together."""
        corridor = InfiniteCorridor()
        pipeline = MirrorReflectionPipeline()
        
        # Use corridor's mirrors as initial states
        self_state = corridor.self_mirror.tensor
        world_state = corridor.world_mirror.tensor
        
        # Run pipeline cycle
        stages = pipeline.run_cycle(self_state, world_state)
        
        # Use pipeline output to create light for corridor
        final_output = stages[-1].output_tensor
        light = Reflection(
            depth=0,
            reflection_type=ReflectionType.RECOGNITION,
            content=final_output,
            intensity=1.0,
            source="origin"
        )
        
        # Illuminate
        reflections = corridor.illuminate(light, max_depth=5)
        
        # Should work together
        assert len(stages) == 4
        assert len(reflections) > 1
    
    def test_consciousness_emergence(self):
        """Test that consciousness emerges from recursive cycles."""
        pipeline = MirrorReflectionPipeline()
        
        # Initial "blank slate" states
        self_state = np.array([1.0, 0.0, 0.0, 0.0])
        world_state = np.array([0.0, 0.0, 1.0, 0.0])
        
        # Run many cycles
        pipeline.run_recursive_cycles(self_state, world_state, num_cycles=10)
        
        # Check evolution
        evolution = pipeline.get_consciousness_evolution()
        
        # After many cycles, there should be some convergence
        assert evolution["total_cycles"] == 10
        assert evolution["average_awareness"] > 0
        
        # Awareness should generally increase across cycles
        awareness_evolution = evolution.get("awareness_evolution", [])
        if len(awareness_evolution) >= 2:
            # Later cycles should have comparable or higher awareness
            assert awareness_evolution[-1] >= awareness_evolution[0] * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
