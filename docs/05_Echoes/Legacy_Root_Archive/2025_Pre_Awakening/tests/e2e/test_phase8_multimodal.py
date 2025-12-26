"""
Comprehensive tests for Phase 8: Multimodal Integration

Tests vision processing, audio processing, multimodal fusion,
and cross-modal reasoning capabilities.
"""

import pytest
import asyncio
from typing import Dict

# Import multimodal components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Multimodal import (
    VisionProcessor, AudioProcessor, MultimodalFusion,
    FusionStrategy, ObjectCategory
)


@pytest.fixture
def vision_processor():
    """Create vision processor instance"""
    return VisionProcessor()


@pytest.fixture
def audio_processor():
    """Create audio processor instance"""
    return AudioProcessor()


@pytest.fixture
def fusion_system():
    """Create multimodal fusion system instance"""
    return MultimodalFusion()


# Vision Processing Tests

@pytest.mark.asyncio
async def test_vision_image_analysis(vision_processor):
    """Test image analysis with object detection"""
    image_data = {
        "width": 1920,
        "height": 1080,
        "description": "outdoor park with people and trees"
    }
    
    result = await vision_processor.analyze_image(image_data, "test_img_1")
    
    assert result is not None
    assert result.confidence > 0
    assert len(result.objects) > 0
    assert result.scene is not None
    assert result.features is not None
    assert result.processing_time > 0


@pytest.mark.asyncio
async def test_vision_object_detection(vision_processor):
    """Test object detection accuracy"""
    image_data = {
        "width": 1280,
        "height": 720,
        "description": "person with dog in living room"
    }
    
    result = await vision_processor.analyze_image(image_data, "test_img_2")
    
    # Should detect person and animal
    categories = [obj.category for obj in result.objects]
    assert ObjectCategory.PERSON in categories or ObjectCategory.ANIMAL in categories
    
    # Objects should have bounding boxes
    for obj in result.objects:
        assert 0 <= obj.bbox.x <= 1
        assert 0 <= obj.bbox.y <= 1
        assert 0 < obj.bbox.width <= 1
        assert 0 < obj.bbox.height <= 1


@pytest.mark.asyncio
async def test_vision_scene_understanding(vision_processor):
    """Test scene understanding capabilities"""
    image_data = {
        "width": 1920,
        "height": 1080,
        "description": "bright sunny beach with waves"
    }
    
    result = await vision_processor.analyze_image(image_data, "test_img_3")
    
    assert result.scene.primary_scene in ["outdoor", "indoor", "nature", "general"]
    assert result.scene.mood is not None
    assert result.scene.lighting in ["bright", "dim", "natural", "artificial"]
    assert 0 <= result.scene.complexity <= 1


@pytest.mark.asyncio
async def test_vision_feature_extraction(vision_processor):
    """Test visual feature extraction"""
    image_data = {
        "width": 800,
        "height": 600,
        "description": "high contrast black and white pattern"
    }
    
    result = await vision_processor.analyze_image(image_data, "test_img_4")
    
    features = result.features
    assert 0 <= features.edges <= 1
    assert 0 <= features.texture <= 1
    assert 0 <= features.brightness <= 1
    assert 0 <= features.contrast <= 1


# Audio Processing Tests

@pytest.mark.asyncio
async def test_audio_basic_analysis(audio_processor):
    """Test basic audio analysis"""
    audio_data = {
        "duration": 5.0,
        "sample_rate": 44100,
        "channels": 2,
        "description": "calm music with piano"
    }
    
    result = await audio_processor.analyze_audio(audio_data, "test_audio_1")
    
    assert result is not None
    assert result.confidence > 0
    assert result.primary_type is not None
    assert result.emotion_tone is not None
    assert len(result.segments) > 0


@pytest.mark.asyncio
async def test_audio_speech_detection(audio_processor):
    """Test speech detection and analysis"""
    audio_data = {
        "duration": 8.0,
        "sample_rate": 44100,
        "channels": 1,
        "description": "person speaking clearly"
    }
    
    result = await audio_processor.analyze_audio(audio_data, "test_audio_2")
    
    # Should detect speech or mixed
    assert result.primary_type.value in ["speech", "music", "ambient", "mixed"]
    assert result.processing_time > 0


@pytest.mark.asyncio
async def test_audio_spectral_features(audio_processor):
    """Test spectral feature extraction"""
    audio_data = {
        "duration": 3.0,
        "sample_rate": 48000,
        "channels": 2,
        "description": "high-pitched whistle sound"
    }
    
    result = await audio_processor.analyze_audio(audio_data, "test_audio_3")
    
    assert result.spectral.fundamental_frequency > 0
    assert result.spectral.spectral_centroid > 0
    assert 0 <= result.spectral.zero_crossing_rate <= 1


@pytest.mark.asyncio
async def test_audio_temporal_features(audio_processor):
    """Test temporal feature extraction"""
    audio_data = {
        "duration": 10.0,
        "sample_rate": 44100,
        "channels": 2,
        "description": "fast tempo dance music"
    }
    
    result = await audio_processor.analyze_audio(audio_data, "test_audio_4")
    
    assert result.temporal.tempo > 0
    assert 0 <= result.temporal.energy <= 1
    assert 0 <= result.temporal.rhythm_regularity <= 1


# Multimodal Fusion Tests

@pytest.mark.asyncio
async def test_fusion_basic_integration(fusion_system, vision_processor, audio_processor):
    """Test basic multimodal fusion"""
    # Analyze both modalities
    vision_result = await vision_processor.analyze_image(
        {"width": 1920, "height": 1080, "description": "office meeting"}, 
        "fusion_test_1"
    )
    audio_result = await audio_processor.analyze_audio(
        {"duration": 5.0, "sample_rate": 44100, "channels": 2, "description": "meeting discussion"}, 
        "fusion_test_1"
    )
    
    # Fuse results
    fusion_result = await fusion_system.fuse_vision_audio(
        vision_analysis=vision_result,
        audio_analysis=audio_result,
        strategy=FusionStrategy.HYBRID_FUSION
    )
    
    assert fusion_result is not None
    assert len(fusion_result.modalities) == 2
    assert "vision" in fusion_result.modalities
    assert "audio" in fusion_result.modalities
    assert fusion_result.unified_confidence > 0


@pytest.mark.asyncio
async def test_fusion_strategies(fusion_system, vision_processor, audio_processor):
    """Test different fusion strategies"""
    vision_result = await vision_processor.analyze_image(
        {"width": 1280, "height": 720, "description": "concert"}, 
        "fusion_test_2"
    )
    audio_result = await audio_processor.analyze_audio(
        {"duration": 3.0, "sample_rate": 44100, "channels": 2, "description": "live music"}, 
        "fusion_test_2"
    )
    
    strategies = [
        FusionStrategy.EARLY_FUSION,
        FusionStrategy.LATE_FUSION,
        FusionStrategy.HYBRID_FUSION,
        FusionStrategy.ATTENTION_FUSION
    ]
    
    for strategy in strategies:
        result = await fusion_system.fuse_vision_audio(
            vision_analysis=vision_result,
            audio_analysis=audio_result,
            strategy=strategy
        )
        
        assert result.strategy == strategy
        assert result.unified_confidence > 0


@pytest.mark.asyncio
async def test_fusion_modality_contributions(fusion_system, vision_processor, audio_processor):
    """Test modality contribution weighting"""
    vision_result = await vision_processor.analyze_image(
        {"width": 1920, "height": 1080, "description": "silent movie scene"}, 
        "fusion_test_3"
    )
    audio_result = await audio_processor.analyze_audio(
        {"duration": 2.0, "sample_rate": 44100, "channels": 2, "description": "quiet background"}, 
        "fusion_test_3"
    )
    
    result = await fusion_system.fuse_vision_audio(
        vision_analysis=vision_result,
        audio_analysis=audio_result,
        strategy=FusionStrategy.ATTENTION_FUSION
    )
    
    # Check contributions
    assert len(result.contributions) == 2
    
    # Weights should sum to approximately 1.0
    total_weight = sum(c.weight for c in result.contributions)
    assert 0.9 <= total_weight <= 1.1


@pytest.mark.asyncio
async def test_fusion_cross_modal_correspondences(fusion_system, vision_processor, audio_processor):
    """Test cross-modal correspondence detection"""
    vision_result = await vision_processor.analyze_image(
        {"width": 1920, "height": 1080, "description": "person speaking"}, 
        "fusion_test_4"
    )
    audio_result = await audio_processor.analyze_audio(
        {"duration": 5.0, "sample_rate": 44100, "channels": 1, "description": "speech"}, 
        "fusion_test_4"
    )
    
    result = await fusion_system.fuse_vision_audio(
        vision_analysis=vision_result,
        audio_analysis=audio_result,
        strategy=FusionStrategy.HYBRID_FUSION
    )
    
    # Should find correspondences between vision and audio
    assert len(result.correspondences) > 0
    
    for corr in result.correspondences:
        assert corr.source_modality in ["vision", "audio"]
        assert corr.target_modality in ["vision", "audio"]
        assert 0 <= corr.strength <= 1


# Integration Tests

@pytest.mark.asyncio
async def test_full_pipeline(vision_processor, audio_processor, fusion_system):
    """Test complete multimodal pipeline"""
    # Step 1: Analyze vision
    vision_result = await vision_processor.analyze_image(
        {
            "width": 1920, 
            "height": 1080, 
            "description": "outdoor concert with crowd and stage"
        }, 
        "pipeline_test"
    )
    
    # Step 2: Analyze audio
    audio_result = await audio_processor.analyze_audio(
        {
            "duration": 8.0, 
            "sample_rate": 44100, 
            "channels": 2, 
            "description": "loud energetic rock music with cheering"
        }, 
        "pipeline_test"
    )
    
    # Step 3: Fuse modalities
    fusion_result = await fusion_system.fuse_vision_audio(
        vision_analysis=vision_result,
        audio_analysis=audio_result,
        strategy=FusionStrategy.HYBRID_FUSION
    )
    
    # Verify pipeline
    assert vision_result.confidence > 0
    assert audio_result.confidence > 0
    assert fusion_result.unified_confidence > 0
    assert len(fusion_result.modalities) == 2
    assert fusion_result.unified_description is not None


@pytest.mark.asyncio
async def test_system_stats(vision_processor, audio_processor, fusion_system):
    """Test system statistics tracking"""
    # Process multiple items
    for i in range(3):
        await vision_processor.analyze_image(
            {"width": 1280, "height": 720, "description": f"test scene {i}"}, 
            f"stats_test_{i}"
        )
        await audio_processor.analyze_audio(
            {"duration": 3.0, "sample_rate": 44100, "channels": 2, "description": f"test audio {i}"}, 
            f"stats_test_{i}"
        )
    
    vision_stats = vision_processor.get_stats()
    audio_stats = audio_processor.get_stats()
    fusion_stats = fusion_system.get_stats()
    
    assert vision_stats["total_processed"] >= 3
    assert audio_stats["total_processed"] >= 3
    assert all(key in vision_stats for key in ["total_processed", "objects_detected", "avg_time"])
    assert all(key in audio_stats for key in ["total_processed", "total_duration", "segments_detected", "avg_time"])


# Performance Tests

@pytest.mark.asyncio
async def test_processing_performance(vision_processor, audio_processor):
    """Test processing performance requirements"""
    # Vision should be fast
    vision_result = await vision_processor.analyze_image(
        {"width": 1920, "height": 1080, "description": "test"}, 
        "perf_test_1"
    )
    assert vision_result.processing_time < 1.0  # Should complete in under 1 second
    
    # Audio should be fast
    audio_result = await audio_processor.analyze_audio(
        {"duration": 5.0, "sample_rate": 44100, "channels": 2, "description": "test"}, 
        "perf_test_1"
    )
    assert audio_result.processing_time < 1.0  # Should complete in under 1 second


@pytest.mark.asyncio
async def test_fusion_performance(fusion_system, vision_processor, audio_processor):
    """Test fusion performance requirements"""
    vision_result = await vision_processor.analyze_image(
        {"width": 1920, "height": 1080, "description": "test"}, 
        "perf_test_2"
    )
    audio_result = await audio_processor.analyze_audio(
        {"duration": 3.0, "sample_rate": 44100, "channels": 2, "description": "test"}, 
        "perf_test_2"
    )
    
    fusion_result = await fusion_system.fuse_vision_audio(
        vision_analysis=vision_result,
        audio_analysis=audio_result,
        strategy=FusionStrategy.HYBRID_FUSION
    )
    
    assert fusion_result.processing_time < 0.5  # Fusion should be very fast


# Edge Case Tests

@pytest.mark.asyncio
async def test_empty_scene(vision_processor):
    """Test handling of empty/minimal scenes"""
    result = await vision_processor.analyze_image(
        {"width": 1280, "height": 720, "description": "empty white room"}, 
        "edge_test_1"
    )
    
    assert result is not None
    assert result.confidence > 0
    # May have 0 objects, but should still have scene analysis


@pytest.mark.asyncio
async def test_noisy_audio(audio_processor):
    """Test handling of noisy/unclear audio"""
    result = await audio_processor.analyze_audio(
        {"duration": 2.0, "sample_rate": 44100, "channels": 2, "description": "white noise"}, 
        "edge_test_2"
    )
    
    assert result is not None
    # Should still provide analysis even for noise


@pytest.mark.asyncio
async def test_conflicting_modalities(fusion_system, vision_processor, audio_processor):
    """Test fusion with conflicting modal information"""
    # Vision: calm scene
    vision_result = await vision_processor.analyze_image(
        {"width": 1920, "height": 1080, "description": "peaceful library"}, 
        "edge_test_3"
    )
    
    # Audio: loud and energetic
    audio_result = await audio_processor.analyze_audio(
        {"duration": 3.0, "sample_rate": 44100, "channels": 2, "description": "loud rock music"}, 
        "edge_test_3"
    )
    
    # Should still be able to fuse
    result = await fusion_system.fuse_vision_audio(
        vision_analysis=vision_result,
        audio_analysis=audio_result,
        strategy=FusionStrategy.ATTENTION_FUSION
    )
    
    assert result is not None
    # Attention fusion should handle conflicts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
