"""
Phase 8 Multimodal Integration Demo

Demonstrates vision processing, audio processing, and multimodal fusion capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Multimodal import (
    VisionProcessor, AudioProcessor, MultimodalFusion,
    FusionStrategy
)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


async def demo_vision_processing():
    """Demonstrate vision processing"""
    print_section("Vision Processing Demo")
    
    processor = VisionProcessor()
    
    # Simulate analyzing an image
    image_data = {
        "width": 1920,
        "height": 1080,
        "description": "outdoor park with people and dog, bright sunny day"
    }
    
    print("\n📸 Analyzing image...")
    print(f"Description: {image_data['description']}")
    
    analysis = await processor.analyze_image(image_data, "demo_image_1")
    
    print(f"\n✅ Analysis complete in {analysis.processing_time:.3f}s")
    print(f"Confidence: {analysis.confidence:.2%}")
    
    print(f"\n🎯 Detected Objects ({len(analysis.objects)}):")
    for obj in analysis.objects:
        print(f"  - {obj.label} ({obj.category.value}): {obj.confidence:.2%}")
        print(f"    Position: ({obj.bbox.x:.2f}, {obj.bbox.y:.2f})")
    
    print(f"\n🌅 Scene Analysis:")
    print(f"  Type: {analysis.scene.primary_scene}")
    print(f"  Mood: {analysis.scene.mood}")
    print(f"  Lighting: {analysis.scene.lighting}")
    print(f"  Complexity: {analysis.scene.complexity:.2f}")
    print(f"  Dominant Colors: {', '.join(analysis.scene.dominant_colors)}")
    
    print(f"\n🔍 Visual Features:")
    print(f"  Edges: {analysis.features.edges:.2f}")
    print(f"  Texture: {analysis.features.texture:.2f}")
    print(f"  Brightness: {analysis.features.brightness:.2f}")
    print(f"  Contrast: {analysis.features.contrast:.2f}")
    
    stats = processor.get_stats()
    print(f"\n📊 Processing Stats:")
    print(f"  Total Images: {stats['total_processed']}")
    print(f"  Objects Detected: {stats['objects_detected']}")
    print(f"  Avg Processing Time: {stats['avg_time']:.3f}s")
    
    return analysis


async def demo_audio_processing():
    """Demonstrate audio processing"""
    print_section("Audio Processing Demo")
    
    processor = AudioProcessor()
    
    # Simulate analyzing audio
    audio_data = {
        "duration": 8.5,
        "sample_rate": 44100,
        "channels": 2,
        "description": "happy energetic music with strong rhythm"
    }
    
    print("\n🎵 Analyzing audio...")
    print(f"Description: {audio_data['description']}")
    print(f"Duration: {audio_data['duration']}s")
    
    analysis = await processor.analyze_audio(audio_data, "demo_audio_1")
    
    print(f"\n✅ Analysis complete in {analysis.processing_time:.3f}s")
    print(f"Confidence: {analysis.confidence:.2%}")
    
    print(f"\n🎧 Audio Type: {analysis.primary_type.value}")
    print(f"Emotion: {analysis.emotion_tone.value}")
    
    print(f"\n📋 Segments ({len(analysis.segments)}):")
    for i, seg in enumerate(analysis.segments, 1):
        print(f"  {i}. {seg.start_time:.1f}s-{seg.end_time:.1f}s: {seg.audio_type.value} ({seg.confidence:.2%})")
        if seg.transcription:
            print(f"     Transcription: \"{seg.transcription}\"")
    
    print(f"\n🌊 Spectral Features:")
    print(f"  Fundamental Frequency: {analysis.spectral.fundamental_frequency:.0f} Hz")
    print(f"  Spectral Centroid: {analysis.spectral.spectral_centroid:.0f} Hz")
    print(f"  Zero Crossing Rate: {analysis.spectral.zero_crossing_rate:.2f}")
    
    print(f"\n⏱️ Temporal Features:")
    if analysis.temporal.tempo:
        print(f"  Tempo: {analysis.temporal.tempo:.0f} BPM")
    print(f"  Energy: {analysis.temporal.energy:.2f}")
    print(f"  Rhythm Regularity: {analysis.temporal.rhythm_regularity:.2f}")
    print(f"  Dynamic Range: {analysis.temporal.dynamic_range:.2f}")
    
    stats = processor.get_stats()
    print(f"\n📊 Processing Stats:")
    print(f"  Total Audio Clips: {stats['total_processed']}")
    print(f"  Total Duration: {stats['total_duration']:.1f}s")
    print(f"  Segments Detected: {stats['segments_detected']}")
    print(f"  Avg Processing Time: {stats['avg_time']:.3f}s")
    
    return analysis


async def demo_multimodal_fusion():
    """Demonstrate multimodal fusion"""
    print_section("Multimodal Fusion Demo")
    
    fusion_system = MultimodalFusion()
    
    # Create coordinated vision and audio data
    vision_data = {
        "width": 1920,
        "height": 1080,
        "description": "calm indoor office with person working, natural lighting"
    }
    
    audio_data = {
        "duration": 5.0,
        "sample_rate": 44100,
        "channels": 1,
        "description": "calm ambient office noise with occasional speech"
    }
    
    print("\n🔄 Fusing vision and audio modalities...")
    print(f"Vision: {vision_data['description']}")
    print(f"Audio: {audio_data['description']}")
    
    result = await fusion_system.fuse_vision_audio(
        vision_data=vision_data,
        audio_data=audio_data,
        strategy=FusionStrategy.HYBRID_FUSION
    )
    
    print(f"\n✅ Fusion complete in {result.processing_time:.3f}s")
    print(f"Strategy: {result.strategy.value}")
    print(f"Unified Confidence: {result.unified_confidence:.2%}")
    
    print(f"\n🎯 Modality Contributions:")
    for contrib in result.contributions:
        print(f"\n  {contrib.modality.upper()} (weight: {contrib.weight:.2%}):")
        print(f"    Confidence: {contrib.confidence:.2%}")
        for insight in contrib.key_insights:
            print(f"    - {insight}")
    
    print(f"\n🔗 Cross-Modal Correspondences:")
    for corr in result.correspondences:
        print(f"\n  {corr.correspondence_type.replace('_', ' ').title()}:")
        print(f"    {corr.source_modality} ↔ {corr.target_modality}")
        print(f"    Strength: {corr.strength:.2%}")
        print(f"    {corr.description}")
    
    print(f"\n📝 Unified Description:")
    print(f"  {result.unified_description}")
    
    print(f"\n💡 Insights:")
    for key, value in result.insights.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    stats = fusion_system.get_stats()
    print(f"\n📊 Fusion Stats:")
    print(f"  Total Fusions: {stats['total_fusions']}")
    print(f"  Avg Processing Time: {stats['avg_time']:.3f}s")
    print(f"  Modality Combinations:")
    for combo, count in stats['modality_combinations'].items():
        print(f"    {combo}: {count}")


async def demo_multiple_scenarios():
    """Demonstrate multiple fusion scenarios"""
    print_section("Multiple Scenario Demo")
    
    fusion_system = MultimodalFusion()
    
    scenarios = [
        {
            "name": "Concert",
            "vision": {
                "width": 1920, "height": 1080,
                "description": "energetic crowded concert with bright stage lights"
            },
            "audio": {
                "duration": 10.0, "sample_rate": 44100, "channels": 2,
                "description": "excited energetic music with strong rhythm and crowd noise"
            }
        },
        {
            "name": "Nature Walk",
            "vision": {
                "width": 1920, "height": 1080,
                "description": "calm outdoor forest with trees and flowers, natural lighting"
            },
            "audio": {
                "duration": 7.0, "sample_rate": 44100, "channels": 1,
                "description": "calm peaceful ambient nature sounds with bird songs"
            }
        },
        {
            "name": "News Broadcast",
            "vision": {
                "width": 1920, "height": 1080,
                "description": "indoor office with person presenting, professional lighting, text on screen"
            },
            "audio": {
                "duration": 15.0, "sample_rate": 44100, "channels": 1,
                "description": "neutral professional speech with clear articulation"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📺 Scenario: {scenario['name']}")
        print(f"  Vision: {scenario['vision']['description']}")
        print(f"  Audio: {scenario['audio']['description']}")
        
        result = await fusion_system.fuse_vision_audio(
            vision_data=scenario['vision'],
            audio_data=scenario['audio']
        )
        
        print(f"  Unified: {result.unified_description}")
        print(f"  Confidence: {result.unified_confidence:.2%}")
        print(f"  Alignment: {result.insights.get('modality_alignment', 'N/A')}")


async def main():
    """Run all demonstrations"""
    print("\n" + "🌈"*30)
    print(" PHASE 8: MULTIMODAL INTEGRATION DEMO")
    print("🌈"*30)
    
    try:
        # Demo 1: Vision Processing
        vision_analysis = await demo_vision_processing()
        
        # Demo 2: Audio Processing  
        audio_analysis = await demo_audio_processing()
        
        # Demo 3: Multimodal Fusion
        await demo_multimodal_fusion()
        
        # Demo 4: Multiple Scenarios
        await demo_multiple_scenarios()
        
        print_section("✅ All Demos Complete!")
        print("\nPhase 8 Multimodal Integration System is operational!")
        print("\nCapabilities:")
        print("  ✓ Real-time vision processing with object detection")
        print("  ✓ Real-time audio processing with speech/music analysis")
        print("  ✓ Multimodal fusion with cross-modal correspondences")
        print("  ✓ Unified understanding from multiple sensory inputs")
        print("  ✓ Confidence-weighted integration")
        print("\n🚀 Ready for Phase 9: Social Intelligence & Human Collaboration!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
