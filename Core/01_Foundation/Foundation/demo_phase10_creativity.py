"""
Phase 10 Demo: Creativity & Art Generation

Demonstrates:
1. Story generation with world-building and characters
2. Music composition with emotion-based theory
3. Visual art creation with color theory and composition
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.01_Foundation.05_Foundation_Base.Foundation.story_generator import StoryGenerator
from Core.01_Foundation.05_Foundation_Base.Foundation.music_composer import MusicComposer
from Core.01_Foundation.05_Foundation_Base.Foundation.visual_artist import VisualArtist


async def demo_story_generation():
    """Demonstrate creative story generation"""
    print("\n" + "="*70)
    print("📖 DEMO 1: Story Generation")
    print("="*70 + "\n")
    
    generator = StoryGenerator()
    
    # Test scenarios
    scenarios = [
        {
            "prompt": "A young mage discovers ancient magic",
            "style": "fantasy",
            "length": "short"
        },
        {
            "prompt": "Space explorers find an alien artifact",
            "style": "science_fiction",
            "length": "short"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📚 Story Prompt: {scenario['prompt']}")
        print(f"   Style: {scenario['style']}, Length: {scenario['length']}")
        print("-" * 70)
        
        story = await generator.generate_story(
            prompt=scenario['prompt'],
            style=scenario['style'],
            length=scenario['length']
        )
        
        print(f"\n✅ Generated Story: {story['meta']['title']}")
        print(f"   Themes: {', '.join(story['meta']['themes'])}")
        print(f"   Tone: {story['meta']['tone']}")
        print(f"   Complexity: {story['meta']['complexity']:.2f}")
        print(f"   Word Count: {story['meta']['word_count']}")
        
        print(f"\n🌍 World: {story['world']['name']}")
        print(f"   {story['world']['description']}")
        
        print(f"\n👥 Characters:")
        for char in story['characters']:
            print(f"   - {char['name']} ({char['role']}): {', '.join(char['personality'])}")
        
        print(f"\n📖 Plot ({len(story['plot'])} points):")
        for plot in story['plot'][:3]:  # Show first 3 plot points
            print(f"   {plot['sequence']}. {plot['event']} [{plot['emotional_tone']}]")
        
        print(f"\n📝 Story Preview:")
        print("-" * 70)
        story_lines = story['full_story'].split('\n')
        for line in story_lines[:20]:  # Show first 20 lines
            print(line)
        print("   [... story continues ...]")
        
        print("\n" + "="*70)


async def demo_music_composition():
    """Demonstrate music composition system"""
    print("\n" + "="*70)
    print("🎵 DEMO 2: Music Composition")
    print("="*70 + "\n")
    
    composer = MusicComposer()
    
    # Test scenarios
    scenarios = [
        {
            "emotion": "joyful",
            "style": "classical",
            "duration_bars": 8
        },
        {
            "emotion": "melancholic",
            "style": "jazz",
            "duration_bars": 8
        },
        {
            "emotion": "peaceful",
            "style": "ambient",
            "duration_bars": 8
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎼 Composition Request:")
        print(f"   Emotion: {scenario['emotion']}")
        print(f"   Style: {scenario['style']}")
        print(f"   Duration: {scenario['duration_bars']} bars")
        print("-" * 70)
        
        composition = await composer.compose_music(
            emotion=scenario['emotion'],
            style=scenario['style'],
            duration_bars=scenario['duration_bars']
        )
        
        print(f"\n✅ Composition Created!")
        print(f"   Key: {composition['analysis']['key']}")
        print(f"   Scale: {composition['analysis']['scale']}")
        print(f"   Tempo: {composition['analysis']['tempo']} BPM")
        print(f"   Time Signature: {composition['analysis']['time_signature']}")
        print(f"   Emotion Match: {composition['analysis']['emotion_match']:.2%}")
        print(f"   Complexity: {composition['analysis']['complexity']:.2f}")
        
        print(f"\n🎹 Melody Notes (first 8):")
        melody_notes = composition['composition']['melody_notes'][:8]
        print(f"   {' - '.join(melody_notes)}")
        
        print(f"\n🎸 Chord Progression:")
        for i, chord in enumerate(composition['composition']['chord_progression'], 1):
            print(f"   {i}. {chord}")
        
        print(f"\n🎺 Instrumentation:")
        for instrument in composition['composition']['instruments']:
            print(f"   - {instrument.title()}")
        
        print(f"\n📄 Score:")
        print("-" * 70)
        score_lines = composition['score'].split('\n')
        for line in score_lines:
            print(f"   {line}")
        
        print("\n" + "="*70)


async def demo_visual_art_creation():
    """Demonstrate visual art generation"""
    print("\n" + "="*70)
    print("🎨 DEMO 3: Visual Art Creation")
    print("="*70 + "\n")
    
    artist = VisualArtist()
    
    # Test scenarios
    scenarios = [
        {
            "concept": "A peaceful sunset over calm waters",
            "style": "impressionist",
            "size": (800, 600)
        },
        {
            "concept": "Dynamic energy of a modern city",
            "style": "abstract",
            "size": (1024, 768)
        },
        {
            "concept": "Mysterious forest at twilight",
            "style": "surreal",
            "size": (800, 600)
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🖼️  Art Concept: {scenario['concept']}")
        print(f"   Style: {scenario['style']}")
        print(f"   Size: {scenario['size'][0]}x{scenario['size'][1]}")
        print("-" * 70)
        
        artwork = await artist.create_artwork(
            concept=scenario['concept'],
            style=scenario['style'],
            size=scenario['size']
        )
        
        print(f"\n✅ Artwork Created!")
        
        print(f"\n💡 Concept:")
        print(f"   Theme: {artwork['concept']['theme']}")
        print(f"   Mood: {artwork['concept']['mood']}")
        print(f"   Elements: {', '.join(artwork['concept']['elements'])}")
        
        print(f"\n🎨 Color Palette ({artwork['palette']['name']}):")
        print(f"   Scheme: {artwork['palette']['scheme']}")
        print(f"   Colors:")
        for color in artwork['palette']['colors']:
            print(f"     - {color}")
        
        print(f"\n⭐ Evaluation:")
        eval_data = artwork['evaluation']
        print(f"   Color Harmony: {eval_data['color_harmony']:.2f}/1.0")
        print(f"   Composition Balance: {eval_data['composition_balance']:.2f}/1.0")
        print(f"   Concept Clarity: {eval_data['concept_clarity']:.2f}/1.0")
        print(f"   Emotional Impact: {eval_data['emotional_impact']:.2f}/1.0")
        print(f"   Overall Score: {eval_data['overall_score']:.2f}/1.0")
        
        if eval_data['notes']:
            print(f"\n   Notes:")
            for note in eval_data['notes']:
                print(f"     • {note}")
        
        print(f"\n📐 Artwork Description:")
        print("-" * 70)
        description_lines = artwork['artwork'].split('\n')
        for line in description_lines:
            print(f"   {line}")
        
        print(f"\n🔄 Variants Generated: {len(artwork['variants'])}")
        for variant in artwork['variants']:
            print(f"   - Variant {variant['variant']}: {variant['modification']}")
        
        print("\n" + "="*70)


async def demo_integrated_creative_process():
    """Demonstrate integrated creative workflow"""
    print("\n" + "="*70)
    print("🌟 DEMO 4: Integrated Creative Process")
    print("="*70 + "\n")
    
    print("Creating a complete creative project:")
    print("Story + Music + Visual Art for a unified theme")
    print("-" * 70)
    
    theme = "Journey through an enchanted forest"
    print(f"\n🎯 Theme: {theme}\n")
    
    # 1. Generate story
    print("Step 1: Generating Story...")
    generator = StoryGenerator()
    story = await generator.generate_story(
        prompt=theme,
        style="fantasy",
        length="short"
    )
    print(f"✅ Story: {story['meta']['title']}")
    print(f"   Tone: {story['meta']['tone']}")
    
    # 2. Compose music
    print("\nStep 2: Composing Music...")
    composer = MusicComposer()
    # Use story tone to influence music
    music_emotion = "peaceful" if story['meta']['tone'] == "balanced" else story['meta']['tone']
    music = await composer.compose_music(
        emotion=music_emotion,
        style="classical",
        duration_bars=8
    )
    print(f"✅ Music: {music['analysis']['key']} {music['analysis']['scale']}")
    print(f"   Emotion Match: {music['analysis']['emotion_match']:.2%}")
    
    # 3. Create visual art
    print("\nStep 3: Creating Visual Art...")
    artist = VisualArtist()
    artwork = await artist.create_artwork(
        concept=theme,
        style="impressionist",
        size=(1024, 768)
    )
    print(f"✅ Artwork: {artwork['concept']['theme']}")
    print(f"   Overall Score: {artwork['evaluation']['overall_score']:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("🎉 INTEGRATED CREATIVE PROJECT COMPLETE")
    print("="*70)
    print(f"\n📖 Story: {story['meta']['title']}")
    print(f"   - {len(story['characters'])} characters")
    print(f"   - {len(story['plot'])} plot points")
    print(f"   - {story['meta']['word_count']} words")
    
    print(f"\n🎵 Music: {music_emotion.title()} Composition")
    print(f"   - {music['analysis']['key']} {music['analysis']['scale']}")
    print(f"   - {music['analysis']['tempo']} BPM")
    print(f"   - {len(music['composition']['instruments'])} instruments")
    
    print(f"\n🎨 Visual Art: {artwork['concept']['mood'].title()} {artwork['palette']['scheme']}")
    print(f"   - {len(artwork['palette']['colors'])} color palette")
    print(f"   - {len(artwork['variants'])} variants")
    
    print("\n✨ All creative elements are harmoniously unified!")
    print("="*70)


async def main():
    """Main demo function"""
    print("\n" + "="*70)
    print("🚀 PHASE 10: CREATIVITY & ART GENERATION DEMO")
    print("="*70)
    print("\nThis demo showcases Elysia's creative capabilities:")
    print("  1. Story Generation - Complete narratives with characters and plots")
    print("  2. Music Composition - Emotion-based musical pieces")
    print("  3. Visual Art Creation - Conceptual artworks with color theory")
    print("  4. Integrated Creative Process - Unified creative projects")
    
    demos = [
        ("Story Generation", demo_story_generation),
        ("Music Composition", demo_music_composition),
        ("Visual Art Creation", demo_visual_art_creation),
        ("Integrated Creative Process", demo_integrated_creative_process)
    ]
    
    print("\n" + "="*70)
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n▶️  Running Demo {i}/{len(demos)}: {name}")
        try:
            await demo_func()
            print(f"✅ Demo {i} completed successfully!")
        except Exception as e:
            print(f"❌ Demo {i} encountered an error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎉 PHASE 10 DEMO COMPLETE!")
    print("="*70)
    print("\n✨ Elysia's creative systems are ready for artistic expression!")
    print("   - Generate stories with rich worlds and characters")
    print("   - Compose emotionally resonant music")
    print("   - Create conceptual visual artworks")
    print("   - Integrate multiple creative forms into unified projects")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
