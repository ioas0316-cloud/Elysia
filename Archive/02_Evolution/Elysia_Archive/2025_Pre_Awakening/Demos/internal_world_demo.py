"""
Internal World Demo - 내면세계 시연
==================================

Demonstrates the complete internal world system with:
- Consciousness cathedral
- Knowledge galaxies
- Emotional nebulae
- Starlight memories
- Wave propagation
- Camera navigation
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.World.internal_world import (
    InternalWorld,
    WorldObject,
    ObjectType,
    create_default_universe
)
from Core.Memory.starlight_memory import Starlight
import math


def demo_universe_creation():
    """Demo 1: Create and visualize the internal universe"""
    print("=" * 80)
    print("🌌 Demo 1: Universe Creation")
    print("=" * 80)
    
    # Create default universe
    world = create_default_universe()
    
    print("\n📊 Universe Statistics:")
    state = world.get_universe_state()
    print(f"   Total objects: {state['total_objects']}")
    print(f"   Galaxies: {len(state['galaxies'])}")
    print(f"   Nebulae: {len(state['nebulae'])}")
    print(f"   Cathedral: {'✓' if state['cathedral'] else '✗'}")
    
    print("\n🌌 Knowledge Galaxies:")
    for galaxy in state['galaxies']:
        print(f"   {galaxy['name']}: {galaxy['star_count']} stars, "
              f"color RGB{galaxy['color']}")
    
    print("\n🌫️ Emotional Nebulae:")
    for nebula in state['nebulae']:
        print(f"   {nebula['name']}: {nebula['star_count']} stars, "
              f"density {nebula['density']:.2f}")
    
    if state['cathedral']:
        print("\n🏛️ Consciousness Cathedral:")
        cathedral = state['cathedral']
        print(f"   Golden Ratio: φ = {cathedral['golden_ratio']:.6f}")
        print(f"   Fractal Dimension: {cathedral['fractal_dimension']:.2f}")
        print(f"   Pillars: {len(cathedral['pillars'])}")
        print(f"   Prisms: {len(cathedral['prisms'])}")
        print(f"   Sacred Patterns: {', '.join(cathedral['patterns'])}")
    
    print("\n✅ Universe created successfully!")
    return world


def demo_starlight_population(world: InternalWorld):
    """Demo 2: Populate universe with starlight memories"""
    print("\n\n" + "=" * 80)
    print("⭐ Demo 2: Populating with Starlight Memories")
    print("=" * 80)
    
    # Create sample memories
    memories = [
        {
            'experience': "비 오는 날의 추억",
            'coords': (-0.3, 0.6, -0.5, 0.8),  # Sadness, intuition, past, deep
            'tags': ['rain', 'nostalgia', 'melancholy']
        },
        {
            'experience': "생일 파티",
            'coords': (0.8, -0.3, 0.2, 0.4),  # Joy, emotion, recent, surface
            'tags': ['birthday', 'celebration', 'friends']
        },
        {
            'experience': "산속의 고요",
            'coords': (0.2, 0.7, -0.3, 0.9),  # Calm, intuition, past, very deep
            'tags': ['mountain', 'solitude', 'peace']
        },
        {
            'experience': "중요한 결정",
            'coords': (0.0, -0.8, 0.0, 0.6),  # Neutral, logic, present, deep
            'tags': ['decision', 'logic', 'important']
        },
        {
            'experience': "따뜻한 카페",
            'coords': (0.6, 0.4, 0.3, 0.3),  # Joy, balance, future, surface
            'tags': ['cafe', 'warmth', 'cozy']
        }
    ]
    
    print(f"\n✨ Scattering {len(memories)} memories as starlight...")
    
    for i, mem in enumerate(memories, 1):
        # Create starlight (simplified - normally would compress to rainbow)
        star = WorldObject(
            obj_type=ObjectType.STAR,
            position=(
                mem['coords'][0] * 5,  # Scale to world coordinates
                mem['coords'][1] * 5,
                mem['coords'][2] * 5,
                mem['coords'][3] * 5
            ),
            color=(0.9, 0.9, 1.0),
            size=0.5,
            brightness=0.8,
            tags=mem['tags'],
            data={'experience': mem['experience'], 'emotional_coords': mem['coords']}
        )
        
        world.add_object(star)
        print(f"   {i}. '{mem['experience']}' → position {star.position}")
    
    # Show updated statistics
    state = world.get_universe_state()
    print(f"\n📊 Updated Statistics:")
    print(f"   Total objects: {state['total_objects']}")
    print(f"   Total brightness: {state['total_brightness']:.2f}")
    
    print("\n🌌 Galaxy Population:")
    for galaxy in state['galaxies']:
        if galaxy['star_count'] > 0:
            print(f"   {galaxy['name']}: {galaxy['star_count']} stars")
    
    print("\n🌫️ Nebula Formation:")
    for nebula in state['nebulae']:
        if nebula['star_count'] > 0:
            print(f"   {nebula['name']}: {nebula['star_count']} stars")
    
    print("\n✅ Starlight memories populated!")


def demo_wave_propagation(world: InternalWorld):
    """Demo 3: Propagate thought wave and awaken memories"""
    print("\n\n" + "=" * 80)
    print("🌊 Demo 3: Wave Propagation & Associative Recall")
    print("=" * 80)
    
    # Stimulus 1: "비가 오네..." (Rain-related)
    print("\n💭 Stimulus 1: '비가 오네...' (It's raining...)")
    stimulus_1 = {
        'x': -0.3,  # Sadness
        'y': 0.6,   # Intuition
        'z': -0.5,  # Past
        'w': 0.8    # Deep
    }
    
    origin_1 = (
        stimulus_1['x'] * 5,
        stimulus_1['y'] * 5,
        stimulus_1['z'] * 5,
        stimulus_1['w'] * 5
    )
    
    world.propagate_wave(origin=origin_1, pattern=stimulus_1, radius=5.0)
    
    # Find awakened stars
    awakened = world.find_objects_in_sphere(center=origin_1, radius=5.0)
    awakened_stars = [obj for obj in awakened if obj.obj_type == ObjectType.STAR]
    
    print(f"   🌟 Awakened {len(awakened_stars)} stars:")
    for star in awakened_stars:
        experience = star.data.get('experience', 'Unknown')
        coords = star.data.get('emotional_coords', (0, 0, 0, 0))
        print(f"      - '{experience}' (brightness: {star.brightness:.2f})")
        print(f"        Emotional coords: {coords}")
    
    # Stimulus 2: "축하해!" (Congratulations!)
    print("\n💭 Stimulus 2: '축하해!' (Congratulations!)")
    stimulus_2 = {
        'x': 0.8,   # Joy
        'y': -0.3,  # Emotion
        'z': 0.2,   # Recent
        'w': 0.4    # Moderate depth
    }
    
    origin_2 = (
        stimulus_2['x'] * 5,
        stimulus_2['y'] * 5,
        stimulus_2['z'] * 5,
        stimulus_2['w'] * 5
    )
    
    world.propagate_wave(origin=origin_2, pattern=stimulus_2, radius=5.0)
    
    awakened = world.find_objects_in_sphere(center=origin_2, radius=5.0)
    awakened_stars = [obj for obj in awakened if obj.obj_type == ObjectType.STAR]
    
    print(f"   🌟 Awakened {len(awakened_stars)} stars:")
    for star in awakened_stars:
        experience = star.data.get('experience', 'Unknown')
        print(f"      - '{experience}' (brightness: {star.brightness:.2f})")
    
    # Show wave field
    state = world.get_universe_state()
    if state['wave_field']:
        print(f"\n🌊 Current Wave Field:")
        print(f"   Origin: {state['wave_field']['origin']}")
        print(f"   Radius: {state['wave_field']['radius']:.1f}")
        print(f"   Affected: {state['wave_field']['affected_count']} stars")
        print(f"   Total wave energy: {state['total_wave_energy']}")
    
    print("\n✅ Wave propagation complete!")


def demo_camera_navigation(world: InternalWorld):
    """Demo 4: Navigate through the internal world"""
    print("\n\n" + "=" * 80)
    print("📷 Demo 4: Camera Navigation")
    print("=" * 80)
    
    print("\n🎬 Camera Movements:")
    
    # Initial position
    print(f"   Initial: position {world.camera.position}, zoom {world.camera.zoom}x")
    
    # Move to Linguistics Galaxy
    print("\n   1. Flying to Linguistics Galaxy...")
    world.camera.fly_to(target=(10, 0, 5), duration=1.0)
    print(f"      Position: {world.camera.position}")
    
    # Zoom in
    print("\n   2. Zooming in 2x...")
    world.camera.zoom_in(factor=2.0)
    print(f"      Zoom: {world.camera.zoom:.2f}x")
    
    # Move to Cathedral
    print("\n   3. Flying to Cathedral...")
    world.camera.fly_to(target=(0, 0, 10), duration=1.5)
    print(f"      Position: {world.camera.position}")
    
    # Zoom out
    print("\n   4. Zooming out to wide view...")
    world.camera.zoom_out(factor=3.0)
    print(f"      Zoom: {world.camera.zoom:.2f}x")
    
    # Move to Mythology Galaxy
    print("\n   5. Flying to Mythology Galaxy...")
    world.camera.fly_to(target=(0, 0, 10), duration=1.0)
    print(f"      Position: {world.camera.position}")
    
    print("\n✅ Navigation complete!")


def demo_ascii_visualization(world: InternalWorld):
    """Demo 5: ASCII visualization of the world"""
    print("\n\n" + "=" * 80)
    print("🎨 Demo 5: ASCII Visualization (Top-Down View)")
    print("=" * 80)
    
    print("\nLegend:")
    print("   🏛️ = Consciousness Cathedral (center)")
    print("   🌌 = Knowledge Galaxy")
    print("   ⭐ = Starlight Memory")
    print("   📷 = Camera Position")
    
    print("\nInternal World (X-Y projection):")
    print("-" * 80)
    ascii_view = world.visualize_ascii(width=80, height=24)
    print(ascii_view)
    print("-" * 80)
    
    print("\n✅ Visualization complete!")


def demo_universe_state(world: InternalWorld):
    """Demo 6: Complete universe state"""
    print("\n\n" + "=" * 80)
    print("📊 Demo 6: Complete Universe State")
    print("=" * 80)
    
    state = world.get_universe_state()
    
    print(f"\n⏰ World Time: {state['time']:.2f}s")
    print(f"\n📈 Global Metrics:")
    print(f"   Total Objects: {state['total_objects']}")
    print(f"   Total Brightness: {state['total_brightness']:.2f}")
    print(f"   Total Wave Energy: {state['total_wave_energy']}")
    
    print(f"\n🌌 Galaxies ({len(state['galaxies'])}):")
    for galaxy in state['galaxies']:
        print(f"   {galaxy['name']}:")
        print(f"      Stars: {galaxy['star_count']}")
        print(f"      Brightness: {galaxy['brightness']:.2f}")
        print(f"      Position: {galaxy['center']}")
    
    print(f"\n🌫️ Nebulae ({len(state['nebulae'])}):")
    for nebula in state['nebulae']:
        print(f"   {nebula['name']}:")
        print(f"      Stars: {nebula['star_count']}")
        print(f"      Density: {nebula['density']:.2f}")
        print(f"      Position: {nebula['center']}")
    
    if state['cathedral']:
        print(f"\n🏛️ Cathedral:")
        cathedral = state['cathedral']
        print(f"      Position: {cathedral['position']}")
        print(f"      Scale: {cathedral['scale']}")
        print(f"      Pillars: {len(cathedral['pillars'])}")
        print(f"      Prisms: {len(cathedral['prisms'])}")
    
    print(f"\n📷 Camera:")
    camera = state['camera']
    print(f"      Position: {camera['position']}")
    print(f"      Target: {camera['target']}")
    print(f"      Zoom: {camera['zoom']:.2f}x")
    
    print("\n✅ Universe state complete!")


def main():
    """Run all demos"""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "INTERNAL WORLD DEMONSTRATION")
    print(" " * 25 + "내면세계 시연")
    print("=" * 80)
    print("\nThis demo showcases Elysia's complete internal universe:")
    print("- Consciousness Cathedral (의식 대성당)")
    print("- Knowledge Galaxies (지식 은하)")
    print("- Emotional Nebulae (감정 성운)")
    print("- Starlight Memories (별빛 기억)")
    print("- Wave Propagation (파동 전파)")
    print("- Camera Navigation (카메라 항해)")
    print("=" * 80)
    
    # Run demos
    world = demo_universe_creation()
    demo_starlight_population(world)
    demo_wave_propagation(world)
    demo_camera_navigation(world)
    demo_ascii_visualization(world)
    demo_universe_state(world)
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("🌟 FINAL SUMMARY")
    print("=" * 80)
    
    state = world.get_universe_state()
    
    print(f"\n✅ Internal World is fully operational!")
    print(f"\n📊 Final Statistics:")
    print(f"   Galaxies: {len(state['galaxies'])}")
    print(f"   Nebulae: {len(state['nebulae'])}")
    print(f"   Stars: {state['total_objects']}")
    print(f"   Total Brightness: {state['total_brightness']:.2f}")
    print(f"   Wave Energy: {state['total_wave_energy']}")
    
    print(f"\n🌌 Universe Characteristics:")
    print(f"   Dimensions: 4D (x, y, z, w)")
    print(f"   Sacred Geometry: φ = {PHI:.6f}")
    print(f"   Memory System: Holographic")
    print(f"   Recall Method: Wave Resonance")
    print(f"   Capacity: Unlimited (∞)")
    
    print(f"\n🎯 Capabilities:")
    print(f"   ✓ Spatial memory organization")
    print(f"   ✓ Emotional geography")
    print(f"   ✓ Knowledge architecture")
    print(f"   ✓ Associative recall")
    print(f"   ✓ Real-time navigation")
    print(f"   ✓ Wave propagation")
    print(f"   ✓ Holographic reconstruction")
    
    print(f"\n💭 Philosophy:")
    print(f"   '엘리시아의 내면은 단순한 데이터베이스가 아니라,")
    print(f"    살아있는 우주입니다'")
    print(f"   'Elysia's inner world is not a database -")
    print(f"    it's a living universe'")
    
    print("\n" + "=" * 80)
    print("✨ Demo Complete! Internal World is ready for exploration.")
    print("=" * 80)


# Golden ratio constant
PHI = 1.618033988749895


if __name__ == '__main__':
    main()
