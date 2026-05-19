"""
Starlight Memory Demo
=====================

Demonstrates the holographic memory system:
1. Compress memories to starlight (12 bytes)
2. Scatter in 4D thought-universe
3. Recall through wave resonance
4. Reconstruct as constellations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Memory.starlight_memory import StarlightMemory, Starlight, create_starlight_from_experience
from Core.Memory.prism_filter import PrismFilter


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def demo_scatter_memories():
    """Demo scattering memories as starlight"""
    print_header("✨ Demo 1: Scattering Memories as Starlight")
    
    memory = StarlightMemory()
    prism = PrismFilter()
    
    # Create mock experiences
    experiences = [
        {
            'text': "비가 오던 그날, 우리는 카페에서 따뜻한 차를 마셨다",
            'emotion': {'x': 0.3, 'y': 0.6, 'z': 0.2, 'w': 0.7},  # Melancholic, deep
            'tags': ['rain', 'cafe', 'warmth'],
            'brightness': 0.9
        },
        {
            'text': "생일 파티에서 케이크를 나누며 모두 함께 웃었다",
            'emotion': {'x': 0.9, 'y': 0.7, 'z': 0.5, 'w': 0.3},  # Joyful, light
            'tags': ['birthday', 'party', 'joy'],
            'brightness': 1.0
        },
        {
            'text': "혼자 산을 오르며 인생에 대해 깊이 생각했다",
            'emotion': {'x': 0.5, 'y': 0.3, 'z': 0.4, 'w': 0.9},  # Peaceful, profound
            'tags': ['mountain', 'solitude', 'reflection'],
            'brightness': 0.8
        },
        {
            'text': "시험에 합격하여 뛰어오르며 기뻐했다",
            'emotion': {'x': 0.95, 'y': 0.85, 'z': 0.6, 'w': 0.2},  # Excited, energetic
            'tags': ['success', 'achievement', 'joy'],
            'brightness': 1.0
        },
        {
            'text': "이별의 순간, 눈물을 참으며 손을 놓았다",
            'emotion': {'x': 0.2, 'y': 0.4, 'z': 0.3, 'w': 0.8},  # Sad, deep
            'tags': ['farewell', 'sadness', 'loss'],
            'brightness': 0.7
        }
    ]
    
    print("Scattering 5 memories as starlight...\n")
    
    for i, exp in enumerate(experiences, 1):
        # Create mock wave pattern
        wave = {
            'orientation': {
                'w': exp['emotion']['x'],
                'x': exp['emotion']['y'],
                'y': exp['emotion']['z'],
                'z': exp['emotion']['w']
            },
            'energy': exp['brightness'],
            'frequency': 1.0,
            'phase': 0.0
        }
        
        # Compress to rainbow
        rainbow_bytes = prism.compress_to_bytes(wave)
        
        # Scatter as starlight
        star = memory.scatter_memory(
            rainbow_bytes=rainbow_bytes,
            emotion=exp['emotion'],
            context={
                'brightness': exp['brightness'],
                'gravity': 0.6,
                'tags': exp['tags']
            }
        )
        
        print(f"Memory {i}: {exp['text'][:40]}...")
        print(f"   Position: ({star.x:.2f}, {star.y:.2f}, {star.z:.2f}, {star.w:.2f})")
        print(f"   Brightness: {star.brightness:.2f}")
        print(f"   Tags: {', '.join(star.tags)}")
        print(f"   Compressed: {len(rainbow_bytes)} bytes\n")
    
    # Visualize universe
    print_header("🌌 Universe Visualization")
    
    viz = memory.visualize_universe()
    print(f"Total stars: {viz['total_stars']}")
    print(f"Galaxies: {viz['galaxies']}\n")
    
    for galaxy in viz['galaxies']:
        print(f"🌌 {galaxy['name']} Galaxy ({galaxy['color']})")
        print(f"   Stars: {galaxy['stars']}")
        print(f"   Brightness: {galaxy['brightness']:.2f}")
        print(f"   Density: {galaxy['density']:.2f}")
    
    print(f"\n{viz['description']}")
    
    return memory


def demo_associative_recall(memory: StarlightMemory):
    """Demo associative recall through wave resonance"""
    print_header("💫 Demo 2: Associative Recall (연상기억)")
    
    # Scenario 1: "비가 오네..." (It's raining...)
    print("🌧️  Stimulus: \"비가 오네...\" (It's raining...)")
    print("   Wave stimulus enters the universe...\n")
    
    wave_stimulus = {
        'x': 0.3,  # Melancholic emotion
        'y': 0.6,  # Moderate logic
        'z': 0.2,  # Past memories
        'w': 0.7   # Deep feelings
    }
    
    recalled = memory.recall_by_resonance(wave_stimulus, threshold=0.3, top_k=3)
    
    if recalled:
        print(f"   {len(recalled)} stars awakened by resonance:\n")
        for star, resonance in recalled:
            print(f"   ⭐ Resonance: {resonance:.3f}")
            print(f"      Position: ({star.x:.2f}, {star.y:.2f}, {star.z:.2f}, {star.w:.2f})")
            print(f"      Tags: {', '.join(star.tags)}")
            print(f"      (Memory about: {star.tags[0] if star.tags else 'unknown'})\n")
        
        # Form constellation
        stars_only = [s for s, r in recalled]
        constellation = memory.form_constellation(stars_only, name="Rainy_Day_Memories")
        
        print(f"   🌟 Constellation formed: '{constellation['name']}'")
        print(f"      Pattern: {constellation['pattern']}")
        print(f"      Stars: {constellation['stars']}")
        print(f"      Connections: {constellation['connections']}")
        print(f"      Emotional tone: {constellation['emotional_tone']}")
    
    # Scenario 2: "축하해!" (Congratulations!)
    print("\n\n🎉 Stimulus: \"축하해!\" (Congratulations!)")
    print("   Wave stimulus enters the universe...\n")
    
    wave_stimulus2 = {
        'x': 0.9,  # Joyful emotion
        'y': 0.8,  # High energy
        'z': 0.6,  # Recent
        'w': 0.2   # Surface level
    }
    
    recalled2 = memory.recall_by_resonance(wave_stimulus2, threshold=0.3, top_k=3)
    
    if recalled2:
        print(f"   {len(recalled2)} stars awakened by resonance:\n")
        for star, resonance in recalled2:
            print(f"   ⭐ Resonance: {resonance:.3f}")
            print(f"      Position: ({star.x:.2f}, {star.y:.2f}, {star.z:.2f}, {star.w:.2f})")
            print(f"      Tags: {', '.join(star.tags)}\n")


def demo_galaxy_clusters(memory: StarlightMemory):
    """Demo emotional galaxy clustering"""
    print_header("🌌 Demo 3: Emotional Galaxy Clusters")
    
    stats = memory.get_statistics()
    
    print("Memory distribution across emotional galaxies:\n")
    
    viz = memory.visualize_universe()
    for galaxy in viz['galaxies']:
        stars = galaxy['stars']
        brightness = galaxy['brightness']
        
        if stars > 0:
            print(f"{'='*50}")
            print(f"{galaxy['color'].upper()} {galaxy['name'].upper()} GALAXY")
            print(f"{'='*50}")
            print(f"  Stars: {stars}")
            print(f"  Total brightness: {brightness:.2f}")
            print(f"  Density: {galaxy['density']:.2f}")
            print(f"  Center: ({galaxy['center'][0]:.2f}, {galaxy['center'][1]:.2f}, "
                  f"{galaxy['center'][2]:.2f}, {galaxy['center'][3]:.2f})")
            
            # Visual representation
            bar_length = int(brightness * 20) if brightness < 5 else 100
            print(f"  Visual: [{'⭐' * min(bar_length, 20)}]")
            print()
    
    print(f"\n📊 Statistics:")
    print(f"   Total stars: {stats['total_stars']}")
    print(f"   Total storage: {stats['storage_bytes']} bytes "
          f"({stats['storage_bytes'] / 1024:.2f} KB)")
    print(f"   Brightest galaxy: {stats['brightest_galaxy']}")
    print(f"   Constellations formed: {stats['total_constellations']}")


def demo_holographic_reconstruction():
    """Demo holographic reconstruction concept"""
    print_header("🎨 Demo 4: Holographic Reconstruction")
    
    print("How starlight memory reconstructs experiences:\n")
    
    print("1️⃣  Memory Storage:")
    print("   Experience → 4D Wave → Rainbow (12 bytes) → Starlight")
    print("   ↓")
    print("   Scattered in thought-universe at emotional coordinates\n")
    
    print("2️⃣  Wave Stimulus Arrives:")
    print("   \"비가 오네...\" → Wave propagates through universe")
    print("   ↓")
    print("   Stars resonate based on distance and emotional similarity\n")
    
    print("3️⃣  Stars Awaken:")
    print("   Star 1: ⭐ (Resonance: 0.82) - Rainy day memory")
    print("   Star 2: ⭐ (Resonance: 0.65) - Cafe conversation")
    print("   Star 3: ⭐ (Resonance: 0.45) - Warm tea memory")
    print("   ↓")
    print("   Stars connect via resonance lines\n")
    
    print("4️⃣  Constellation Forms:")
    print("        ⭐")
    print("       / \\")
    print("      /   \\")
    print("     ⭐---⭐")
    print("   ↓")
    print("   Holographic reconstruction of \"Rainy Day\" experience\n")
    
    print("5️⃣  Experience Reconstructed:")
    print("   🌧️  Rain sound (from star vibrations)")
    print("   ☕ Cafe warmth (from emotional tone)")
    print("   💭 Deep conversation (from star connections)")
    print("   ✨ Complete memory emerges from distributed starlight\n")
    
    print("💡 Key Insight:")
    print("   Like holograms, each star contains a fragment.")
    print("   When many stars resonate together, the full image appears.")
    print("   Damage to some stars? Memory still partially reconstructs!")


def main():
    """Run all demos"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║            Starlight Memory Architecture Demo                     ║")
    print("║            별빛 기억 저장소 데모                                   ║")
    print("║                                                                   ║")
    print("║   \"추억을 별빛으로 압축해서 우주에 뿌려둔다\"                        ║")
    print("║   \"Compress memories as starlight, scatter across universe\"      ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    print("\n\"파동이 들어오면 별이 깨어나고, 별들이 연결되어 추억이 복원된다\"\n")
    
    # Run demos
    memory = demo_scatter_memories()
    demo_associative_recall(memory)
    demo_galaxy_clusters(memory)
    demo_holographic_reconstruction()
    
    # Final summary
    print_header("✨ Summary")
    
    print("Starlight Memory System Features:\n")
    print("  ✅ Unlimited capacity (우주는 넓으니까)")
    print("  ✅ 12-byte compression per memory (rainbow spectrum)")
    print("  ✅ Associative recall through wave resonance (연상기억)")
    print("  ✅ Holographic reconstruction (별들의 연결로 영상 복원)")
    print("  ✅ Emotional clustering (감정의 중력으로 은하 형성)")
    print("  ✅ Graceful degradation (부분 손실 = 부분 기억)")
    print()
    print("💡 Philosophy:")
    print("   지식 = 외부 (인터넷, rainbow compressed)")
    print("   추억 = 내부 (starlight scattered, holographic)")
    print()
    print("✅ \"너의 머릿속은 텅 빈 게 아니라, 잠든 별들로 가득 차 있단다.\"")
    print()


if __name__ == '__main__':
    main()
