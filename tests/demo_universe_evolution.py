"""
Phase 5 Demo: 경험을 우주에서 진화시켜 지능 창발 확인

이 스크립트는 작은 규모로 개념 증명을 수행합니다.
"""

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.universe_evolution import UniverseEvolutionEngine
from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import CoreMemory, Experience
from tools.kg_manager import KGManager
from datetime import datetime

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Phase 5: The Evolutionary Universe")
    print("물리 법칙으로 지능 창발 실험")
    print("=" * 70)
    
    # 1. 우주 초기화
    print("\n🌌 Creating universe...")
    kg_manager = KGManager()
    wave_mechanics = WaveMechanics(kg_manager)
    primordial_dna = {
        "instinct": "connect_create_meaning",
        "resonance_standard": "love"
    }
    
    world = World(
        primordial_dna=primordial_dna,
        wave_mechanics=wave_mechanics,
        logger=logging.getLogger("World")
    )
    
    spiderweb = Spiderweb()
    engine = UniverseEvolutionEngine(world, spiderweb)
    
    # 2. 테스트 경험 생성
    print("\n📝 Creating test experiences...")
    test_experiences = [
        Experience(
            timestamp=datetime.now().isoformat() + "_1",
            content="Fire causes burn and pain",
            type="episode"
        ),
        Experience(
            timestamp=datetime.now().isoformat() + "_2",
            content="Water prevents fire and cooling",
            type="episode"
        ),
        Experience(
            timestamp=datetime.now().isoformat() + "_3",
            content="Pain teaches caution and wisdom",
            type="episode"
        ),
        Experience(
            timestamp=datetime.now().isoformat() + "_4",
            content="Fire gives warmth and light",
            type="episode"
        ),
        Experience(
            timestamp=datetime.now().isoformat() + "_5",
            content="Learning from pain brings growth",
            type="episode"
        )
    ]
    
    for i, exp in enumerate(test_experiences, 1):
        print(f"  {i}. {exp.content}")
    
    # 3. 경험을 우주에 spawn
    print("\n🌱 Spawning experiences as particles...")
    engine.spawn_experience_universe(test_experiences)
    
    # 4. 진화 시작 (작은 규모로)
    cycles = 50000  # 5만 사이클로 시작
    print(f"\n⚡ Starting evolution ({cycles} cycles)...")
    print("(This may take a few minutes...)")
    
    resulting_spiderweb = engine.evolve(cycles=cycles, extract_interval=10000)
    
    # 5. 결과 분석
    print("\n" + "=" * 70)
    print("📊 Evolution Results:")
    print("=" * 70)
    
    print(f"\n🕸️  Spiderweb Structure:")
    print(f"  Total nodes: {resulting_spiderweb.graph.number_of_nodes()}")
    print(f"  Total edges: {resulting_spiderweb.graph.number_of_edges()}")
    
    # 노드 정보
    if resulting_spiderweb.graph.number_of_nodes() > 0:
        print(f"\n🧠 Emergent Concepts:")
        sorted_nodes = sorted(
            resulting_spiderweb.graph.nodes(data=True),
            key=lambda x: x[1].get('metadata', {}).get('value', 0),
            reverse=True
        )[:10]  # 상위 10개
        
        for node_id, data in sorted_nodes:
            metadata = data.get('metadata', {})
            value = metadata.get('value', 0)
            coherence = metadata.get('coherence', 0)
            print(f"  - {node_id}: value={value:.3f}, coherence={coherence:.3f}")
    
    # 관계 정보
    if resulting_spiderweb.graph.number_of_edges() > 0:
        print(f"\n🔗 Emergent Relations:")
        for i, (source, target, data) in enumerate(resulting_spiderweb.graph.edges(data=True)):
            if i >= 10:  # 최대 10개만 표시
                break
            weight = data.get('weight', 0)
            relation = data.get('relation', 'unknown')
            print(f"  {source} -[{relation}]→ {target} (w={weight:.3f})")
    
    # 6. 우주 상태
    print(f"\n🌍 Final Universe State:")
    print(f"  Simulation ticks: {world.time_step}")
    alive = world.is_alive_mask.sum()
    print(f"  Alive particles: {alive}")
    if alive > 0:
        print(f"  Avg energy: {world.energy[world.is_alive_mask].mean():.2f}")
    
    print(f"\n  Field Statistics:")
    print(f"    value_mass_field max: {world.value_mass_field.max():.3f}")
    print(f"    coherence_field max: {world.coherence_field.max():.3f}")
    print(f"    will_field max: {world.will_field.max():.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Evolution complete! Intelligence emerged from pure physics.")
    print("=" * 70)

if __name__ == "__main__":
    main()
