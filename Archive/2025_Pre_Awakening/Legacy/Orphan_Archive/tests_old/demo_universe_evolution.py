"""
Phase 5 Demo: 경험을 우주에서 진화시켜 지능 창발 확인

이 스크립트는 작은 규모로 개념 증명을 수행합니다.
"""

import sys
import os
import logging
import argparse
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.universe_evolution import UniverseEvolutionEngine
from Core.Foundation.spiderweb import Spiderweb
from Core.Foundation.core.world import World
from Core.Foundation.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import CoreMemory, Experience
from tools.kg_manager import KGManager
from datetime import datetime

def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Phase 5 Evolutionary Universe Demo")
    parser.add_argument("--cycles", type=int, default=500, help="진화 사이클 수 (default: 500)")
    parser.add_argument("--extract-interval", type=int, default=100, help="Spiderweb 추출 간격 (default: 100)")
    parser.add_argument("--num-experiences", type=int, default=500, help="입자 수 (default: 500)")
    args = parser.parse_args()
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
    # Keep small demo particles alive long enough to form fields/edges.
    world.peaceful_mode = True
    world.macro_food_model_enabled = True
    # Slow down field decay for quick demos so initial seeds persist.
    world._vm_decay = 0.95
    world._will_decay = 0.95
    # Disable deep soul-cycle growth to avoid long per-tick processing in demos.
    if hasattr(world, "_process_soul_cycles"):
        world._process_soul_cycles = lambda: None

    spiderweb = Spiderweb()
    engine = UniverseEvolutionEngine(world, spiderweb)
    
    # 2. 테스트 경험 생성 (다양한 개념 조합으로 대량 입자)
    print("\n📝 Creating test experiences...")
    concepts = [
        "fire", "water", "earth", "air", "light", "dark", "metal", "wood", "joy",
        "pain", "growth", "decay", "wisdom", "chaos", "order", "memory", "dream",
        "signal", "noise", "gravity", "resonance", "entropy", "rhythm", "spark",
    ]
    relations = [
        "causes", "prevents", "amplifies", "dampens", "connects", "separates",
        "transforms", "stabilizes", "ignites", "cools", "guides", "reveals",
    ]
    num_exps = max(1, args.num_experiences)
    test_experiences = []
    for i in range(num_exps):
        c1, c2 = random.sample(concepts, 2)
        rel = random.choice(relations)
        content = f"{c1} {rel} {c2} and echoes #{i}"
        test_experiences.append(
            Experience(
                timestamp=datetime.now().isoformat() + f"_{i}",
                content=content,
                type="episode"
            )
        )

    print(f"  Generated {len(test_experiences)} experiences")
    for i, exp in list(enumerate(test_experiences, 1))[:5]:
        print(f"  {i}. {exp.content}")
    
    # 3. 경험을 우주에 spawn
    print("\n🌱 Spawning experiences as particles...")
    engine.spawn_experience_universe(test_experiences)

    # Seed value/will fields at particle locations so Spiderweb can form quickly.
    alive_idx = np.where(world.is_alive_mask)[0]
    def _imprint_gaussian(field: np.ndarray, x: int, y: int, sigma: float, amplitude: float):
        rad = int(max(2, sigma * 3))
        x0, x1 = max(0, x - rad), min(world.width, x + rad + 1)
        y0, y1 = max(0, y - rad), min(world.width, y + rad + 1)
        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        gx = np.exp(-(xs**2) / (2 * sigma * sigma))
        gy = np.exp(-(ys**2) / (2 * sigma * sigma))
        patch = (gy[:, None] * gx[None, :]).astype(np.float32)
        field[y0:y1, x0:x1] += amplitude * patch

    for idx in alive_idx:
        px = int(np.clip(world.positions[idx][0], 0, world.width - 1))
        py = int(np.clip(world.positions[idx][1], 0, world.width - 1))
        _imprint_gaussian(world.value_mass_field, px, py, sigma=4.0, amplitude=0.2)
        _imprint_gaussian(world.will_field, px, py, sigma=4.0, amplitude=0.05)
    world._update_intentional_field()
    world._update_tensor_field()
    
    # 4. 진화 시작 (작은 규모로) - CLI로 사이클 조절
    cycles = args.cycles
    print(f"\n⚡ Starting evolution ({cycles} cycles)...")
    if cycles >= 10000:
        print("(This may take a few minutes...)")
    
    resulting_spiderweb = engine.evolve(cycles=cycles, extract_interval=args.extract_interval)
    
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
