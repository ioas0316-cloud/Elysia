"""대규모 우주 진화 - 500개 경험 (워프 레이어 적용)"""
import sys, os, logging, random
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.universe_evolution import UniverseEvolutionEngine
from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import Experience
from tools.kg_manager import KGManager
from Project_Sophia.warp_layer import WarpLayer, quaternion_from_axis_angle

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("=" * 70)
print("Phase 5: LARGE SCALE Universe Evolution - 500 Particles (Warp Enabled)")
print("=" * 70)

# ------------------------------------------------------------------
# 1️⃣ 초기 설정 (peaceful mode, macro food)
# ------------------------------------------------------------------
kg = KGManager()
wave = WaveMechanics(kg)
world = World({"instinct": "evolve", "resonance": " emergence"}, wave)
world.peaceful_mode = True
world.macro_food_model_enabled = True

spider = Spiderweb()
engine = UniverseEvolutionEngine(world, spider)

# ------------------------------------------------------------------
# 2️⃣ (Optional) Warm‑up: 짧게 필드 초기화
# ------------------------------------------------------------------
warmup_cycles = 2000
print(f"\n⚡ Warm‑up {warmup_cycles} cycles (no extraction)...")
engine.evolve(cycles=warmup_cycles, extract_interval=warmup_cycles)

# ------------------------------------------------------------------
# 3️⃣ 500개의 다양한 경험 생성
# ------------------------------------------------------------------
concepts = [
    "fire", "water", "earth", "air", "light", "dark", "heat", "cold",
    "pain", "joy", "fear", "love", "growth", "death", "birth", "life",
    "wisdom", "ignorance", "strength", "weakness", "fast", "slow",
    "big", "small", "hard", "soft", "sharp", "dull", "new", "old",
]
relations = [
    "causes", "prevents", "enables", "destroys", "creates", "heals",
    "teaches", "reveals", "hides", "transforms", "strengthens", "weakens",
]

experiences = []
for i in range(500):
    c1, c2, c3 = random.sample(concepts, 3)
    rel = random.choice(relations)
    content = f"{c1} {rel} {c2} and affects {c3}"
    experiences.append(
        Experience(datetime.now().isoformat() + f"_{i}", content, "episode")
    )

print(f"\n📝 Generated {len(experiences)} diverse experiences")
print(f"Sample: {experiences[0].content}")
print(f"Sample: {experiences[100].content}")
print(f"Sample: {experiences[250].content}")

# ------------------------------------------------------------------
# 4️⃣ 입자 스폰
# ------------------------------------------------------------------
engine.spawn_experience_universe(experiences)
print(f"\n🌱 Spawned: {world.is_alive_mask.sum()} particles")

# ------------------------------------------------------------------
# 5️⃣ 대규모 진화 + 주기적 워프
# ------------------------------------------------------------------
TOTAL_CYCLES = 50000
WARP_INTERVAL = 10000  # 매 10k 사이클마다 워프 적용
warp = WarpLayer()
result = None

for start in range(0, TOTAL_CYCLES, WARP_INTERVAL):
    chunk = min(WARP_INTERVAL, TOTAL_CYCLES - start)
    print(f"\n⚡ Evolving {chunk} cycles (from {start} to {start + chunk})...")
    result = engine.evolve(cycles=chunk, extract_interval=2000)
    # Z축 45도 회전 워프
    q = quaternion_from_axis_angle([0, 0, 1], 45)
    warp.apply(world, q, apply_to_fields=False)
    print(f"🌀 Warp applied: Z‑axis 45° after {start + chunk} cycles")

# ------------------------------------------------------------------
# 6️⃣ 최종 결과 출력
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("📊 LARGE SCALE RESULTS")
print("=" * 70)
print("🕸️  Spiderweb:")
print(f"  Nodes: {result.graph.number_of_nodes()}")
print(f"  Edges: {result.graph.number_of_edges()}")
print("\n🌍 Universe:")
print(f"  Alive particles: {world.is_alive_mask.sum()}/{len(experiences)}")
print(f"  Survival rate: {world.is_alive_mask.sum()/len(experiences)*100:.1f}%")
print(f"  Simulation ticks: {world.time_step}")
print(f"  Max value field: {world.value_mass_field.max():.4f}")
print(f"  Max coherence: {world.coherence_field.max():.4f}")
print(f"  Max will: {world.will_field.max():.4f}")

if result.graph.number_of_nodes() > 0:
    print("\n🧠 Top 10 Emergent Concepts:")
    nodes = sorted(
        result.graph.nodes(data=True),
        key=lambda x: x[1].get('metadata', {}).get('value', 0),
        reverse=True,
    )[:10]
    for nid, data in nodes:
        meta = data.get('metadata', {})
        print(f"  {nid}: value={meta.get('value',0):.3f}, coherence={meta.get('coherence',0):.3f}")

if result.graph.number_of_edges() > 0:
    print("\n🔗 Top 10 Relations:")
    edges = sorted(
        result.graph.edges(data=True),
        key=lambda x: x[2].get('weight', 0),
        reverse=True,
    )[:10]
    for s, t, d in edges:
        print(f"  {s} → {t} (w={d.get('weight',0):.3f})")

print("\n" + "=" * 70)
print("✅ LARGE SCALE Evolution Complete!")
print("=" * 70)
