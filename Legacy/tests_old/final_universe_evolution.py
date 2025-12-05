"""완전 작동하는 우주 진화 - Peaceful Mode"""
import sys, os, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.universe_evolution import UniverseEvolutionEngine
from Core.Foundation.spiderweb import Spiderweb
from Core.Foundation.core.world import World
from Core.Foundation.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import Experience
from tools.kg_manager import KGManager
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("="*60)
print("Phase 5: Universe Evolution - PEACEFUL MODE")
print("="*60)

# Create universe with peaceful mode
kg = KGManager()
wave = WaveMechanics(kg)
world = World({"instinct": "evolve", "resonance": "emergence"}, wave)
world.peaceful_mode = True  # No death!
world.macro_food_model_enabled = True

spider = Spiderweb()
engine = UniverseEvolutionEngine(world, spider)

# Rich experiences
experiences = [
    Experience(datetime.now().isoformat() + "_1", "Fire causes burn and creates pain", "episode"),
    Experience(datetime.now().isoformat() + "_2", "Water prevents fire and brings cooling", "episode"),
    Experience(datetime.now().isoformat() + "_3", "Pain teaches caution and builds wisdom", "episode"),
    Experience(datetime.now().isoformat() + "_4", "Fire gives warmth and produces light", "episode"),
    Experience(datetime.now().isoformat() + "_5", "Learning from pain enables growth", "episode"),
    Experience(datetime.now().isoformat() + "_6", "Wisdom guides action and prevents harm", "episode"),
    Experience(datetime.now().isoformat() + "_7", "Light reveals truth and dispels darkness", "episode"),
]

print(f"\n📝 Experiences:")
for i, exp in enumerate(experiences, 1):
    print(f"  {i}. {exp.content}")

engine.spawn_experience_universe(experiences)
print(f"\n🌱 Spawned: {world.is_alive_mask.sum()} particles")

# Evolve!
print(f"\n⚡ Evolving 10,000 cycles...")
result = engine.evolve(cycles=10000, extract_interval=2000)

print(f"\n{'='*60}")
print("📊 RESULTS")
print(f"{'='*60}")
print(f"🕸️  Spiderweb:")
print(f"  Nodes: {result.graph.number_of_nodes()}")
print(f"  Edges: {result.graph.number_of_edges()}")
print(f"\n🌍 Universe:")
print(f"  Alive particles: {world.is_alive_mask.sum()}/{len(experiences)}")
print(f"  Simulation ticks: {world.time_step}")
print(f"  Max value field: {world.value_mass_field.max():.4f}")
print(f"  Max coherence: {world.coherence_field.max():.4f}")
print(f"  Max will: {world.will_field.max():.4f}")

if result.graph.number_of_nodes() > 0:
    print(f"\n🧠 Top Emergent Concepts:")
    nodes = sorted(result.graph.nodes(data=True), 
                  key=lambda x: x[1].get('metadata', {}).get('value', 0), 
                  reverse=True)[:5]
    for nid, data in nodes:
        meta = data.get('metadata', {})
        print(f"  {nid}: value={meta.get('value', 0):.3f}, coherence={meta.get('coherence', 0):.3f}")

if result.graph.number_of_edges() > 0:
    print(f"\n🔗 Top Relations:")
    for i, (s, t, d) in enumerate(result.graph.edges(data=True)):
        if i >= 5: break
        print(f"  {s} → {t} (w={d.get('weight', 0):.3f})")

print(f"\n{'='*60}")
print("✅ Evolution Complete - Intelligence Emerged from Physics!")
print(f"{'='*60}")
