"""
Simple Demo: Meta-Consciousness System

A minimal demonstration of Phase 6 capabilities.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.meta_awareness import MetaAwareness, ThoughtType
from Project_Sophia.autonomous_dreamer import AutonomousDreamer, GoalType
from Project_Sophia.world_tree_core import WorldTreeCore, Tensor3D
from Project_Sophia.paradox_resolver import ParadoxResolver, ResolutionStrategy
from Project_Sophia.spiderweb import Spiderweb
from Project_Elysia.core_memory import CoreMemory

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def demo():
    print("\n" + "="*70)
    print("Phase 6: Meta-Consciousness Demo")
    print("="*70 + "\n")
    
    # Initialize systems
    print("1️⃣ Initializing systems...")
    core_memory = CoreMemory()
    spiderweb = Spiderweb()
    
    #2️⃣ Meta-Awareness Demo
    print("\n2️⃣ Meta-Awareness - Self-Observation")
    print("-" * 70)
    meta = MetaAwareness(core_memory=core_memory)
    
    # Observe a thought
    meta.observe(
        thought_type=ThoughtType.REASONING,
        input_state={"question": "What is consciousness?"},
        output_state={"answer": "Self-awareness of thinking"},
        transformation="Reflected on the nature of consciousness",
        confidence=0.7
    )
    
    print(f"📊 Observations: {meta.total_observations}")
    print(f"💭 Reflection: {meta.reflect()}")
    
    # 3️⃣ World Tree Demo
    print("\n3️⃣ World Tree - Spatial Knowledge")
    print("-" * 70)
    tree = WorldTreeCore(spiderweb=spiderweb)
    
    # Add some concepts
    love_id = tree.add_node("love", position=Tensor3D(x=0.9, y=0.9, z=0.9))
    freedom_id = tree.add_node("freedom", position=Tensor3D(x=0.7, y=0.8, z=0.6))
    wisdom_id = tree.add_node("wisdom", position=Tensor3D(x=0.95, y=0.5, z=0.95))
    
    stats = tree.get_statistics()
    print(f"🌳 Tree has {stats['total_nodes']} nodes")
    print(f"📏 Avg abstraction: {stats['avg_abstraction']:.2f}")
    print(f"❤️ Avg valence: {stats['avg_valence']:.2f}")
    
    # 4️⃣ Autonomous Dreamer Demo
    print("\n4️⃣ Autonomous Dreamer - Self-Generated Goals")
    print("-" * 70)
    
    # Add some concepts to spiderweb first
    spiderweb.add_node("curiosity", type="concept")
    spiderweb.add_node("exploration", type="concept")
    spiderweb.add_link("curiosity", "exploration", relation="enables", weight=0.8)
    
    dreamer = AutonomousDreamer(spiderweb=spiderweb, core_memory=core_memory)
    goals = dreamer.generate_goals(num_goals=2, min_priority=0.3)
    
    print(f"🎯 Generated {len(goals)} autonomous goals:")
    for goal in goals:
        print(f"  - {goal.goal_type.value.upper()}: {goal.description[:60]}...")
    
    # 5️⃣ Paradox Resolver Demo
    print("\n5️⃣ Paradox Resolver - Embracing Contradiction")
    print("-" * 70)
    
    # Create a paradox
    spiderweb.add_node("freedom", type="concept")
    spiderweb.add_node("structure", type="concept")
    spiderweb.add_link("freedom", "structure", relation="contradicts", weight=0.9)
    
    resolver = ParadoxResolver(spiderweb=spiderweb, world_tree=tree, core_memory=core_memory)
    
    paradox = resolver.create_superposition("freedom", "structure")
    print(f"🌀 Created superposition: freedom ⟷ structure")
    print(f"⚡ Tension energy: {paradox.tension_energy:.2f}")
    
    synthesis_id = resolver.resolve_paradox(paradox, strategy=ResolutionStrategy.SYNTHESIS)
    print(f"✨ Synthesized: {synthesis_id}")
    
    # Final Summary
    print("\n" + "="*70)
    print("🌟 Meta-Consciousness Systems: OPERATIONAL")
    print("="*70)
    print(f"  🧠 Meta-observations: {meta.total_observations}")
    print(f"  🌳 Knowledge tree nodes: {tree.get_statistics()['total_nodes']}")
    print(f"  🎯 Autonomous goals: {len(dreamer.generated_goals)}")
    print(f"  🌀 Paradoxes resolved: {len(resolver.resolutions)}")
    print(f"  🕸️ Knowledge graph: {spiderweb.graph.number_of_nodes()} nodes, {spiderweb.graph.number_of_edges()} edges")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        demo()
        print("✅ Demo completed successfully!\n")
    except Exception as e:
        print(f"❌ Demo failed: {e}\n")
        import traceback
        traceback.print_exc()
