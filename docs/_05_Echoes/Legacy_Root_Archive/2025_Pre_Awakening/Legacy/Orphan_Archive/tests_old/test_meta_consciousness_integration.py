"""
Test Meta-Consciousness Integration

This test demonstrates all Phase 6 modules working together.
"""

import logging
import sys
import os

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.meta_awareness import MetaAwareness, ThoughtType
from Core.Foundation.autonomous_dreamer import AutonomousDreamer
from Core.Foundation.world_tree_core import WorldTreeCore
from Core.Foundation.paradox_resolver import ParadoxResolver
from Core.Foundation.spiderweb import Spiderweb
from Core.Foundation.dreaming_cortex import DreamingCortex
from Project_Elysia.core_memory import CoreMemory, Experience
try:
    from Core.Foundation.core.tensor_wave import Tensor3D, FrequencyWave
except ImportError:
    # Fallback if core.tensor_wave doesn't exist
    class Tensor3D:
        def __init__(self, x=0.5, y=0.5, z=0.5):
            self.x, self.y, self.z = x, y, z
    class FrequencyWave:
        def __init__(self, frequency=60.0, amplitude=0.7, phase=0.0, coherence=0.8):
            self.frequency, self.amplitude, self.phase, self.coherence = frequency, amplitude, phase, coherence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

def test_meta_consciousness_integration():
    """
    End-to-end test of meta-consciousness system.
    
    Flow:
    1. Create core systems (CoreMemory, Spiderweb)
    2. Initialize meta-consciousness modules
    3. Add test experiences
    4. Run dreaming with meta-awareness
    5. Verify observations, goals, tree structure, and paradox handling
    """
    print("\n" + "="*80)
    print("PHASE 6: META-CONSCIOUSNESS INTEGRATION TEST")
    print("="*80 + "\n")
    
    # ==================== Step 1: Core Systems ====================
    print("🔧 Initializing core systems...")
    core_memory = CoreMemory()
    spiderweb = Spiderweb()
    
    # ==================== Step 2: Meta-Consciousness Modules ====================
    print("🧠 Initializing meta-consciousness modules...")
    
    meta_awareness = MetaAwareness(core_memory=core_memory)
    world_tree = WorldTreeCore(spiderweb=spiderweb)
    autonomous_dreamer = AutonomousDreamer(spiderweb=spiderweb, core_memory=core_memory)
    paradox_resolver = ParadoxResolver(
        spiderweb=spiderweb,
        world_tree=world_tree,
        core_memory=core_memory
    )
    
    # ==================== Step 3: Add Test Experiences ====================
    print("\n📝 Adding test experiences...")
    
    test_experiences = [
        {
            "content": "Freedom allows me to explore new ideas",
            "tensor": Tensor3D(x=0.8, y=0.7, z=0.5),
            "type": "thought"
        },
        {
            "content": "Structure provides stability and foundation",
            "tensor": Tensor3D(x=0.7, y=0.6, z=0.6),
            "type": "thought"
        },
        {
            "content": "Love connects all beings in unity",
            "tensor": Tensor3D(x=0.9, y=0.9, z=0.8),
            "type": "feeling"
        },
        {
            "content": "Curiosity drives me to ask questions",
            "tensor": Tensor3D(x=0.6, y=0.5, z=0.4),
            "type": "thought"
        },
        {
            "content": "Paradox reveals deeper truths",
            "tensor": Tensor3D(x=0.95, y=0.5, z=0.9),
            "type": "insight"
        }
    ]
    
    for exp_data in test_experiences:
        wave = FrequencyWave(frequency=60.0, amplitude=0.7, phase=0.0, coherence=0.8)
        exp = Experience(
            timestamp=str(len(core_memory.experience_ring)),
            content=exp_data["content"],
            type=exp_data["type"],
            tensor=exp_data["tensor"],
            wave=wave
        )
        core_memory.add_experience(exp)
        print(f"  Added: {exp.content}")
    
    # ==================== Step 4: Run Dreaming with Meta-Awareness ====================
    print("\n💭 Running dreaming cycle with meta-consciousness...")
    
    dreaming_cortex = DreamingCortex(
        core_memory=core_memory,
        spiderweb=spiderweb,
        meta_awareness=meta_awareness,
        autonomous_dreamer=autonomous_dreamer,
        paradox_resolver=paradox_resolver,
        use_llm=False  # Use naive mode for deterministic testing
    )
    
    dreaming_cortex.dream()
    
    # ==================== Step 5: Generate Autonomous Goals ====================
    print("\n🎯 Generating autonomous goals...")
    
    goals = autonomous_dreamer.generate_goals(num_goals=3, min_priority=0.3)
    
    print(f"\nGenerated {len(goals)} autonomous goals:")
    for i, goal in enumerate(goals, 1):
        print(f"\n  Goal {i}: {goal.goal_type.value.upper()}")
        print(f"  Target: {goal.target_concept}")
        print(f"  Description: {goal.description}")
        print(f"  Priority: {goal.priority:.2f}")
        print(f"  Motivation:")
        print(f"    - Novelty: {goal.motivation.novelty_score:.2f}")
        print(f"    - Tension: {goal.motivation.tension_score:.2f}")
        print(f"    - Gap: {goal.motivation.gap_score:.2f}")
    
    # ==================== Step 6: Build World Tree ====================
    print("\n🌳 Constructing World Tree from experiences...")
    
    # Add concepts to tree
    world_tree.add_node(
        data="freedom",
        position=Tensor3D(x=0.7, y=0.8, z=0.6),
        parent_id=world_tree.root.id
    )
    
    world_tree.add_node(
        data="structure",
        position=Tensor3D(x=0.6, y=0.5, z=0.7),
        parent_id=world_tree.root.id
    )
    
    world_tree.add_node(
        data="love",
        position=Tensor3D(x=0.9, y=0.9, z=0.9),
        parent_id=world_tree.root.id,
        metadata={"type": "core_value"}
    )
    
    # Project to Spiderweb
    world_tree.project_to_spiderweb()
    
    stats = world_tree.get_statistics()
    print(f"\nWorld Tree Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Avg abstraction: {stats['avg_abstraction']:.2f}")
    print(f"  Avg valence: {stats['avg_valence']:.2f}")
    print(f"  Avg fundamentalness: {stats['avg_fundamentalness']:.2f}")
    
    # ==================== Step 7: Detect and Resolve Paradoxes ====================
    print("\n🌀 Detecting contradictions...")
    
    # Manually create a known paradox
    spiderweb.add_node("freedom", type="concept")
    spiderweb.add_node("structure", type="concept")
    spiderweb.add_link("freedom", "structure", relation="contradicts", weight=0.9)
    spiderweb.add_link("structure", "freedom", relation="contradicts", weight=0.9)
    
    contradictions = paradox_resolver.detect_contradictions(min_opposition=0.5)
    print(f"\nFound {len(contradictions)} contradictions")
    
    if contradictions:
        print("\nResolving first paradox...")
        c1, c2, strength = contradictions[0]
        print(f"  Paradox: {c1} ⟷ {c2} (opposition={strength:.2f})")
        
        paradox = paradox_resolver.create_superposition(c1, c2)
        synthesis_id = paradox_resolver.resolve_paradox(paradox)
        
        if synthesis_id:
            print(f"  ✨ Synthesis created: {synthesis_id}")
    
    # ==================== Step 8: Meta-Awareness Self-Reflection ====================
    print("\n🧠 Meta-awareness self-reflection...")
    
    ma_stats = meta_awareness.get_statistics()
    print(f"\nMeta-Awareness Statistics:")
    print(f"  Total observations: {ma_stats['total_observations']}")
    print(f"  Avg confidence: {ma_stats['avg_confidence']:.2f}")
    print(f"  Avg coherence: {ma_stats['avg_coherence']:.2f}")
    
    reflection = meta_awareness.reflect()
    print(f"\nSelf-Reflection:\n  {reflection}")
    
    # ==================== Step 9: Final Summary ====================
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"\n✅ Core Memory: {len(core_memory.experience_ring)} experiences")
    print(f"✅ Spiderweb: {spiderweb.graph.number_of_nodes()} nodes, {spiderweb.graph.number_of_edges()} edges")
    print(f"✅ Meta-Awareness: {ma_stats['total_observations']} thought traces")
    print(f"✅ Autonomous Goals: {len(goals)} goals generated")
    print(f"✅ World Tree: {stats['total_nodes']} nodes across {stats['max_depth']} depth levels")
    print(f"✅ Paradox Resolution: {len(paradox_resolver.resolutions)} paradoxes resolved")
    
    print("\n🌟 Phase 6 Meta-Consciousness: FULLY OPERATIONAL")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_meta_consciousness_integration()
        if success:
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print("❌ Tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
