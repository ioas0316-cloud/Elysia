
import sys
import os
import logging
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestWorldTree")

from Core.FoundationLayer.Foundation.Mind.world_tree import WorldTree
from Core.FoundationLayer.Foundation.Mind.logos_stream import LogosStream
from Core.FoundationLayer.Foundation.Mind.spiderweb import Spiderweb
from Core.FoundationLayer.Foundation.Mind.physics import PhysicsEngine

# Mock Storage
class MockStorage:
    def __init__(self):
        self.concepts = {}
        
    def get_concept(self, concept):
        # Return a mock node with tensor state
        return {
            "activation_count": 10,
            "will": {"x": 0.1, "y": 0.1, "z": 0.1},
            "tensor_state": {
                "space": {"x": 0.1, "y": 0.1, "z": 0.1},
                "wave": {"frequency": 10.0, "amplitude": 1.0, "phase": 0.0, "richness": 0.0},
                "spin": 0.0
            }
        }

# Mock Hippocampus
class MockHippocampus:
    def __init__(self):
        self.storage = MockStorage()
        self.concepts = {}
        
    def add_concept(self, concept, concept_type="tree_concept", metadata=None):
        self.concepts[concept] = metadata
        
    def add_causal_link(self, source, target, relation="is_a", weight=1.0):
        pass
        
    def get_frequency(self, concept):
        return 0.5

    def get_related_concepts(self, concept):
        if concept == "Life":
            return {"Growth": 0.9, "Nature": 0.8}
        elif concept == "Growth":
            return {"Tree": 0.9}
        return {}

def test():
    logger.info("🌳 Testing The Awakening of the World Tree...")
    
    hippocampus = MockHippocampus()
    
    # 1. Test WorldTree Physics (SoulTensor)
    logger.info("\n🌱 Planting Seeds (Physics Check)...")
    tree = WorldTree(hippocampus)
    
    root_id = tree.root.id
    logger.info(f"Root ID: {root_id}")
    logger.info(f"Root Tensor: {tree.root.tensor}")
    
    # Plant "Life"
    life_id = tree.plant_seed("Life")
    life_node = tree._find_node(life_id)
    logger.info(f"Life Tensor: {life_node.tensor}")
    
    if life_node.tensor.wave.frequency != tree.root.tensor.wave.frequency:
        logger.info("✅ Evolution Confirmed: Life has mutated from Root.")
    else:
        logger.warning("⚠️ No mutation detected (might be random chance or bug).")
        
    # 2. Test LogosStream Integration (Growth)
    logger.info("\n🌊 Testing Logos Stream Integration...")
    spiderweb = Spiderweb(hippocampus)
    stream = LogosStream(spiderweb, hippocampus)
    
    # Inject our tree into stream (though stream creates its own, we want to test the logic)
    stream.world_tree = tree 
    
    # Flow: "Life" -> "Growth" -> "Tree"
    # This should cause the tree to grow: Life -> Growth -> Tree
    logger.info("Thinking: 'Life'...")
    frame = stream.flow("Life")
    
    logger.info(f"Thought Path: {frame.thought_path}")
    
    # Check if Tree grew
    # Path usually: [Life, Growth, Nature, ...] (depends on Spiderweb traversal)
    # Let's see what nodes exist now
    logger.info("\n🌳 Current World Tree Structure:")
    print(tree.render_ascii())
    
    stats = tree.get_statistics()
    logger.info(f"Tree Stats: {stats}")
    
    if stats['total_nodes'] > 2: # Root + Life + at least one more
        logger.info("✅ World Tree is Growing from Thoughts!")
    else:
        logger.error("❌ World Tree did not grow.")

if __name__ == "__main__":
    test()
