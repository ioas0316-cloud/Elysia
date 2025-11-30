
import sys
import os
import logging
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLogosStream")

from Core.Mind.logos_stream import LogosStream
from Core.Mind.spiderweb import Spiderweb
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.world_tree import WorldTree

def test():
    logger.info("🌊 Testing Logos Stream Flow...")
    
    hippocampus = Hippocampus()
    spiderweb = Spiderweb(hippocampus)
    from Core.Prediction.predictive_world import PredictiveWorldModel
    predictive_world = PredictiveWorldModel(project_root=os.getcwd())
    stream = LogosStream(spiderweb, hippocampus, predictive_world)
    stream.world_tree = WorldTree(hippocampus)
    
    logger.info("Thinking: 'Life'...")
    start = time.time()
    frame = stream.flow("Life")
    end = time.time()
    
    logger.info(f"Thought Path: {frame.thought_path}")
    logger.info(f"Time Taken: {end - start:.2f}s")
    
    logger.info("\n🌳 World Tree:")
    print(stream.world_tree.render_ascii())

if __name__ == "__main__":
    test()
