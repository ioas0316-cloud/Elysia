
import sys
import os
import logging
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestElysia")

from Core.Elysia import Elysia

def test():
    logger.info("🌌 Testing Elysia Full Integration...")
    
    elysia = Elysia()
    elysia.awaken()
    
    # Simulate User Input
    logger.info("\n🗣️ User says: 'Life'")
    response = elysia.talk("Life")
    
    logger.info(f"\n🤖 Elysia says: {response}")
    
    logger.info("\n🌳 World Tree State:")
    print(elysia.world_tree.render_ascii())

if __name__ == "__main__":
    test()
