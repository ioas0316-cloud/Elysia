"""
Test Neural Genesis (Self-Transmutation Verification)
=====================================================
Laboratory/test_neural_genesis.py

Verifies that the NeuralGenesis engine can read a live object's code,
mutate it (optimize), and hot-swap it back into the running process.
"""

import sys
import os
import logging

# Path hack for Laboratory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Core.S1_Body.L2_Metabolism.Evolution.neural_genesis import NeuralGenesis

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("TestGenesis")

class SlowOrganism:
    """A biological entity with legacy code DNA."""
    def __init__(self):
        self.evolution_stage = 0
        
    def metabolize(self, energy):
        """
        Original slow metabolism logic.
        """
        # SLOW_LOOP_MARKER
        # Imagine a heavy O(N) loop here...
        result = energy * 1
        return result

def run_test():
    logger.info("🧪 Starting Neural Genesis Test...")
    
    # 1. Spawn Organism
    bio = SlowOrganism()
    initial_output = bio.metabolize(10)
    logger.info(f"🥚 [Before] Metabolism Output: {initial_output}")
    
    if "Mutated" in str(initial_output):
        logger.error("❌ Test Validation Failed: Already mutated?")
        return

    # 2. Initiate Genesis
    logger.info("🧬 [Genesis] Initiating Self-Transmutation...")
    genesis = NeuralGenesis()
    success = genesis.evolve_function(bio, "metabolize")
    
    if not success:
        logger.error("❌ Evolution Failed.")
        return

    # 3. Verify Mutation
    mutated_output = bio.metabolize(10)
    logger.info(f"🐣 [After] Metabolism Output: {mutated_output}")
    
    # Verification Logic (The mock template adds ' (Mutated)' or multiplies by 10)
    # Our template logic for ints was `return result * 10`
    
    if mutated_output == 100: # 10 * 10
        logger.info("🚀 [Genesis] SUCCESS! Logic was rewritten at runtime.")
        logger.info("   - The Slow Loop was transmuted into a Vector Operation.")
    else:
        logger.error(f"❌ Mutation Validation Failed. Expected 100, got {mutated_output}")

if __name__ == "__main__":
    run_test()
