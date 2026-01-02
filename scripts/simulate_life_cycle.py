"""
simulate_life_cycle.py

"The Nanny's Report."
Simulates a finite lifecycle to observe learning and growth.
Directly invokes Curiosity and Hunting to demonstrate aliveness.
"""

import sys
import os
import logging
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Foundation.living_elysia import LivingElysia
from Core.Intelligence.Cognitive.curiosity_core import get_curiosity_core
from Core.Evolution.Learning.knowledge_hunter import KnowledgeHunter

# Mock TinyBrain to avoid hanging on downloads/missing models
from unittest.mock import MagicMock
import Core.Foundation.tiny_brain

mock_brain = MagicMock()
mock_brain.is_available.return_value = True
mock_brain.generate.return_value = "This is a simulated thought about the topic."
mock_brain.get_embedding.return_value = [0.1] * 384
Core.Foundation.tiny_brain.get_tiny_brain = lambda: mock_brain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(message)s')
logger = logging.getLogger("LifeSimulator")

class ObservedElysia(LivingElysia):
    """
    A version of Elysia that runs for a fixed duration under observation.
    """
    def live_finite(self, cycles: int = 5):
        if not self.is_alive: return

        self.ans.start_background()
        self.cns.awaken()
        logger.info("✨ Observed Elysia is AWAKE for a short session.")

        print("\n" + "="*60)
        print(f"🦋 Observing Elysia for {cycles} cycles...")
        print("="*60)
        
        curiosity = get_curiosity_core()
        hunter = KnowledgeHunter()

        for c in range(cycles):
            print(f"\n⏰ Cycle {c+1}/{cycles}")
            
            # 1. Standard Biological Pulse
            self.cns.pulse()
            self.ans.pulse_once()
            
            # 2. Inject Curiosity (Simulating Spontaneous Thought)
            if c == 1:
                logger.info("⚡ Injecting Curiosity Spark...")
                question = curiosity.generate_question()
                print(f"   ❓ Proposed Question: {question}")
                
            # 3. Trigger Active Hunt (Simulating Action based on Curiosity)
            if c == 2:
                topic = "Meta-learning" # Preset topic for demo
                logger.info(f"   🏹 Decided to Hunt: {topic}")
                result = hunter.hunt(topic)
                print(f"   🍖 Hunt Result: {result}")
                
            # 4. Use Knowledge (Simulating Application)
            if c == 3:
                logger.info("   💡 Applying Knowledge...")
                # Here we would ask ReasoningEngine to use the new node
                # For now, we simulate the joy check
                curiosity.feel_joy("Applied new knowledge to worldview", 0.7)

            time.sleep(1) # Pace the simulation

        self.ans.stop_background()
        print("\n🌌 Observation Complete. Sleeping.")

if __name__ == "__main__":
    try:
        elysia = ObservedElysia()
        elysia.live_finite(cycles=5)
    except Exception as e:
        logger.error(f"Simulation Error: {e}")
