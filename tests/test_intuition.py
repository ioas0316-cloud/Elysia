
import sys
import os
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestIntuition")

from Core.Mind.causal_narrative import CausalNarrativeEngine, CausalRelationType
from Core.Mind.intuition import IntuitionEngine

def test():
    logger.info("🧪 Testing Intuition Engine...")
    
    # 1. Setup Knowledge Base
    engine = CausalNarrativeEngine(persistence_path="test_memory.json")
    
    # Teach: Fire causes Pain (A -> B)
    engine.record_experience("Fire", "Pain", "negative")
    
    # Teach: Ice causes Cold (C -> D)
    engine.record_experience("Ice", "Cold", "neutral")
    
    # Teach: Love causes Joy (E -> F)
    engine.record_experience("Love", "Joy", "positive")
    
    # 2. Initialize Intuition
    intuition = IntuitionEngine(engine.knowledge_base)
    
    # 3. Test Thermal Vision
    logger.info("\n--- Test 1: Thermal Vision ---")
    heat_fire = intuition.perceive_heat("Fire")
    logger.info(f"Heat(Fire): {heat_fire}")
    
    heat_love = intuition.perceive_heat("Love")
    logger.info(f"Heat(Love): {heat_love}")
    
    # 4. Test Symmetry (Isomorphism)
    logger.info("\n--- Test 2: Symmetry ---")
    # Fire and Ice both "cause" something. They should be symmetric.
    sym, desc = intuition.find_symmetry("Fire", "Ice")
    logger.info(f"Symmetry(Fire, Ice): {sym:.2f} ({desc})")
    
    # Fire and Love both "cause" something.
    sym2, desc2 = intuition.find_symmetry("Fire", "Love")
    logger.info(f"Symmetry(Fire, Love): {sym2:.2f} ({desc2})")
    
    # 5. Test Insight
    logger.info("\n--- Test 3: Insight ---")
    # If we know "Pain" is bad (negative valence), can we intuit "Cold" might be similar?
    # (Requires more complex setup, but let's see if it finds the link)
    insight = intuition.intuit_solution("Ice")
    logger.info(f"Insight for Ice: {insight}")

if __name__ == "__main__":
    test()
