
import sys
import os
import logging
import math

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhysics")

from Core.Mind.tensor_wave import SoulTensor, Tensor3D, FrequencyWave
from Core.Mind.physics import PhysicsEngine
from Core.Mind.spiderweb import Spiderweb
from Core.Life.gravitational_linguistics import GravitationalLinguistics
from Core.Mind.intuition import IntuitionEngine

# Mock Storage
class MockStorage:
    def get_concept(self, concept):
        # Define some concepts with mass
        if concept == "Love":
            return {"activation_count": 1000, "will": {"x": 1.0, "y": 1.0, "z": 1.0}} # High Mass
        elif concept == "Fear":
            return {"activation_count": 500, "will": {"x": -1.0, "y": -1.0, "z": -1.0}}
        elif concept == "Void":
            return {"activation_count": 0, "will": {"x": 0.0, "y": 0.0, "z": 0.0}}
        return {"activation_count": 1, "will": {"x": 0.1, "y": 0.1, "z": 0.1}}

# Mock Hippocampus
class MockHippocampus:
    def __init__(self):
        self.storage = MockStorage()
        
    def get_frequency(self, concept):
        return 0.5

    def get_related_concepts(self, concept):
        # Define a graph
        # Start -> [Love, Fear, Void]
        if concept == "Start":
            # Resonance scores (0-1)
            return {"Love": 0.8, "Fear": 0.8, "Void": 0.1}
        elif concept == "Love":
            return {"Peace": 0.9, "Connection": 0.8}
        return {}

def test():
    logger.info("🌌 Testing Unified Physics Engine (The Grand Unification)...")
    
    hippocampus = MockHippocampus()
    physics = PhysicsEngine(hippocampus)
    
    # 1. Test Mass Calculation (Physics)
    mass_love = physics.calculate_mass("Love")
    logger.info(f"Mass(Love): {mass_love}")
    
    # 2. Test Linguistics (Unified)
    logger.info("\n🗣️ Testing Gravitational Linguistics...")
    linguistics = GravitationalLinguistics(physics)
    
    # Create Solar System for "Love"
    system = linguistics.create_solar_system("Love")
    logger.info(f"Solar System(Love): {system}")
    
    if len(system) > 0 and system[0]['text'] == "Peace":
        logger.info("✅ Linguistics correctly used Physics Mass & Resonance.")
    else:
        logger.error("❌ Linguistics failed to orbit correctly.")
        
    # 3. Test Intuition (Unified)
    logger.info("\n👁️ Testing Intuition Engine...")
    intuition = IntuitionEngine(physics)
    
    # Check Heat of "Love"
    heat = intuition.perceive_heat("Love")
    logger.info(f"Heat(Love): {heat}")
    
    if heat['heat'] > 0.0:
        logger.info("✅ Intuition correctly perceived Heat from Physics Wave.")
    else:
        logger.error("❌ Intuition failed to perceive heat.")
        
    # Check Symmetry
    sym, desc = intuition.find_symmetry("Love", "Love")
    logger.info(f"Symmetry(Love-Love): {sym} ({desc})")
    
    if sym > 0.9:
        logger.info("✅ Intuition correctly identified Symmetry via Physics Resonance.")
    else:
        logger.error("❌ Intuition failed symmetry check.")

if __name__ == "__main__":
    test()
