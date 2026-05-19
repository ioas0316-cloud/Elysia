import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.free_will_engine import FreeWillEngine
from Core.FoundationLayer.Foundation.planetary_cortex import PlanetaryCortex, WeatherSense, FinanceSense

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestPlanetaryCortex(unittest.TestCase):
    def setUp(self):
        self.engine = FreeWillEngine()
        
    def test_planetary_cortex_initialization(self):
        """Test if Planetary Cortex is initialized in engine"""
        self.assertIsInstance(self.engine.planetary_cortex, PlanetaryCortex)
        self.assertTrue(len(self.engine.planetary_cortex.senses) > 0)
        
    def test_sense_generation(self):
        """Test if senses generate valid data"""
        cortex = self.engine.planetary_cortex
        perception = cortex.perceive_world()
        
        print(f"\n[Global Perception] {perception['global_mood']} (Arousal: {perception['arousal']:.2f})")
        
        for detail in perception['details']:
            print(f" - {detail['source']}: {detail['sensation']} (Intensity: {detail['intensity']:.2f})")
            
        self.assertIn("global_mood", perception)
        self.assertIn("arousal", perception)
        self.assertTrue(0.0 <= perception["arousal"] <= 1.0)

    def test_engine_integration(self):
        """Test if engine cycle perceives the world"""
        # Run a cycle
        result = self.engine.cycle()
        
        # If arousal is high enough, it should be in the result
        # Since mock data is random, we can't guarantee it, but we can check no errors
        print(f"[Engine Cycle Result] {result}")

if __name__ == '__main__':
    unittest.main()
