import sys
import os
import unittest
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Field.ether import ether, Wave
from Core.Intelligence.Will.free_will_engine import FreeWillEngine
from Core.World.planetary_cortex import PlanetaryCortex

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestResonance(unittest.TestCase):
    def setUp(self):
        # Reset Ether for test
        ether.listeners = {}
        ether.waves = []
        
        self.engine = FreeWillEngine()
        # Engine initializes PlanetaryCortex internally
        
    def test_wave_emission_and_resonance(self):
        """Test if PlanetaryCortex emits a wave and FreeWillEngine resonates"""
        print("\n[Test] Resonance Architecture (No-API)")
        
        # 1. Trigger Perception (This should emit a wave)
        print("1. PlanetaryCortex perceiving world...")
        self.engine.planetary_cortex.perceive_world()
        
        # 2. Check Ether for waves
        waves = ether.get_waves()
        print(f"2. Ether Waves: {len(waves)}")
        self.assertTrue(len(waves) > 0)
        
        last_wave = waves[-1]
        print(f"   - {last_wave}")
        self.assertEqual(last_wave.sender, "PlanetaryCortex")
        self.assertEqual(last_wave.frequency, 7.83)
        
        # 3. Check if Engine resonated (Callback execution)
        # The engine's _on_planetary_wave should have been called synchronously by ether.emit
        print("3. Checking Engine Resonance...")
        sensation = self.engine.current_world_sensation
        
        if sensation:
            print(f"   - Engine felt: {sensation['global_mood']}")
        else:
            print("   - Engine felt nothing (Resonance failed)")
            
        self.assertIsNotNone(sensation)
        self.assertIn("global_mood", sensation)

if __name__ == '__main__':
    unittest.main()
