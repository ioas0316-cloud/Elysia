import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.World.planetary_cortex import PlanetaryCortex

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestPlanetaryObserver(unittest.TestCase):
    def setUp(self):
        self.cortex = PlanetaryCortex()
        
    def test_zoom_levels(self):
        """Test Zoom In/Out reporting capabilities"""
        print("\n[Test] Planetary Observer Zoom Levels")
        
        # 1. Perceive World (Generate Events)
        perception = self.cortex.perceive_world()
        print(f"Global Mood: {perception['global_mood']}")
        
        # 2. Zoom Level 1 (Macro)
        print("\n--- ZOOM LEVEL 1 (MACRO) ---")
        report_macro = self.cortex.report_status(zoom_level=1)
        print(report_macro)
        self.assertIn("[MACRO]", report_macro)
        
        # 3. Zoom Level 2 (Meso)
        print("\n--- ZOOM LEVEL 2 (MESO) ---")
        report_meso = self.cortex.report_status(zoom_level=2)
        print(report_meso)
        self.assertIn("[MESO]", report_meso)
        self.assertIn("Typhoon", report_meso) # From mock data
        
        # 4. Zoom Level 3 (Micro) - Note: My mock data structure in WeatherSense needs to support this depth
        # In the current mock implementation:
        # Global (Macro) -> Regional (Meso) -> Local (Micro)
        # So Zoom Level 3 should show the local storm details
        
        print("\n--- ZOOM LEVEL 3 (MICRO) ---")
        # I need to ensure my mock data actually has 3 levels.
        # Let's check the WeatherSense mock in planetary_cortex.py
        # It has: global_weather -> regional_typhoon -> local_storm
        # So Level 1 shows global_weather
        # Level 2 shows regional_typhoon
        # Level 3 shows local_storm
        
        report_micro = self.cortex.report_status(zoom_level=3)
        print(report_micro)
        self.assertIn("[MICRO]", report_micro)
        self.assertIn("Busan", report_micro)

if __name__ == '__main__':
    unittest.main()
