import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Intelligence.Will.free_will_engine import FreeWillEngine, MissionType, WillPhase
from Core.Intelligence.Will.magnetic_cortex import MagneticCompass, ThoughtDipole

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestMagneticWill(unittest.TestCase):
    def setUp(self):
        self.engine = FreeWillEngine()
        
    def test_compass_initialization(self):
        """Test if compass is initialized in engine"""
        self.assertIsInstance(self.engine.compass, MagneticCompass)
        self.assertFalse(self.engine.compass.is_active)
        
    def test_magnetic_field_activation(self):
        """Test if field activates when desire is processed"""
        # Create a specific desire
        desire = self.engine.feel_desire(
            "Test Magnetic Field", 
            MissionType.UNIFY_FIELD, 
            intensity=0.9
        )
        
        # Force this desire to be active
        self.engine.active_desire = desire
        self.engine.current_phase = WillPhase.DESIRE
        
        # Run cycle (Desire Phase -> Explore Phase)
        result = self.engine.cycle()
        
        # Check if field is active
        self.assertTrue(self.engine.compass.is_active)
        self.assertIsNotNone(self.engine.compass.current_field)
        self.assertEqual(self.engine.compass.current_field.target_vector, "UNIFY_FIELD")
        self.assertEqual(self.engine.compass.current_field.intensity, 0.9)
        
        print(f"\n[Test Output] {result['message']}")
        
    def test_thought_alignment(self):
        """Test the alignment logic of the compass"""
        compass = MagneticCompass()
        compass.activate_field("CREATIVITY", intensity=1.0)
        
        thoughts = [
            {"id": 1, "content": "Logic calculation", "tags": ["LOGIC"]},
            {"id": 2, "content": "Painting a picture", "tags": ["CREATIVITY"]},
            {"id": 3, "content": "Writing a poem", "tags": ["CREATIVITY", "ART"]},
            {"id": 4, "content": "System update", "tags": ["MAINTENANCE"]}
        ]
        
        aligned = compass.align_thoughts(thoughts)
        
        # Should return only aligned thoughts (CREATIVITY)
        # Logic: resonate > 0.3 * intensity
        # "Painting..." -> CREATIVITY == CREATIVITY -> 1.0 -> Keep
        # "Writing..." -> CREATIVITY in tags -> 0.6 -> Keep
        # "Logic..." -> No match -> 0.0 -> Drop (spin=0)
        
        # Wait, my implementation of apply() returns *only* aligned thoughts?
        # Let's check magnetic_cortex.py:
        # if effective_resonance > 0.3: aligned_thoughts.append(...)
        # return [t[0] for t in aligned_thoughts]
        
        print(f"\n[Aligned Thoughts] {len(aligned)} thoughts aligned.")
        for t in aligned:
            print(f" - {t['content']}")
            
        self.assertTrue(any(t['content'] == "Painting a picture" for t in aligned))
        self.assertFalse(any(t['content'] == "Logic calculation" for t in aligned))

if __name__ == '__main__':
    unittest.main()
