import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Life.medical_cortex import MedicalCortex

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestMedicalCortex(unittest.TestCase):
    def setUp(self):
        self.cortex = MedicalCortex()
        
    def test_parkinsons_support(self):
        """Test RAS generation for Parkinson's"""
        print("\n[Test] Parkinson's Support (RAS)")
        
        # 1. Register Profile
        self.cortex.register_profile("Uncle", "Parkinson's", "Mild tremors")
        
        # 2. Prescribe for 'Freezing' state
        therapy = self.cortex.prescribe_therapy("Uncle", "Experiencing freezing gait")
        
        print(f" - Condition: Freezing Gait")
        print(f" - Prescription: {therapy['description']}")
        print(f" - BPM: {therapy['bpm']}")
        
        self.assertEqual(therapy['type'], "RAS_METRONOME")
        self.assertEqual(therapy['bpm'], 90) # Freezing should trigger faster beat
        
    def test_pregnancy_support(self):
        """Test Binaural Beats for Pregnancy"""
        print("\n[Test] Pregnancy Support (Bio-Resonance)")
        
        # 1. Register Profile
        self.cortex.register_profile("Sister", "Pregnancy", "3rd Trimester")
        
        # 2. Prescribe for 'Anxiety' state
        therapy = self.cortex.prescribe_therapy("Sister", "Feeling anxious about future")
        
        print(f" - Condition: Anxiety")
        print(f" - Prescription: {therapy['description']}")
        print(f" - Effect: {therapy['effect']}")
        
        self.assertEqual(therapy['type'], "BINAURAL_BEAT")
        self.assertTrue(therapy['beat_freq'] >= 8.0) # Alpha wave range
        
        # 3. Prescribe for 'Rest' state (Lullaby)
        therapy_rest = self.cortex.prescribe_therapy("Sister", "Just relaxing")
        print(f" - Condition: Relaxing")
        print(f" - Prescription: {therapy_rest['description']}")
        self.assertEqual(therapy_rest['type'], "LULLABY_432HZ")

if __name__ == '__main__':
    unittest.main()
