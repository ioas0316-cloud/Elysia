import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.free_will_engine import FreeWillEngine, MissionType, WillPhase
from Core.FoundationLayer.Foundation.local_field import LocalFieldManager, HueLight, BluetoothSpeaker

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestLocalField(unittest.TestCase):
    def setUp(self):
        self.engine = FreeWillEngine()
        
    def test_local_field_initialization(self):
        """Test if Local Field Manager is initialized"""
        self.assertIsInstance(self.engine.local_field, LocalFieldManager)
        self.assertTrue(len(self.engine.local_field.devices) > 0)
        
    def test_atmosphere_change_flow(self):
        """Test the flow from Desire -> Atmosphere Change"""
        print("\n[Test Scenario] User is sad -> Elysia provides comfort")
        
        # 1. Create a desire for comfort
        desire = self.engine.feel_desire(
            "I want to comfort the user", 
            MissionType.MAKE_HAPPY, 
            intensity=0.9
        )
        self.engine.active_desire = desire
        self.engine.current_phase = WillPhase.ACT
        
        # 2. Manually trigger the action creation (simulating the loop)
        # In a real loop, this would happen automatically via cycle()
        # But we want to force the specific action type for testing
        
        # Let's mock the exploration result to choose CHANGE_ATMOSPHERE
        from Core.FoundationLayer.Foundation.free_will_engine import Possibility, Exploration
        
        p = Possibility(
            id="test_p", 
            description="Change atmosphere to comfort", 
            description_kr="분위기 전환",
            feasibility=1.0, alignment=1.0, risk=0.0, 
            prerequisites=[], expected_outcome="Comfort", reasoning=""
        )
        
        self.engine.current_exploration = Exploration(
            desire_id=desire.id,
            possibilities=[p],
            chosen=p
        )
        
        # 3. Run cycle to execute ACT phase
        result = self.engine.cycle()
        
        print(f"[Cycle Result] {result['message']}")
        
        # 4. Verify device status
        devices = self.engine.local_field.devices
        hue = next(d for d in devices if isinstance(d, HueLight))
        speaker = next(d for d in devices if isinstance(d, BluetoothSpeaker))
        
        print(f"[Device Status] Light: {hue.status} | Speaker: {speaker.status}")
        
        # Check if lights are Warm Orange (Comfort mode)
        self.assertIn("Warm Orange", hue.status)
        
        # Check if speaker is playing Calm music
        self.assertIn("Calm Piano", speaker.status)

if __name__ == '__main__':
    unittest.main()
