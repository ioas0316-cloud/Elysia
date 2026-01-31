"""
Test Synesthesia Upgrade (High-Res Qualia)
==========================================
tests/test_synesthesia_upgrade.py

Verifies that SynesthesiaEngine can correctly digest:
1. BridgeSynapse (Audio Causality)
2. FlowCausality (Voice Emotion)
"""

import sys
import os
import unittest
from dataclasses import dataclass

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L3_Phenomena.synesthesia_engine import SynesthesiaEngine, SignalType

# Mock Data Structures (to avoid loading full ML modules)
@dataclass
class BridgeSynapse:
    layer_idx: int
    head_idx: int
    audio_time_idx: int
    token_idx: int
    attention_weight: float

@dataclass
class FlowCausality:
    vector_dim: int
    impact_score: float
    primary_effect: str

@dataclass
class SpacetimeCausality:
    token: str
    time_frame: int
    spatial_region: str
    intensity: float

class TestSynesthesiaUpgrade(unittest.TestCase):
    def setUp(self):
        self.engine = SynesthesiaEngine()

    def test_digested_audio(self):
        """Test conversion of Audio Causality to Signal"""
        print("\n🧪 Testing Audio Digestion...")
        # ... (Existing Audio Test)
        synapse = BridgeSynapse(31,0,50,12,0.95)
        signal = self.engine.from_digested_audio(synapse)
        self.assertEqual(signal.original_type, SignalType.AUDITORY)
        self.assertEqual(signal.payload['target_token'], 12)

    def test_digested_voice(self):
        """Test conversion of Voice Emotion to Signal"""
        print("\n🧪 Testing Voice Digestion...")
        # ... (Existing Voice Test)
        flow = FlowCausality(7,0.88,"Pitch/Tone")
        signal = self.engine.from_digested_voice(flow)
        self.assertEqual(signal.original_type, SignalType.EMOTIONAL)
    
    def test_digested_vision(self):
        """Test conversion of Spacetime Causality to Signal"""
        print("\n🧪 Testing Visual Digestion...")
        
        # Mocking a spacetime event: Token "Cat" caused Frame 24 Center activation
        spacetime = SpacetimeCausality(
            token="Cat",
            time_frame=24,
            spatial_region="Center",
            intensity=0.85
        )
        
        signal = self.engine.from_digested_vision(spacetime)
        
        print(f"   Input: Spacetime(Token={spacetime.token}, Time={spacetime.time_frame})")
        print(f"   Output: Signal(Freq={signal.frequency}, Amp={signal.amplitude})")
        
        self.assertEqual(signal.original_type, SignalType.VISUAL)
        self.assertEqual(signal.payload['token'], "Cat")
        self.assertEqual(signal.payload['time'], 24)
        # Check Frequency mapping: 200 + (24 * 5) = 320
        self.assertEqual(signal.frequency, 320.0)

if __name__ == '__main__':
    unittest.main()
