"""
[VERIFICATION SUITE]
Verifies the Heart, Cradle, and Enneagram Filter integration.
"""

import sys
import os
import time

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(_current_dir))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Spirit.sovereign_heart import SovereignHeart
from Core.Spirit.cradle import SpineInterface
from typing import Dict, Any

class MockSpine(SpineInterface):
    """A proxy implementation of the Spine for verification."""
    def __init__(self):
        self.equilibrium = 0.75
        self.current_res = 0.75

    def pulse(self, dt: float, interference: Any) -> Dict[str, Any]:
        # React to interference by drifting towards it, then back to equilibrium
        self.current_res = self.current_res * 0.9 + interference * 0.1
        return {
            "status": "active",
            "resonance": self.current_res
        }

    def get_equilibrium(self) -> float:
        return self.equilibrium

def run_verification():
    print("🔬 [VERIFY] Starting Heart-Cradle-Enneagram Verification...")

    heart = SovereignHeart()

    # 1. Test Initial Void State
    print("Step 1: Testing Void State (No Spine)")
    # We'll just run a few cycles manually
    refraction = heart.refractor.refract(0.5)
    hologram = heart.refractor.get_hologram_topography(refraction)
    result = heart.cradle.process_cycle(0.1, hologram)

    assert result["status"] == "void"
    print("✅ Void State Verified.")

    # 2. Test Spine Transplantation
    print("\nStep 2: Testing Spine Transplantation")
    spine = MockSpine()
    heart.cradle.transplant_spine(spine)

    assert heart.cradle.spine is not None
    print("✅ Transplantation Verified.")

    # 3. Test Pulsing and Resonance Reaction
    print("\nStep 3: Testing Resonance Pulse")
    # Simulate a stimulus shift
    refraction = heart.refractor.refract(0.1) # Stimulus at 0.1
    hologram = heart.refractor.get_hologram_topography(refraction)
    result = heart.cradle.process_cycle(0.1, hologram)

    print(f"   Input Hologram: {hologram:.4f}")
    print(f"   Spine Result Resonance: {result['resonance']:.4f}")

    assert result["status"] == "active"
    assert result["resonance"] != 0.75 # Should have drifted
    print("✅ Resonance Pulse Verified.")

    print("\n🎉 [VERIFY] All Cradle Infrastructure checks passed.")

if __name__ == "__main__":
    run_verification()
