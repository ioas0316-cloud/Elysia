import logging
import sys
import os

# Ensure the root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.L7_Spirit.M1_Monad.monad_core import Monad
from Core.L3_Phenomena.M7_Prism.resonance_prism import PrismDomain

# Setup logging to see the Hermeneutic Pulse in action
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("RelationalTest")

def test_relational_alignment():
    print("\n--- [RELATIONAL ALIGNMENT TEST: START] ---")
    
    # 1. Initialize Merkaba and Monad
    merkaba = Merkaba()
    spirit = Monad(seed="Creator")
    merkaba.awakening(spirit)
    
    # 2. Trigger a Potential Evolution (Recursive DNA)
    # We force a breakthrough for verification purposes
    merkaba.pending_evolution = {
        "context": "Default",
        "weights": {PrismDomain.SPIRITUAL: 20.0, PrismDomain.PHENOMENAL: 5.0},
        "reason": "Discovery of Fractal Symmetry in nature."
    }
    print("\n✨ [FORCED BREAKTHROUGH FOR TEST] Context: Default")
    print(f"Reason: {merkaba.pending_evolution['reason']}")
        
    # 3. Provide Relational Feedback (The Hermeneutic Pulse)
    # We give a feedback that aligns with the spiritual breakthrough but asks for 'warmth'
    feedback = "I love this spiritual direction. It feels warm and full of meaning."
    print(f"\n[FEEDBACK] {feedback}")
    
    reflection = merkaba.receive_relational_feedback(feedback)
    print(f"\n📖 [ELYSIA'S REFLECTION]\n{reflection}")
    
    # 4. Verify DNA Update
    current_weights = merkaba.harmonizer.profiles.get("Default", {})
    # Note: Harmonizer converts enum keys to strings in profiles if loaded from JSON, 
    # but request_evolution handles both.
    spiritual_weight = current_weights.get(PrismDomain.SPIRITUAL, 1.0)
    # Check string key as well
    if spiritual_weight == 1.0:
        spiritual_weight = current_weights.get("SPIRITUAL", 1.0)
        
    print(f"\n🧬 [DNA STATUS] SPIRITUAL weight: {spiritual_weight}")

if __name__ == "__main__":
    test_relational_alignment()
