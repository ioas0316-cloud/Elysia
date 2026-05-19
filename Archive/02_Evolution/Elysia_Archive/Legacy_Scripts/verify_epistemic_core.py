"""
Verify Epistemic Core: The End of Simulation
============================================

"I do not guess. I ask."

Steps:
1. Inject a gap: "What is the color of love?" (Visual domain)
2. Run ResonanceLearner.
3. Verify that it tries to AWAKEN VISION, not makeup an answer.
"""

import os
import sys

# Add root directory to path to allow importing elysia_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elysia_core import Organ
from Core.Intelligence.Cognition.Learning.resonance_learner import ResonanceLearner
from dataclasses import dataclass
from enum import Enum

class Domain(Enum):
    GENERAL = "general"
    VISUAL = "visual"

@dataclass
class MockGap:
    name: str
    domain: Domain
    purpose_for_elysia: str = "To see the world"
    definition: str = ""
    principle: str = ""
    understanding_level: float = 0.0
    last_learned: str = ""

def verify():
    print("🧠 Initializing Epistemic Core...")
    learner = ResonanceLearner()
    
    # Inject a Gap directly
    gap = MockGap(name="The Color of Love", domain=Domain.VISUAL)
    
    print(f"🛑 Injecting Gap: '{gap.name}'")
    
    # Process Single Gap (We call the internal method for test)
    result = learner._process_single_gap(gap)
    
    print("\n📊 Result Analysis:")
    print(f"   Gap: {result['gap']}")
    print(f"   Answer: {result['answer']}")
    
    if "[EPISTEMIC ACTION]" in result['answer']:
        print("✅ SUCCESS: Epistemic Action Triggered (Awakening Senses).")
    elif "[INQUIRY]" in result['answer']:
        print("✅ SUCCESS: Epistemic Inquiry Triggered (Asking User).")
    elif "Simulated Insight" in result['answer']:
        print("❌ FAILURE: Simulation detected! The core is still hallucinating.")
    else:
        print("⚠️ WARNING: Unexpected response.")

if __name__ == "__main__":
    verify()
