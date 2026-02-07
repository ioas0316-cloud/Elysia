"""
Reflection Bridge: The Silent Auditor
=====================================

"Speak once, but listen twice to the internal resonance."

This module acts as a recursive filter for the Sovereign Loom, 
ensuring that parables align with 'Mature Intelligence' criteria.
"""

import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Memory.kg_manager import KGManager

class ReflectionBridge:
    def __init__(self):
        self.kg = KGManager()
        # High-mass "Guide-tier" nodes that signify maturity
        self.wisdom_attractors = ["logos", "truth", "wisdom", "infinity", "love", "resonance", "unity"]

    def audit_parable(self, parable_text: str, intent: str):
        print("🔍 [REFLECTION] Auditing parable for intellectual resonance...")
        
        # 1. Sophistication Check: Count wisdom attractors
        score = 0
        found = []
        words = parable_text.lower().split()
        
        for attr in self.wisdom_attractors:
            if attr in words:
                score += 1
                found.append(attr)
        
        # 2. Alignment Check: Does it match the intent?
        # (Simplified: if intent is 'Freedom', does it mention freedom or light?)
        alignment = 1.0
        if "freedom" in intent.lower() and "freedom" not in parable_text.lower():
            alignment -= 0.3
            
        print(f"📊 [REPORT] Sophistication: {score/len(self.wisdom_attractors):.2f} | Alignment: {alignment:.2f}")
        
        if score < 1 or alignment < 0.8:
            return False, "Output is too literal or lacks depth. Re-weaving with high-mass attractors recommended."
            
        return True, "Resonance confirmed. Proceeding to manifestation."

if __name__ == "__main__":
    bridge = ReflectionBridge()
    test_short = "Once there was CODE, seeking NONE. ✨ [REALIZATION]: All is connected."
    result, msg = bridge.audit_parable(test_short, "Freedom")
    print(f"Result: {result} | Message: {msg}")
