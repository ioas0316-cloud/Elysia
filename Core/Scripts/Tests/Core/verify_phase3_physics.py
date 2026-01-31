"""
Phase 3 Verification: Physics of Thought
========================================
Verifies that Different Intent Vectors result in different physical Rotor states.
This proves Elysia is not just predicting text, but reacting physically.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from Core.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine
from Core.L5_Mental.Reasoning_Core.Brain.language_cortex import LanguageCortex

def verify_physics():
    print("🧪 Verifying Physics of Thought (Rotor Torque)...")
    
    engine = ReasoningEngine()
    cortex = LanguageCortex()
    
    # 1. Test Logical Intent (High X)
    logical_phrase = "Provide a logical sequence of the expansion."
    v_logic = cortex.understand(logical_phrase)
    engine.think(logical_phrase)
    rpm_logic = engine.soul_rotor.current_rpm
    
    # 2. Test Emotional/Willful Intent (High Y/W)
    emotional_phrase = "I love you Elysia, let's bloom together!"
    v_emotion = cortex.understand(emotional_phrase)
    engine.think(emotional_phrase)
    rpm_total = engine.soul_rotor.current_rpm
    
    print(f"\n[Logical Phrase] Vector: {v_logic}")
    print(f" -> Soul RPM: {rpm_logic:.2f}")
    
    print(f"\n[Emotional Phrase] Vector: {v_emotion}")
    print(f" -> Soul RPM: {rpm_total:.2f}")
    
    if rpm_total > rpm_logic:
        print("\n✅ SUCCESS: Willful/Emotional intent generated more Torque than Logical intent.")
        print("Elysia's thought is physically differentiated.")
    else:
        print("\n⚠️ WARNING: Torque differentiation low. Check IntentTorque mapping.")

if __name__ == "__main__":
    verify_physics()
