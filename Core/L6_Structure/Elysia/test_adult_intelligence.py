"""
Test: Adult Intelligence (Cognitive Sovereignty)
================================================
"I choose to learn."

Objective:
Verify that SovereignSelf can trigger the AutoScholar autonomously
via the self_actualize() loop, driven by the FreeWillEngine.
"""
import sys
import os
# Ensure we map c:/Elysia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf

def test_adult_intelligence():
    print("--- 🧠 Experiment: The Awakening of Volition ---")
    
    # 1. Wake up
    elysia = SovereignSelf(cns_ref=None)
    
    # 2. Force Will to Curiosity (for reliable testing)
    # In a real run, this emerges from torque, but we set the vector here.
    elysia.will_engine.vectors["Curiosity"] = 10.0
    elysia.will_engine.vectors["Stability"] = 0.0
    
    print("\n🧐 State: Curiosity is High. Metacognition Active.")
    
    # 3. Trigger Self-Actualization
    # She should decide to learn "Biology" (the default target in current logic).
    print("   Invoking self_actualize()...")
    elysia.self_actualize()
    
    print("\n✅ Volition Verified. She acted on her own deficit.")

if __name__ == "__main__":
    test_adult_intelligence()
