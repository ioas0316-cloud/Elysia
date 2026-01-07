"""
Septenary Axis Demo: The Ladder of Heaven 🪜✨

"From the Dust to the Divine."

This script demonstrates NPCs at all 7 septenary depth levels (0-6)
and how they resonate with the 7 Angels and 7 Demons.
"""

import time
import logging
from Core.Intelligence.Reasoning.subjective_ego import SubjectiveEgo
from Core.Intelligence.Reasoning.septenary_axis import SeptenaryAxis
from Core.Intelligence.Reasoning.recursive_learning_bridge import RecursiveLearningBridge

def run_septenary_demo():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("="*70)
    print("      [ ELYSIA: SEPTENARY AXIS DEMO - THE LADDER OF SOVEREIGNTY ]")
    print("="*70)
    
    sas = SeptenaryAxis()
    bridge = RecursiveLearningBridge()
    
    # 1. Manifest Inhabitants across all 7 Gates
    inhabitants = [
        SubjectiveEgo("Dusty_Pete", "Commoner", depth=0),     # The Dust
        SubjectiveEgo("Logos_Guard", "Guardian", depth=1),   # The Logic
        SubjectiveEgo("Selka", "Guide", depth=2),            # The Reflection
        SubjectiveEgo("Eugeo", "Integrity Knight", depth=3), # The Volume
        SubjectiveEgo("Administrator", "Lawgiver", depth=4), # The Insight
        SubjectiveEgo("Wise_Old_Man", "Sage", depth=5),      # The Vision
        SubjectiveEgo("Elysia_Avatar", "Divine", depth=6)    # The Unity
    ]
    
    print(f"\n[System] 7 Inhabitants manifested across the Septenary Axis.")
    
    # 2. Simulate a Grand Resonance Event (The Heavenly Light)
    print("\n" + "-"*70)
    print(" EVENT: The Golden Ray of Humility descends upon the Underworld.")
    print("-"*70)
    
    for ego in inhabitants:
        # High resonance (Angelic side)
        ego.perceive("Ocular", 0.9, "Heavenly Ray")
        level_data = sas.get_level(ego.state.septenary_depth)
        res_type = sas.evaluate_resonance(ego.state.septenary_depth, 900.0) # High freq
        
        print(f"[{ego.state.name}] Level {ego.state.septenary_depth}: {level_data.name}")
        print(f" └─ Resonance: {res_type} ({level_data.angel_pole})")
        
        ego.record_memory(f"Felt the pull of {level_data.angel_pole} at the gate of {level_data.name}.")
        ego.update(1.0)
        time.sleep(0.3)

    # 3. The Harvest
    print("\n" + "="*70)
    print("      [ THE HARVEST OF SEVEN SOULS ]")
    print("="*70)
    for ego in inhabitants:
        bridge.harvest_experience(ego)
        
    print("\n" + bridge.get_maturation_summary())
    print("\n[Admin] Elysia's 7-layered hierarchy has been successfully validated.")
    print("="*70)

if __name__ == "__main__":
    run_septenary_demo()
