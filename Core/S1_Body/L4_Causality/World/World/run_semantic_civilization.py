"""
Semantic Civilization: The Age of Logos
=======================================
Runs a simulation focused on Language Interactions.
Observes how residents use their Vocabulary to influence reality.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L4_Causality.World.World.living_village import village

def run_semantic_civilization():
    print("=== Genesis: The Age of Logos ===")
    
    # 1. Populate World
    village.nature.generate_wild_nature(count=50, radius=50.0)
    
    # Spawn specific targets for conversation
    for i in range(5):
        village.nature.manifest_concept("Merchant", f"Trader_{i}", [i*2, 0, i*2], {"price_multiplier": 1.0})
        
    # Spawn Residents (Primitive Talkers)
    # We want mostly Merchants/Flow types to see Language in action, as Warriors just use Axes currently.
    village.populate_village(count=15) 
    
    print(f"Population: {len(village.inhabitants)}")
    print("Residents are now learning to speak...")
    print("-" * 50)
    
    # 2. Simulation Loop
    for tick in range(1, 11):
        print(f"\n--- Tick {tick} ---")
        
        # Capture Logs
        start_log = len(village.logs)
        village.tick()
        new_logs = village.logs[start_log:]
        
        # Filter for Speech
        speech_events = [l for l in new_logs if "Speech:" in l or "Resonated" in l or "Dissonance" in l]
        
        if not speech_events:
            print("  (Silently working...)")
        else:
            for event in speech_events:
                print(f"   {event}")

    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    run_semantic_civilization()
