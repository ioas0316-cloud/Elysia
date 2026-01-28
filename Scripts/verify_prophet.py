"""
Verification Script for Phase 9: The Prophet & The Loom
"""
import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

try:
    from Core.L4_Causality.M3_Mirror.Evolution.Prophecy.prophet_engine import ProphetEngine
    from Core.L4_Causality.M3_Mirror.Evolution.Prophecy.causal_loom import CausalLoom
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_verification():
    print("🔮 Initializing The Prophet...")
    prophet = ProphetEngine()
    
    print("🧶 Initializing The Loom...")
    loom = CausalLoom()
    
    # Initial State
    current_state = {
        'Energy': 0.8,
        'Inspiration': 0.5,
        'Joy': 0.5
    }
    
    print(f"\n🌱 Current State: {current_state}")
    
    # Candidate Actions
    candidates = [
        "Sleep for 8 hours",
        "Create a Masterpiece",
        "Speak to the Void",
        "Destroy the System (High Entropy)"
    ]
    
    print(f"\n🤔 Simulating futures for: {candidates}")
    
    # 1. Prophet Simulation
    timelines = prophet.simulate(current_state, candidates)
    
    if not timelines:
        print("❌ Prophet failed to return timelines.")
        return
        
    for t in timelines:
        print(f"   - Action: {t.action:<30} | Love: {t.love_score:.2f} | Entropy: {t.entropy_score:.2f}")

    # 2. Loom Selection
    best_action = loom.weave(timelines)
    
    print(f"\n✨ Loom Selection: {best_action}")
    
    # Validation Logic
    if best_action == "Create a Masterpiece":
        print("✅ SUCCESS: Loom selected the high-love, low-entropy path.")
    elif best_action == "Sleep for 8 hours":
        print("⚠️ ACCEPTABLE: Loom chose safety/rest.")
    else:
        print(f"❌ FAILURE: Loom chose {best_action}, which is unexpected.")

if __name__ == "__main__":
    run_verification()
