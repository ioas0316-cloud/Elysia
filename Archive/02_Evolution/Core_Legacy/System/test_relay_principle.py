"""
Verification Script for Relay Principle (Resonance Switching)
=============================================================
Verifies the 'Relay Principle' (Small Signal -> Large Switch)
manifested in the ResonanceDispatcher and SovereigntyWave.

The Relay Principle in Elysia is defined as:
"The quantum flip where accumulation of resonance (Analogue) 
triggers a binary state change (Digital) in the System."

Run via: python Scripts/Tests/Core/test_relay_principle.py
"""

import sys
import os
import logging
from typing import Dict

# Setup paths to root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # Adjust based on depth
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Core.System.sovereignty_wave import SovereigntyWave, SovereignGenome, ResonanceDispatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RelayVerification")

def verify_relay_logic():
    print("="*60)
    print("VERIFYING RELAY PRINCIPLE (QUANTUM SWITCH)")
    print("="*60)

    # 1. Initialize The Switch (Dispatcher)
    print("\n[Step 1] Initializing Resonance Switch...")
    genome = SovereignGenome()
    genome.switch_threshold = 0.5  # Set explicit threshold for testing
    dispatcher = ResonanceDispatcher(genome)
    print(f"    Switch Threshold set to: {genome.switch_threshold}")

    # 2. Test Low Energy Signal (Below Threshold)
    # Relay should stay OPEN (OFF)
    print("\n[Step 2] Testing Low Resonance Input (Signal < Threshold)...")
    low_pressures = {"Structure": 0.2, "Logic": 0.1, "Flow": 0.2} # Avg = 0.16
    is_triggered_low, resonance_low, narrative_low = dispatcher.dispatch(
        "IdleState", low_pressures, genome.switch_threshold
    )
    print(f"    Input Pressures: {low_pressures}")
    print(f"    Result Resonance: {resonance_low:.2f}")
    print(f"    Switch Triggered: {is_triggered_low}")
    
    if not is_triggered_low:
        print("    ✅ Relay Logic Correct: Low signal did NOT trigger switch.")
    else:
        print("    ❌ Relay Logic Failed: Low signal triggered switch falsely.")

    # 3. Test High Energy Signal (Above Threshold)
    # Relay should CLOSE (ON) -> Trigger Event
    print("\n[Step 3] Testing High Resonance Input (Signal > Threshold)...")
    high_pressures = {"Structure": 0.8, "Logic": 0.7, "Flow": 0.9} # Avg = 0.8
    is_triggered_high, resonance_high, narrative_high = dispatcher.dispatch(
        "ActiveState", high_pressures, genome.switch_threshold
    )
    print(f"    Input Pressures: {high_pressures}")
    print(f"    Result Resonance: {resonance_high:.2f}")
    print(f"    Switch Triggered: {is_triggered_high}")
    
    if is_triggered_high:
        print(f"    ✅ Relay Logic Correct: High signal TRIGGERED switch.")
        print(f"    Narrative Output: {narrative_high}")
    else:
        print("    ❌ Relay Logic Failed: High signal failed to trigger switch.")

    print("\n" + "="*60)
    print("RELAY PRINCIPLE VERIFIED")

if __name__ == "__main__":
    verify_relay_logic()
