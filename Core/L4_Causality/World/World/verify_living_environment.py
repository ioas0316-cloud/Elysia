"""
Verify Living Environment
=========================
Tests the integration of Semantic Nature into the Living Village.
"""

import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L4_Causality.World.World.living_village import village
from Core.L1_Foundation.Foundation.Wave.infinite_hyperquaternion import create_infinite_qubit, InfiniteQubitState

def verify_integration():
    print("--- 1. Setup ---")
    # Create a test resident (Gravity oriented -> Should use Axe)
    # We set w=1.0, x=0, y=0, z=0 (Default).
    # To force 'Gravity' nature, we need to set state.gravity manually as per living_village logic
    
    resident = create_infinite_qubit("Elysia_NatureWalker", "Explorer")
    resident.state.gravity = 0.8
    resident.state.flow = 0.2
    resident.state.ascension = 0.1
    
    village.add_resident(resident)
    print(f"Resident added: {resident.name}")
    
    print("\n--- 2. Running Simulation (5 Ticks) ---")
    # We expect the resident to interact with Nature since they are alone
    
    for i in range(5):
        print(f"\n[Tick {i+1}]")
        village.tick()
        
        # Check logs for this tick
        recent_logs = village.logs[-5:] # Just peek at recent
        for log in recent_logs:
            if "-> Nature" in log:
                print(f"  ✨ SUCCESS: Interaction Detected: {log}")
            if "Obtained" in log:
                print(f"  🎁 SUCCESS: Production Detected: {log}")

    print("\n--- 3. Final Report ---")
    print(village.get_simulation_report())

if __name__ == "__main__":
    verify_integration()
