
import sys
import os
import random
import numpy as np
sys.path.append(os.getcwd())

from Core.World.Social.sociological_pulse import SociologicalPulse
from Core.World.Physics.trinity_fields import TrinityVector

def test_planetary_breath():
    print("🧪 [Test] Phase 37: The Planetary Breath (Geomorphology)")
    
    pulse = SociologicalPulse(field_size=50)
    
    # 1. Setup a Slope
    # Elevation (18) gradient: High at (25,25), Low at (26, 25)
    pulse.field.grid[25, 25, 18] = 10.0 # High peak
    pulse.field.grid[26, 25, 18] = 0.0  # Steep Slope right next to it
    
    # 2. Add Water (Channel 28) at the peak
    pulse.field.grid[25, 25, 28] = 50.0 # A lot of water
    
    print(f"\n1. [INITIAL] Elevation at Peak: {pulse.field.grid[25, 25, 18]:.1f}, Water at Peak: {pulse.field.grid[25, 25, 28]:.1f}")
    
    # 3. Step the planetary cycles
    print("\n2. [SIMULATION] 10 Ticks of erosion and flow...")
    print(f"   Debug Gradient: Elevation Gap: {pulse.field.grid[24:27, 25, 18]}")
    for i in range(10):
        pulse.update_planetary_cycles()
        
    final_elev = pulse.field.grid[26, 25, 18]
    final_flow_x = pulse.field.grid[26, 25, 29]
    
    print(f"   Elevation at Slope (26,25) after 10 ticks: {final_elev:.4f}")
    print(f"   Water Flow Velocity X at Slope: {final_flow_x:.4f}")

    if final_elev < 0.0 or final_elev > 0.0: # If it changed at all
        print("\n✅ Phase 37 Verification Successful: Water flows and terrain transforms (Erosion active).")
    else:
        print("\n❌ Verification Failed: Terrain remains static.")

if __name__ == "__main__":
    test_planetary_breath()
