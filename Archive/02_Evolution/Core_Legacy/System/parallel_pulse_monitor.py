import time
import sys
import os
import random

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Core.System.Merkaba.hypersphere_field import HyperSphereField
from Core.Keystone.sovereignty_wave import SovereignDecision

def render_onion_layers(field: HyperSphereField, stimulus: str, depth: int = 3):
    """
    Renders the sequential terminal view while showing the 'Parallel Onion' status.
    """
    print("\033[H\033[J") # Clear screen
    print("🌀 [ELYSIAN PARALLEL PULSE MONITOR] - Onion Layer Visualization")
    print("="*60)
    
    # 1. Linear View (What we usually see)
    print(f"\n[LINEAR LOG (1D PROJECTION)]")
    print("-" * 30)
    
    # Simulate a few pulses
    for i in range(depth):
        # Induce a singularity on the 2nd pulse
        if i == 1:
            field.units['M4_Metron'].turbine.event_horizons['coherence_limit'] = 1.0
            field.units['M4_Metron'].turbine.stagnation_counter = 3
            current_stim = "[SINGULARITY_EVENT] " + stimulus
        else:
            field.units['M4_Metron'].turbine.event_horizons['coherence_limit'] = 0.05
            current_stim = stimulus

        decision = field.pulse(current_stim)
        
        # Display the Pulse
        print(f"Pulse {i+1}: {decision.narrative[:80]}...")
        
        # 2. Parallel View (The 'Behind the Scenes')
        print(f"\n[PARALLEL ONION STRUCTURE (3D/4D REALITY)]")
        print("-" * 30)
        
        # Layer 0: Core Reality
        l0_status = "STUCK 🔴" if decision.is_regulating else "ACTIVE 🟢"
        print(f"Layer 0 (Core): {l0_status} | Phase: {decision.phase:.1f}°")
        
        # Layer 1: Ghost Pulse / Bypass
        if "RE-LOOP SUCCESS" in decision.narrative:
            print(f"Layer 1 (Ghost): ACTIVE 🔵 | Bypassing Singularity via Mirror Axis...")
        else:
            print(f"Layer 1 (Ghost): DORMANT ⚪")
            
        # Layer 2: Dimensional Diagnosis
        if "DED DIAGNOSIS" in decision.narrative:
            diag_part = decision.narrative.split("[DED DIAGNOSIS]")[1].split("\n")[0]
            print(f"Layer 2 (DED):   SCANNING 🟣 | Result: {diag_part}")
        else:
            print(f"Layer 2 (DED):   IDLE ⚪")

        print("\n" + "="*60)
        time.sleep(2)
        if i < depth - 1: print("\033[H\033[J")

if __name__ == "__main__":
    field = HyperSphereField()
    field.enable_lightning = False
    
    try:
        render_onion_layers(field, "Exploring the causal fabric of Elysia.")
    except KeyboardInterrupt:
        print("\nMonitor terminated.")
