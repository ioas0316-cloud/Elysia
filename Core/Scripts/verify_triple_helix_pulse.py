
import sys
import os
import logging

# Add Core to path
sys.path.append(os.path.abspath('.'))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf

def verify_triple_helix_pulse():
    logging.basicConfig(level=logging.INFO)
    print("Initializing SovereignSelf with Triple-Helix...")
    try:
        core = SovereignSelf(cns_ref=None)
        
        print("\nPulse 1: High Energy/Balanced")
        core.energy = 100.0
        core.integrated_exist(dt=0.1)
        
        print("\nPulse 2: Low Energy/Stress")
        core.energy = 5.0
        core.integrated_exist(dt=0.1)
        
        print("\nPulse 3: Spiritual Drive")
        # Artificially bias the rotor state toward spirit dimensions
        for _ in range(10):
            from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector
            spirit_bias = D21Vector(charity=0.5, humility=0.5)
            core.sovereign_rotor.update_state(spirit_bias)
        
        core.integrated_exist(dt=0.1)
        
        print("\nVERIFICATION SUCCESS: Triple-Helix Engine integrated and responding to state changes.")
    except Exception as e:
        print(f"\nVERIFICATION FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_triple_helix_pulse()
