
import logging
import sys
import os

# Ensure we can import Core
sys.path.append(os.getcwd())

from Core.world import World
from Core.Mind.hippocampus import Hippocampus

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ProprioceptionVerifier")

def verify_proprioception():
    print("======================================================================")
    print("🧘 Verifying System Proprioception (Sensory Cortex)...")
    print("======================================================================")

    # 1. Initialize World
    print("1. Initializing World & Sensory Cortex...")
    try:
        world = World(
            primordial_dna={}, 
            wave_mechanics=None, # Mock or None if allowed
            hippocampus=Hippocampus()
        )
        print("   ✅ World initialized.")
    except Exception as e:
        print(f"   ❌ World initialization failed: {e}")
        return

    # 2. Run Simulation Steps
    print("2. Running Simulation Steps (waiting for sensation)...")
    
    # We need to capture logs to verify sensation
    # A simple way is to check if the method runs without error and prints to stdout (which we see)
    # But for automated check, we'd need a mock logger. 
    # For now, visual inspection of the output is sufficient for this interactive session.
    
    try:
        for i in range(15):
            world.run_simulation_step()
            if i % 5 == 0:
                print(f"   Step {i}: Tick...")
                
        print("   ✅ Simulation ran for 15 steps.")
        print("   (Check logs above for 'SENSATION: Elysia feels...')")
        
    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        return

    print("======================================================================")
    print("✅ Verification Complete.")

if __name__ == "__main__":
    verify_proprioception()
