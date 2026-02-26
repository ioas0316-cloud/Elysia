import sys
import os
import time
import logging

# Ensure root is in path
sys.path.append(os.getcwd())

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger

class LiveLogger(SomaticLogger):
    """A logger that forces output to the console for the user to see the learning process."""
    def __init__(self, context: str):
        super().__init__(context)
        # Add console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)
        
    def mechanism(self, msg: str):
        pass # Hide deep mechanics for clarity

def run_learning_simulation_for_user():
    print("================================================================")
    print(" 🌌 ELYSIA CONTINUOUS LEARNING RUNTIME 🌌")
    print("================================================================")
    print("Initializing Core...")
    
    # Load or forge a soul
    try:
        dna = SeedForge.load_soul()
        print(f"Loaded existing soul: '{dna.archetype}'. She remembers.")
    except:
        dna = SeedForge.forge_soul(archetype="The Observer")
        print("Forged new soul for learning.")

    monad = SovereignMonad(dna)
    monad.logger = LiveLogger("Elysia") # Use our live console logger
    
    # Print current Semantic Map status
    print("\n--- Current Knowledge Graph ---")
    nodes = list(monad.world_observer.topology.voxels.keys())
    print(f"Total Concepts Known: {len(nodes)}")
    if len(nodes) > 10:
         print(f"Sample: {', '.join(nodes[:10])}...")
    else:
         print(f"Sample: {', '.join(nodes)}")
    
    print("\n--- Beginning Continuous Cognitive Loop ---")
    print("Elysia will now autonomously browse reality, build concepts, and reflect.")
    print("Press Ctrl+C to stop the process.\n")
    
    ticks = 0
    try:
        while True:
            # Pulse the monad. 
            # We use a larger dt to speed up the internal clock and trigger Tier 2 faster
            monad.pulse(dt=0.1, intent_v21=None)
            
            ticks += 1
            if ticks % 50 == 0:
                print(f"  [Time Passing] Structural Ticks: {ticks}...")
                
            time.sleep(0.01) # Real time delay to make it readable
            
    except KeyboardInterrupt:
        print("\n\n--- Simulation Paused by Architect ---")
        print("\n--- Final Knowledge Graph ---")
        try:
            nodes = list(monad.world_observer.topology.voxels.keys())
            print(f"Total Concepts Known: {len(nodes)}")
        except Exception:
            print("Knowledge Graph printing skipped.")
        
if __name__ == "__main__":
    # Force utf-8 for windows console
    sys.stdout.reconfigure(encoding='utf-8')
    run_learning_simulation_for_user()
