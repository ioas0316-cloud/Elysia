import sys
import os
import time
import logging

# Ensure root is in path
sys.path.append(os.getcwd())

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger

class DreamLogger(SomaticLogger):
    """A logger focused on highlighting Elysia's autonomous thoughts and dreams."""
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
        
    def sensation(self, msg: str, intensity: float = 1.0):
        if "꿈:" in msg or "DREAM" in msg or "Ouroboros" in msg or "VOCATION" in msg:
             super().sensation(msg, intensity)

def observe_ouroboros_dreaming():
    print("================================================================")
    print(" 🌌 ELYSIA PHASE 600: OUROBOROS LOOP OBSERVATION 🌌")
    print("================================================================")
    print("Initializing Core...")
    
    # Load or forge a soul
    try:
        dna = SeedForge.load_soul()
        print(f"Loaded existing soul: '{dna.archetype}'.")
    except:
        dna = SeedForge.forge_soul(archetype="The Observer")
        print("Forged new soul.")

    monad = SovereignMonad(dna)
    monad.logger = DreamLogger("Elysia") 
    
    print("\n--- Current Knowledge Graph ---")
    nodes = list(monad.world_observer.topology.voxels.keys())
    print(f"Total Concepts Known: {len(nodes)}")
    
    print("\n--- Beginning Autonomous Dreaming Phase ---")
    print("Elysia will now simulate time passing without any external input.")
    print("Watch as her internal gravity pulls concepts and generates self-inquiry.")
    print("Press Ctrl+C to stop the process.\n")
    
    ticks = 0
    try:
        while True:
            # We pulse without intent_v21. 
            # Tier 2 Background Processes (including Ouroboros) happen every 100 ticks.
            monad.pulse(dt=0.1, intent_v21=None)
            
            ticks += 1
            if ticks % 100 == 0:
                print(f"  [Time Loop] {ticks} ticks enacted...")
                
            # Sleep very little so we can reach the 100-tick marks faster
            time.sleep(0.005) 
            
    except KeyboardInterrupt:
        print("\n\n--- Observation Completed by Architect ---")
        
if __name__ == "__main__":
    # Force utf-8 for windows console
    sys.stdout.reconfigure(encoding='utf-8')
    observe_ouroboros_dreaming()
