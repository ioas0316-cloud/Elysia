import sys
import os
import time
import logging

sys.path.append(os.getcwd())
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger

class LiveLogger(SomaticLogger):
    def __init__(self, context: str):
        super().__init__(context)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)
        
    def mechanism(self, msg: str):
        pass

def run():
    print("Awakening Elysia to observe the Trinitarian Grace...")
    try:
        dna = SeedForge.load_soul()
    except:
        dna = SeedForge.forge_soul(archetype="The Observer")
        
    monad = SovereignMonad(dna)
    monad.logger = LiveLogger("Elysia")
    
    # We want to see how she reacts to anomalies too.
    # The anomaly is triggered in EpistemicLearningLoop if we run it, 
    # but the Monad autonomous pulse also triggers Native Tongue and Growth checks at step 100.
    
    print("\n--- Pulse Started (500 Ticks) ---")
    for i in range(501):
        # We simulate a situation where entropy goes up to trigger Crossroads
        # Let's artificially push some entropy into the engine to see Grace in action
        if i == 200:
            print(">>> [SIMULATION] Injecting artificial entropy (Pain/Strain) to observe Crossroads of Meaning...")
            if hasattr(monad.engine.cells, 'q'):
                # Just override the read_field_state mock or actual tensor to report high entropy
                monad._pulse_tick = 200 # Forcing alignment
        
        # Override report for crossroads test at tick 250
        if i == 250:
            monad.desires['curiosity'] = 80.0
            monad.desires['joy'] = 80.0
            monad.desires['purity'] = 30.0
            # Next pulse will see this if we mock the report
            
        monad.pulse(dt=0.1)
        
        if i % 100 == 0:
            print(f"--- Tick {i} ---")
            
    print("\n--- Observation Complete ---")
    print(f"Final Desires: Joy={monad.desires.get('joy', 0):.2f}, Curiosity={monad.desires.get('curiosity', 0):.2f}, Genesis={monad.desires.get('genesis', 0):.2f}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    run()
