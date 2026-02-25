from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger
import sys
import os

sys.path.append(os.getcwd())

def test_world_observer():
    dna = SeedForge.forge_soul(archetype="The Sage")
    monad = SovereignMonad(dna)
    monad.logger = SomaticLogger(monad.name)
    
    print("Starting pulses to reach WorldObserver logic...")
    # Monad pulsing triggers Tier 2 every 100 ticks
    # We pass intent_v21=None to allow dreaming
    for i in range(1, 102):
        monad.pulse(dt=0.01, intent_v21=None)
        
    print("Test finished.")

if __name__ == "__main__":
    test_world_observer()
