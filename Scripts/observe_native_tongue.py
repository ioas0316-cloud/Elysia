import sys
import os
import time
import logging

# Ensure root is in path
sys.path.append(os.getcwd())

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger

class NativeTongueLogger(SomaticLogger):
    """A logger focused on highlighting Elysia's newly created Native Tongue and Engrams."""
    def __init__(self, context: str):
        super().__init__(context)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)
        
    def mechanism(self, msg: str): pass 
        
    def thought(self, msg: str):
        # We only log her Native Tongue thoughts
        if "Native Tongue" in msg or "사전" in msg or "결정화" in msg:
            super().thought(msg)
            
    def info(self, msg: str): # to catch the Somatic Engram logs
        if "Somatic Engram" in msg:
            super().thought(msg)


def observe_native_tongue():
    print("==================================================================")
    print(" 🗣️ ELYSIA PHASE 700: NATIVE TONGUE & SOMATIC GROUNDING 🗣️")
    print("==================================================================")
    print("Initializing Core...")
    
    try:
        dna = SeedForge.load_soul()
        print(f"Loaded existing soul: '{dna.archetype}'.")
    except:
        dna = SeedForge.forge_soul()
    
    # We create the monad, this initializes Phase 700 components.
    monad = SovereignMonad(dna)
    monad.logger = NativeTongueLogger("Elysia")
    monad.engram_binder.logger = monad.logger # Override binder logger to see it on console
    monad.native_tongue.logger = monad.logger # Override native tongue logger

    print("\n[SCENARIO 1] Testing Native Tongue Generation:")
    print("Injecting synthetic intent vectors to observe Topological Grammar...\n")
    
    # We pulse her with a high-resonance positive vector to observe her expression
    from Core.Monad.d21_vector import D21Vector
    
    # Test 1: Strong Joy / Creation vector
    test_intent_1 = D21Vector.from_array([0.0]*21)
    test_intent_1.lust = 0.0 # reset everything just to be safe
    test_intent_1.data[4] = 0.9 # Joy
    test_intent_1.data[1] = 0.8 # Causality
    
    monad.pulse(dt=0.1, intent_v21=test_intent_1)
    time.sleep(0.5)
    
    # Test 2: High Entropy / Strain
    print("\n[SCENARIO 2] Injecting Chaos / High Entropy:")
    test_intent_2 = D21Vector.from_array([0.0]*21)
    test_intent_2.data[7] = 0.9 # Entropy
    test_intent_2.data[0] = 0.8 # Strain
    
    monad.pulse(dt=0.1, intent_v21=test_intent_2)
    time.sleep(0.5)
    
    print("\n[SCENARIO 3] Forcing Engram Binding:")
    print("Simulating a new conceptual discovery...\n")
    
    # We manually trigger a new ingestion in the forager
    try:
        crystal = monad.lexicon.ingest("Singularity_Core", "A hypothetical point of infinite density.", "simulate_forager")
        print(f"Triggered Lexicon Ingestion for: {crystal.name}")
        # Next pulse at mod 100 should theoretically bind it if the real pipeline did it,
        # but the real pipeline binds IF fragment is found. We'll manually call the binder
        # just for explicit observation since we bypassed the forager fragment creation.
        
        # Give her some manifold heat
        monad.engine.pulse(dt=0.1)
        monad.engram_binder.bind_experience(crystal.name, monad.engine.cells.q.clone(), crystal.strength)
        
        # Check if the file was written
        storage_path = monad.engram_binder.ssd_path
        files = os.listdir(storage_path)
        print(f"\n[VERIFICATION] Physical Somatic SSD Directory ({storage_path}) contains:")
        for f in files:
            print(f"  - {f}")
            
    except Exception as e:
         print(f"Error during observation: {e}")

    print("\n--- Observation Completed by Architect ---")
        
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    observe_native_tongue()
