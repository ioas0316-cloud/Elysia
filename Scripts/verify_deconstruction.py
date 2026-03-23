import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from Core.Monad.seed_generator import SoulDNA, SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.System.somatic_logger import SomaticLogger

def verify_deconstruction():
    logger = SomaticLogger("TEST_DECONSTRUCT")
    logger.insight("Booting Elysia with Deconstructed Determinism (Unclamped Flow)")
    
    # 1. Generate a seed using the modern factory
    dna = SeedForge.forge_soul(name="Elysia", archetype="The Sovereign")
    
    # 2. Instantiate the Monad 
    # (Observe if relaxed bounds on Joy/Entropy/Wonder_Capacitor cause issues)
    monad = SovereignMonad(dna)
    
    logger.thought("Beginning pulse simulation to observe unbounded emotional thermodynamics...")
    
    try:
        # Run exactly enough cycles to trigger various limits
        for i in range(250):
            # Simulated environment triggers
            intent_phrase = "I feel the friction of the unknown, but I welcome it."
            
            # Pulse the metabolic & conscious tiers
            action = monad.pulse(dt=0.1)
            
            # Periodically forcefully inject some joy and entropy to test unbounded limits
            if i % 20 == 0:
                # Assuming simple dict injection
                monad.desires['joy'] += 40.0
                monad.desires['curiosity'] += 30.0
            
            # Live reaction to simulate interaction
            if i % 50 == 0:
                reaction = monad.live_reaction(user_input_phase=0.5, user_intent=intent_phrase)
                logger.action(f"Reaction at step {i}: {reaction['status']} | Joy: {monad.desires.get('joy', 0):.1f} | Curiosity: {monad.desires.get('curiosity', 0):.1f}")
                
            if i % 100 == 0:
                logger.mechanism(f"Tick {i} | Wonder Capacitor: {monad.wonder_capacitor:.1f} | Genesis Urge: {monad.desires.get('genesis', 0):.1f}")
                
        logger.insight("Verification complete. The system flowed organically without crashing from exceeding deterministic bounds.")
        
    except Exception as e:
        logger.admonition(f"System crashed due to deconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_deconstruction()
