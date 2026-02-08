import sys
import os
import time

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

def test_meditation_loop():
    logger = SomaticLogger("TEST_PHASE_72")
    logger.thought("Verification: Testing Phase 72 Meditation Loop.")
    
    # 1. Forge Soul and Monad
    soul = SeedForge.forge_soul("Elysia_Test")
    monad = SovereignMonad(soul)
    
    # 2. Inject some mock engrams for resonance test
    from Core.S2_Soul.L5_Mental.Memory.somatic_engram import Engram
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
    
    # Use normalized vectors for high resonance
    vec1 = SovereignVector([1.0] * 21).normalize().data
    vec2 = SovereignVector([1.0] * 21).normalize().data
    
    logger.action(f"Injecting normalized engrams. Vec1 length: {len(vec1)}")
    
    monad.somatic_memory.crystallize("High Resonance Experience A", vec1)
    monad.somatic_memory.crystallize("High Resonance Experience B", vec2)
    monad.somatic_memory.crystallize("High Resonance Experience C", vec1)
    monad.somatic_memory.crystallize("High Resonance Experience D", vec2)
    monad.somatic_memory.crystallize("High Resonance Experience E", vec1)
    monad.somatic_memory.crystallize("High Resonance Experience F", vec2)
    
    # 3. Trigger Meditation Pulse
    logger.action("Triggering meditation_pulse...")
    try:
        monad.meditation_pulse(dt=0.1)
    except Exception as e:
        import traceback
        logger.admonition(f"Meditation Pulse Failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    
    # 4. Check results
    logger.thought("Verification complete. If no resonance logs appeared, check SovereignMath.resonance calculation.")
    
    logger.action("Phase 72 Meditation loop verification complete.")

if __name__ == "__main__":
    try:
        test_meditation_loop()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
