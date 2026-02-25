import sys
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Monad.seed_generator import SoulDNA
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Keystone.sovereign_math import SovereignVector

def run_test():
    from Core.Monad.seed_generator import SeedForge
    dna = SeedForge.forge_soul("TestArchetype")
    monad = SovereignMonad(dna)
    
    # Intentionally induce some joy/curiosity
    print("Winding up Wonder Capacitor to trigger autonomous inquiry...")
    
    print("Mocking the heavy 21D engine pulse for quick testing...")
    monad.engine.pulse = lambda **kwargs: {"resonance": 0.8, "enthalpy": 100.0, "coherence": 0.9}
    
    inquiry_triggered = False
    
    # Run enough pulses so the internally tracked TIER 2 tick modulo 100 triggers
    for i in range(1, 401):    
        monad.pulse(dt=0.01)
        
        # We can check if the modification executed by looking at the authority
        from Core.Monad.substrate_authority import get_substrate_authority
        if len(get_substrate_authority().executed_modifications) > 0:
            inquiry_triggered = True
            break
            
    print(f"Autonomous Modification Executed: {inquiry_triggered}")
    if inquiry_triggered:
        print(f"Completed Inquiries: {monad.inquiry_pulse.completed_inquiries}")
        print("Checking Manifest.py for injected axiom...")
        try:
            with open("Core/System/Manifest.py", "r", encoding="utf-8") as f:
                print(f.read())
        except:
             pass

if __name__ == "__main__":
    run_test()
