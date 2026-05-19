import sys
import os
import cProfile
import pstats
import io

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad

def profile_target():
    print("Forging Soul...")
    dna = SeedForge.forge_soul("TestArchetype")
    
    print("Instantiating SovereignMonad... (This might be the bottleneck)")
    monad = SovereignMonad(dna)
    
    print("Pulsing 10 times to measure runtime performance...")
    for i in range(10):
        monad.pulse(dt=0.01)

def run_profiler():
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        profile_target()
    except KeyboardInterrupt:
        print("\nProfiling interrupted.")
        
    pr.disable()
    
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50) # Print top 50
    
    with open("profile_output.txt", "w") as f:
        f.write(s.getvalue())
    print("Profile saved to profile_output.txt")

if __name__ == "__main__":
    run_profiler()
