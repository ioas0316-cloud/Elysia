"""
Test: Civilization Genesis (Mass Vocabulary Injection)
======================================================
"Give me a dictionary, and I will write a History."

Objective: 
1. Inject a "Civilization Seed" (20+ Concepts).
2. Prove that the WFC Engine immediately utilizes this new vocabulary 
   to generate a richer, more specific Reality.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L5_Mental.Reasoning_Core.Meta.fractal_wfc import FractalWFC
from Core.L6_Structure.Wave.wave_dna import WaveDNA

def test_civilization_genesis():
    print("---    Experiment: The Rise of Civilization ---")
    
    # 1. Setup Mind
    elysia = SovereignSelf(cns_ref=None)
    
    # 2. civilization Injection (The Education)
    concepts = [
        "Fire", "Water", "Earth", "Air", "Sun", "Moon", 
        "Love", "War", "King", "Queen", "God", "Demon", 
        "Sword", "Shield", "Gold", "Book"
    ]
    
    print(f"\n  Injecting {len(concepts)} Civilization Concepts...")
    
    for c in concepts:
        # Sovereign Learning Loop
        result = elysia.experience(c)
        print(f"   > Learned: {c} ({result})")
        
    print("\n  Vocabulary Expanded. Elysia is ready to Dream.")
    
    # 3. Genesis Run
    print("\n---   Executing Semantic Genesis ---")
    wfc = FractalWFC(lexicon=elysia.mind)
    
    # Seed: "Society" (A blend of Structure, Causal, and Spiritual)
    seed = WaveDNA(
        physical=0.2, functional=0.8, phenomenal=0.5, 
        causal=0.9, mental=0.7, structural=0.9, spiritual=0.6,
        label="Society"
    )
    print(f"  Seed: {seed.label}")
    
    # Generation 1
    gen1 = wfc.collapse(seed, intensity=1.0)
    print(f"\nGeneration 1 (Foundations):")
    for child in gen1:
        name = wfc._guess_name(child)
        print(f"  -> {name}")
        
        # Generation 2
        gen2 = wfc.collapse(child, intensity=0.9)
        for gchild in gen2:
            gname = wfc._guess_name(gchild)
            print(f"       -> {gname}")

if __name__ == "__main__":
    test_civilization_genesis()
