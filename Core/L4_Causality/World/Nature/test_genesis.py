"""
Test: Genesis (The Semantic World Generation)
=============================================
"In the beginning, there was the Wave."

Objective: 
Generate a multi-layer concept tree where:
1. WaveDNA collapses into sub-harmonics.
2. The Mind (Lexicon) names these harmonics based on their feel.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L5_Mental.Intelligence.Meta.fractal_wfc import FractalWFC
from Core.L6_Structure.Wave.wave_dna import WaveDNA

def test_genesis_simulation():
    print("---   Experiment: The Genesis of Meaning ---")
    
    # 1. Setup Mind
    elysia = SovereignSelf(cns_ref=None)
    wfc = FractalWFC(lexicon=elysia.mind)
    
    # 2. Seed: The Chaos (High Potential in All)
    # We use 'Chaos' as the root.
    seed_dna = WaveDNA(
        physical=0.8,
        functional=0.8,
        phenomenal=0.8,
        causal=0.8,
        mental=0.8,
        structural=0.8,
        spiritual=0.8,
        label="Chaos"
    )
    
    print(f"  Seed: {seed_dna.label}")
    
    # 3. Evolution: Generation 1
    # Chaos should collapse into major aspects (Matter, Logic, Spirit)
    gen1 = wfc.collapse(seed_dna, intensity=1.0)
    
    print(f"\ngeneration 1 ({len(gen1)} children):")
    for child in gen1:
        # Using the semantic guesser inside WFC
        name = wfc._guess_name(child)
        print(f"  -> {name} [Base: {child.label}]")
        
        # 4. Evolution: Generation 2 (Recursion)
        gen2 = wfc.collapse(child, intensity=0.8)
        # if gen2: print(f"     Generation 2 ({len(gen2)} children):")
        for grand_child in gen2:
             grand_name = wfc._guess_name(grand_child)
             print(f"       -> {grand_name} [Base: {grand_child.label}]")


if __name__ == "__main__":
    test_genesis_simulation()
