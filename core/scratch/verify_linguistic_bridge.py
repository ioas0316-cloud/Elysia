import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from core.sensory_lens_manifold import SensoryLensManifold

def test_bridge():
    manifold = SensoryLensManifold()
    
    sentences = [
        "HELLO", "ELYSIA",
        "THIS", "IS", "A", "NEW", "PHASE",
        "THE", "UNIVERSE", "IS", "GEOMETRY",
        "print('Hello World')",
        "def autopoiesis(): return True"
    ]
    
    print("=== Absorbing Vocabulary & Injecting Tension ===")
    for s in sentences:
        print(f"Injecting: {s}")
        # Inject tension (high tension to force Thoughts)
        # Note: inject_stimulus expects raw bytes
        manifold.inject_stimulus(s.encode('utf-8'), 5.0)
    
    print(f"Vocabulary size: {len(manifold.linguistic_bridge.vocabulary)}")
    
    print("\n=== Simulating Pulse for Maturation ===")
    epiphany_reached = False
    pulses = 0
    
    # We will loop until Epiphany
    while not epiphany_reached and pulses < 600:
        pulses += 1
        manifold.metabolize_consciousness(0.05)
        # Decay internal thoughts as well (normally done via metabolize_consciousness calling manifold_root's metabolize_apoptosis)
        has_epiphany = manifold.ponder()
        
        # Log progress every 20 pulses
        if pulses % 20 == 0:
            thoughts = len(manifold.manifold_root.internal_thoughts)
            branches = len(manifold.manifold_root.children)
            print(f"Pulse {pulses:3d} | Thoughts: {thoughts:2d} | Branches: {branches:2d} | Global Tension: {manifold.master.global_tension:.4f} rad")
            
        if has_epiphany:
            epiphany_reached = True
            spoken_word = manifold.project_epiphany()
            print(f"\n[!!! LINGUISTIC EMERGENCE at Pulse {pulses} !!!]")
            print(f"Elysia Speaks: '{spoken_word}'\n")

if __name__ == "__main__":
    test_bridge()
