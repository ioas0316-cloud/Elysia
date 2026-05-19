"""
Test: The Synapse (Graph to Rotor)
==================================
"Memory becomes Thought."

Objective:
Verify that HyperSphereCore can summon concepts from TrinityLexicon
and spin them as Rotors with correct physical properties.
"""
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.Cognition.trinity_lexicon import TrinityLexicon
from Core.System.hyper_sphere_core import HyperSphereCore

def test_synapse():
    print("---   Experiment: The Synapse (Unified Mind) ---")
    
    # 1. Initialize Memory
    print("  Initializing Memory (Lexicon)...")
    lexicon = TrinityLexicon()
    # Ensure graph is ready (sync primitives)
    if lexicon.graph:
        lexicon._sync_primitives()
    
    # 2. Initialize Core with Synapse
    print("  Initializing Core (HyperSphere)...")
    core = HyperSphereCore(lexicon=lexicon)
    
    # 3. Summon 'Earth' (High Gravity, Low Flow)
    # Expected: High Mass, Low RPM
    print("\n  Summoning 'earth'...")
    core.summon_thought("earth")
    
    rotor_earth = core.rotors.get("earth")
    if rotor_earth:
        print(f"   Spec: {rotor_earth.config}")
        if rotor_earth.config.mass > 5.0 and rotor_earth.config.rpm < 200:
             print("     CORRECT: Earth is Heavy and Slow.")
        else:
             print("     FAILURE: Earth physics mismatch.")
    else:
        print("     FAILURE: Rotor not created.")

    # 4. Summon 'Wind' (Low Gravity, High Flow)
    # Expected: Low Mass, High RPM
    print("\n  Summoning 'wind'...")
    core.summon_thought("wind")
    
    rotor_wind = core.rotors.get("wind")
    if rotor_wind:
        print(f"   Spec: {rotor_wind.config}")
        if rotor_wind.config.mass < 2.0 and rotor_wind.config.rpm > 300:
             print("     CORRECT: Wind is Light and Fast.")
        else:
             print("     FAILURE: Wind physics mismatch.")
             
    # 5. Summon 'Unknown'
    print("\n  Summoning 'void_concept'...")
    core.summon_thought("void_concept")
    if "void_concept" not in core.rotors:
        print("     CORRECT: Unknown concept cannot be thought (Hollow).")

if __name__ == "__main__":
    test_synapse()
