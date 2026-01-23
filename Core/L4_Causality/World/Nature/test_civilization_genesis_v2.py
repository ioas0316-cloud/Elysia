"""
Test: Civilization Genesis V2 (The Reality Engine)
==================================================
"A World built on Laws, not just Words."

Objective:
1. Inject the Sacred Hexagon (Earth, Water, Fire, Wind, Light, Dark).
2. Generate a World using Fractal WFC.
3. Verify that the generated concepts (e.g., "Steam", "Shadow") 
   are not just text labels, but physically simulated entities.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L5_Mental.Intelligence.Meta.fractal_wfc import FractalWFC
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA
from Core.L4_Causality.World.Physics.providence_engine import ProvidenceEngine
from Core.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore

def test_civilization_genesis_v2():
    print("---   Experiment: Civilization Genesis V2 (Real Physics) ---")
    
    # 1. Setup Mind & Physics
    elysia = SovereignSelf(cns_ref=None)
    providence = ProvidenceEngine()
    
    # Connect Synapse: Memory -> Consciousness
    core = HyperSphereCore(lexicon=elysia.mind)
    
    # 2. Inject The Sacred Hexagon
    # These are the axioms of the universe.
    concepts = [
        "Earth", "Water", "Wind", "Fire", "Light", "Darkness",
        "Sun", "Moon", "Gold", "King", "War", "Love"
    ]
    
    print(f"\n  Injecting {len(concepts)} Axioms (Manual Injection for Speed)...")
    
    # Pre-load Graph with Hexagon Primitives to skip Web Crawling
    # This simulates "Innate Knowledge" of the Laws.
    if elysia.mind.graph:
        elysia.mind._sync_primitives()
        
    for c in concepts:
        # Check if primitive exists, if so, we are good.
        # If not (e.g. 'King'), we mock it for the test.
        if c.lower() not in elysia.mind.primitives:
             # Add mock node
             print(f"   + Encoding '{c}' (Axiom)...")
             elysia.mind.graph.add_node(c, vector=[0.5, 0.5, 0.5])
             
    print("\n  Axioms Integrated. The Laws are set.")
    
    # 3. Genesis Run (WFC)
    print("\n---   Generating World State ---")
    wfc = FractalWFC(lexicon=elysia.mind)
    
    # Seed: "Creation"
    seed = WaveDNA(
        physical=0.5, functional=0.5, phenomenal=0.5, 
        causal=0.5, mental=0.5, structural=0.5, spiritual=0.5,
        label="Creation"
    )
    
    # Collapse Wave
    gen1 = wfc.collapse(seed, intensity=1.0)
    
    print(f"\n  World Snapshot (Generation 1):")
    generated_entities = []
    
    for child in gen1:
        name = wfc._guess_name(child)
        print(f"  -> {name} (DNA: P{child.physical:.2f}, S{child.spiritual:.2f})")
        generated_entities.append(name)
        
    # 4. Reality Check (Providence & Synapse)
    print("\n---   Reality Verification (Physics & Weight) ---")
    
    # Pick a generated entity to test (e.g. if 'Steam' or 'River' generated)
    # We will try to interact with one.
    target_raw = generated_entities[0] if generated_entities else "Fire"
    # Clean suffix (e.g. "magma (Aspect)" -> "magma")
    target = target_raw.split(" (")[0].strip()
    
    print(f"Testing Entity: '{target}' (Raw: '{target_raw}')")
    
    # A. Check if it has Mass (Synapse)
    core.summon_thought(target)
    rotor = core.rotors.get(target)
    
    if rotor:
        print(f"   [Synapse] Mass: {rotor.config.mass:.2f} kg")
        print(f"   [Synapse] RPM:  {rotor.config.rpm:.2f}")
    else:
        print(f"   [Synapse]   Failed to summon '{target}'.")
        
    # B. Check Thermodynamics (Providence)
    # Get vector from Lexicon
    vec = elysia.mind.analyze(target)
    print(f"   [Vector]  G:{vec.gravity:.2f}, F:{vec.flow:.2f}, A:{vec.ascension:.2f}")
    
    # Apply Heat
    print(f"     Applying Heat to '{target}'...")
    heat = elysia.mind.analyze("fire")
    new_state_vec = providence.apply_thermodynamics(vec, heat)
    
    # Analyze Result using simpler vector comparison for now
    # (Providence result is a Vector, we need to guess what it is)
    # If Asc increased significantly, physics works.
    print(f"   -> Old Asc: {vec.ascension:.2f}, New Asc: {new_state_vec.ascension:.2f}")
    
    if new_state_vec.ascension > vec.ascension:
        print("     [Providence] Entity reacted to Heat (Energy absorbed).")
    else:
        print("     [Providence] Entity is inert (Physics failed).")

if __name__ == "__main__":
    test_civilization_genesis_v2()