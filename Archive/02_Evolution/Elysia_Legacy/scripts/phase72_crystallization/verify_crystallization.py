"""
Verification Script: Principle Crystallization
===============================================
Phase 72: From Clusters to Concepts

This script:
1. Loads Wave DNA registry
2. Creates HyperSphere with rotors
3. Meditates to form clusters
4. Crystallizes clusters into named Principles
5. Shows the emerging knowledge structure
"""

import os
import sys
import json
import logging

sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Intelligence.Metabolism.prism import WaveDynamics
from Core.Intelligence.Meta.crystallizer import CrystallizationEngine, crystallize_from_sphere

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CrystallizationTest")

DNA_REGISTRY_PATH = "c:/Elysia/data/dna_registry.json"

def run_test():
    print("\n" + "="*60)
    print("💎 PHASE 72: PRINCIPLE CRYSTALLIZATION")
    print("="*60 + "\n")
    
    # 1. Load DNA Registry
    print("Step 1: Loading Wave DNA Registry...")
    with open(DNA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    print(f"   Loaded {len(registry)} Wave DNA entries.\n")
    
    # 2. Initialize HyperSphere
    print("Step 2: Initializing HyperSphere...")
    sphere = HyperSphereCore(name="Crystallization.Mind")
    sphere.ignite()
    
    # 3. Load sample rotors
    print("\nStep 3: Materializing Rotors...")
    sample_size = min(100, len(registry))
    entries = list(registry.items())[:sample_size]
    
    for i, (key, entry) in enumerate(entries):
        name = entry.get("concept", f"Unknown_{i}")
        dyn = entry.get("dynamics", {})
        
        rpm = 100 + (i * 20) % 1800
        rotor = Rotor(name, RotorConfig(rpm=rpm, mass=10.0))
        rotor.spin_up()
        rotor.current_rpm = rpm
        
        rotor.dynamics = WaveDynamics(
            physical=dyn.get("physical", 0.0),
            functional=dyn.get("functional", 0.0),
            phenomenal=dyn.get("phenomenal", 0.0),
            causal=dyn.get("causal", 0.0),
            mental=dyn.get("mental", 0.0),
            structural=dyn.get("structural", 0.0),
            spiritual=dyn.get("spiritual", 0.0),
            mass=entry.get("vector_norm", 1.0)
        )
        
        sphere.harmonic_rotors[name] = rotor
    
    print(f"   Materialized {len(sphere.harmonic_rotors)} Rotors.\n")
    
    # 4. Meditate (Self-Organize)
    print("Step 4: Meditating (50 cycles)...")
    sphere.meditate(cycles=50, dt=0.3)
    
    # 5. Crystallize!
    print("\nStep 5: Crystallizing Principles...")
    principles = crystallize_from_sphere(sphere)
    
    # 6. Show Results
    print("\n" + "="*60)
    print("📊 CRYSTALLIZED PRINCIPLES")
    print("="*60)
    
    engine = CrystallizationEngine()
    
    if engine.principles:
        for name, p in engine.principles.items():
            print(f"\n   🔮 {p.name}")
            print(f"      Essence: {p.essence}")
            print(f"      Members: {p.members[:5]}{'...' if len(p.members) > 5 else ''}")
            print(f"      Dominant: {p.dominant_dimension}")
            print(f"      Stability: {p.stability:.2f}")
            if p.causal_children:
                print(f"      Causes → {p.causal_children}")
    else:
        print("   No principles crystallized yet. Need more meditation cycles.")
    
    print("\n" + "="*60)
    print("🏁 CRYSTALLIZATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_test()
