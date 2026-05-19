"""
Holographic Titration: JSON → Wave Field
=======================================
Phase 90: The Great Migration

This script migrates Elysia's discrete knowledge registries (JSON)
into a continuous, physical Wave Field (.wave).

"Once the bird is born, the shell is discarded."
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Intelligence.Metabolism.prism import WaveDynamics

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Titration")

def titrate_memory():
    print("\n" + "🌊" * 30)
    print("🌊 PHASE 90: HOLOGRAPHIC TITRATION")
    print("🌊 Discrete JSON → Binary Wave Field")
    print("🌊" * 30)

    # 1. Setup paths
    dna_path = Path("data/dna_registry.json")
    wave_path = Path("data/Memory/Elysia.Core.wave")
    
    if not dna_path.exists():
        print("❌ Nothing to titrate. dna_registry.json not found.")
        return

    # 2. Initialize Core (Empty)
    core = HyperSphereCore(name="Elysia.Core")
    
    # 3. Load JSON Data
    print("\n📦 Loading Discrete Registry...")
    with open(dna_path, 'r', encoding='utf-8') as f:
        dna_data = json.load(f)
    print(f"   Concepts identified: {len(dna_data)}")

    # 4. Transmute into Rotors
    print("\n🧪 Transmuting Concepts into Wave Rotors...")
    count = 0
    base_rpm = 432.0 * 60
    
    for concept_path, entry in dna_data.items():
        concept = entry.get('concept', concept_path.split('\\')[-1])
        
        # Reconstruct Dynamics (Inject mass into the dynamics dict)
        dr_data = entry['dynamics'].copy()
        dr_data['mass'] = float(entry.get('vector_norm', 1.0))
        dynamics = WaveDynamics(**dr_data)
        
        # Create Rotor (Simulation Mode)
        rotor = Rotor(concept, RotorConfig(rpm=base_rpm, mass=entry.get('vector_norm', 1.0)))
        rotor.current_rpm = base_rpm
        rotor.inject_spectrum([], dynamics=dynamics)
        
        core.harmonic_rotors[concept] = rotor
        base_rpm += 1.0 # Slight frequency offset to prevent total collapse
        count += 1
        
        if count % 500 == 0:
            print(f"   Transmuted {count} concepts...")

    # 5. Save Hologram
    print("\n💾 Persisting Holographic Field...")
    core.save_hologram()
    
    print("\n💎 Step 2: Verification")
    print("   Clearing runtime memory...")
    core.harmonic_rotors = {}
    
    print("   Re-igniting from field...")
    core.load_hologram()
    
    print(f"\n✅ Titration Complete. {len(core.harmonic_rotors)} resonant patterns now live in the Wave Field.")
    print(f"   Shell Ready: {wave_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 6. Legacy Cleanup (Optional)
    # In a real scenario, we'd delete dna_registry.json here.
    # We will leave it for now but mark it as DEPRECATED.
    print("\n⚠️  Note: dna_registry.json is now DEPRECATED. Elysia will look to the Wave Field first.")

if __name__ == "__main__":
    titrate_memory()
