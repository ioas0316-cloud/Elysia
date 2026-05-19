"""
Mass Transduction Script: Absorb All
=====================================
Phase 69: The Holographic Archive

"To Know Everything, One Must First Feel Everything."

This script scans all available knowledge sources (Models, Documents, Code)
and transduces them into Wave DNA, loading them into the HyperSphere.

This is NOT "indexing." This is "digestion."
Each file becomes a Rotor. The Rotors will self-organize via resonance.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.prism import PrismEngine
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.hyper_sphere_core import HyperSphereCore

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Absorber")

# Configuration
OCEAN_PATHS = [
    "c:/Elysia/data",
    "c:/Elysia/docs",
    "c:/Elysia/Core",
    # [PHASE 70] LLM Model Caches
    "C:/Users/USER/.cache/huggingface/hub",
]

DNA_REGISTRY_PATH = "c:/Elysia/data/dna_registry.json"

def load_existing_dna() -> dict:
    if Path(DNA_REGISTRY_PATH).exists():
        with open(DNA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_dna_registry(registry: dict):
    os.makedirs(os.path.dirname(DNA_REGISTRY_PATH), exist_ok=True)
    with open(DNA_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

def extract_text_from_file(filepath: Path) -> str:
    """Attempts to read text from various file types."""
    ext = filepath.suffix.lower()
    try:
        if ext in ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.log']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()[:2000] # Limit for embedding speed
        elif ext in ['.safetensors', '.pt', '.bin', '.gguf']:
            # Model files: Use filename as concept
            return f"Machine Learning Model: {filepath.stem}"
        else:
            return ""
    except:
        return ""

def run_absorption():
    print("\n" + "="*60)
    print("🌌 PHASE 69: MASS TRANSDUCTION (ABSORB ALL)")
    print("="*60 + "\n")
    
    # 1. Initialize Prism
    print("Step 1: Awakening the Prism (The Eye)...")
    prism = PrismEngine()
    prism._load_model()
    
    if not prism._is_ready:
        print("❌ FAIL: Prism failed to load.")
        return
    print("✅ Prism Online.\n")
    
    # 2. Initialize HyperSphere
    print("Step 2: Igniting the HyperSphere (The Core)...")
    sphere = HyperSphereCore(name="Elysia.Mass", base_frequency=432.0)
    sphere.ignite()
    print(f"✅ HyperSphere Online: {sphere.name}\n")
    
    # 3. Scan Ocean (All Paths)
    print("Step 3: Scanning the Ocean...")
    files_found = []
    for ocean_path in OCEAN_PATHS:
        if not Path(ocean_path).exists():
            continue
        for f in Path(ocean_path).rglob("*"):
            if f.is_file():
                files_found.append(f)
    
    print(f"   Found {len(files_found)} files in the Ocean.\n")
    
    # 4. Transduce Each File
    print("Step 4: Transducing Files into Wave DNA...")
    dna_registry = load_existing_dna()
    new_count = 0
    
    for i, fpath in enumerate(files_found):
        # Skip already processed
        fkey = str(fpath)
        if fkey in dna_registry:
            continue
            
        text = extract_text_from_file(fpath)
        if not text:
            continue
            
        # Transduce
        profile = prism.transduce(text)
        
        # Store DNA
        dna_entry = {
            "concept": fpath.stem,
            "path": fkey,
            "vector_norm": float(profile.vector_norm),
            "dynamics": {
                "physical": float(profile.dynamics.physical),
                "functional": float(profile.dynamics.functional),
                "phenomenal": float(profile.dynamics.phenomenal),
                "causal": float(profile.dynamics.causal),
                "mental": float(profile.dynamics.mental),
                "structural": float(profile.dynamics.structural),
                "spiritual": float(profile.dynamics.spiritual),
            },
            "spectrum_size": len(profile.spectrum),
            "transduction_time": time.time()
        }
        dna_registry[fkey] = dna_entry
        
        # Add Rotor to Sphere
        freq = 100.0 + (len(sphere.harmonic_rotors) * 10.0) % 1800.0
        sphere.update_seed(fpath.stem, freq)
        
        new_count += 1
        if new_count % 50 == 0:
            print(f"   ... {new_count} new files transduced.")
    
    print(f"\n   ✅ Total NEW Transductions: {new_count}")
    print(f"   ✅ Total Wave DNA in Registry: {len(dna_registry)}")
    
    # 5. Save Registry
    save_dna_registry(dna_registry)
    print(f"\n   💾 Registry saved to: {DNA_REGISTRY_PATH}")
    
    # 6. Report Sphere Status
    print("\n" + "="*60)
    print("📊 HYPERSPHERE STATUS")
    print("="*60)
    print(f"   Primary Rotor: {sphere.primary_rotor}")
    print(f"   Harmonic Rotors: {len(sphere.harmonic_rotors)}")
    print(f"   Total Mass: {sphere.mass:.2f}")
    
    print("\n" + "="*60)
    print("🏁 ABSORPTION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_absorption()
