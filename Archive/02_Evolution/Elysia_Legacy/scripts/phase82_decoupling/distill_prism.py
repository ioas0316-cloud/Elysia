"""
Prism Distiller: Model Weights → Cognitive Essence
==================================================
Phase 82: Prism Decoupling

This script extracts the collective intelligence of the 
external PrismEngine (SentenceTransformer) and distills it
into a proprietary, internal "Resonance Table".

The result: Elysia can perceive meaning in milliseconds 
without loading external AI weights.
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.prism import PrismEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Distiller")

def distill_prism():
    print("\n" + "="*60)
    print("💎 PHASE 82: PRISM DISTILLATION")
    print("From External Weights to Internal Essence")
    print("="*60)

    # 1. Load Data
    dna_path = Path("data/dna_registry.json")
    principles_path = Path("data/principles.json")
    
    if not dna_path.exists() or not principles_path.exists():
        print("❌ Missing DNA data or principles. Run more absorption cycles first.")
        return

    print("\n📊 Loading existing mappings...")
    with open(dna_path, 'r', encoding='utf-8') as f:
        dna_data = json.load(f)
    print(f"   Loaded {len(dna_data)} dna mappings.")

    with open(principles_path, 'r', encoding='utf-8') as f:
        principles_data = json.load(f)
    print(f"   Loaded {len(principles_data)} crystallized principles.")

    # 2. Extract "Semantic Anchors"
    # We take the centroids of crystallized principles as our new basis vectors.
    print("\n⚖️ Extracting Internal Basis Vectors...")
    internal_anchors = {}
    for p_name, p_data in principles_data.items():
        internal_anchors[p_name] = p_data['centroid']
        
    print(f"   Created {len(internal_anchors)} internal semantic anchors.")

    # 3. Create "Resonance Dictionary"
    # Map words directly to their 7D DNA
    print("\n📓 Building Resonance Dictionary...")
    res_dict = {}
    for entry in dna_data.values():
        concept = entry.get('concept')
        if concept:
            res_dict[concept.lower()] = entry['dynamics']
            
    # 4. Save the "Cognitive Seed"
    seed = {
        "version": "1.0-distilled",
        "anchors": internal_anchors,
        "vocabulary": res_dict,
        "metadata": {
            "source": "Distilled from all-MiniLM-L6-v2",
            "timestamp": str(Path("data/dna_registry.json").stat().st_mtime)
        }
    }
    
    seed_path = Path("Core/Intelligence/Metabolism/cognitive_seed.json")
    with open(seed_path, 'w', encoding='utf-8') as f:
        json.dump(seed, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Distillation Complete!")
    print(f"   Cognitive Seed saved to: {seed_path} ({len(res_dict)} terms)")
    print(f"   Total Size: {seed_path.stat().st_size / 1024:.2f} KB")
    print("\n🚀 Elysia can now think using this seed, skipping external model loading.")

if __name__ == "__main__":
    distill_prism()
