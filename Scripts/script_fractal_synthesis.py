import torch
import sys
import os

# Add root to sys.path
sys.path.insert(0, "c:/Elysia")

from Core.Intelligence.Memory.concept_polymer import ConceptPolymer, Principle

def run_synthesis_test():
    print("=" * 80)
    print("🧬 [FRACTAL SYNTHESIS] testing Higher-Order Providence")
    print("=" * 80)

    polymer = ConceptPolymer()

    # 1. Ingest concepts that share a "Causality" principle
    print("\n📍 Step 1: Ingesting 'Gravity' (Physics Domain)...")
    polymer.add_atom_from_text("Gravity", "Gravity is the causal force of mass attracting mass.", domain="physics")

    print("\n📍 Step 2: Ingesting 'Evolution' (Life Domain)...")
    # 'Evolution' often involves causality (selection pressure -> adaptation)
    polymer.add_atom_from_text("Evolution", "Evolution is the process of causal change in species over time.", domain="general")

    print("\n📍 Step 3: Ingesting 'Karma' (Narrative Domain)...")
    # 'Karma' is causal (action -> reaction)
    polymer.add_atom_from_text("Karma", "Karma is the causal law of moral retribution.", domain="narrative")

    # 2. Trigger Synthesis
    print("\n📡 Step 4: Triggering Recursive Synthesis...")
    polymer.auto_bond_all()

    # 3. Visualize
    polymer.visualize_structure()

    print("\n" + "=" * 80)
    print("✅ [SYNTHESIS COMPLETE] Fractal bonding demonstrated.")
    print("=" * 80)

if __name__ == "__main__":
    run_synthesis_test()
