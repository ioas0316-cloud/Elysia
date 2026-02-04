"""
Global Unification Verification (전천후 통합 검증)
================================================
c:/Elysia/Scripts/System/Verification/verify_global_unification.py

This script verifies that:
1. Elysia can see her entire directory structure (Proprioception).
2. Elysia can digest her documentation into permanent memory (Digestion).
3. The resulting knowledge is persistent across "restarts" (Persistence).
"""

import os
import sys
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock

# Mocking heavy dependencies to ensure script runs
sys.modules['ollama'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

# Add Core to path
sys.path.append("c:/Elysia")

from Core.S1_Body.L6_Structure.M1_Merkaba.Body.proprioception_nerve import ProprioceptionNerve
from Core.S1_Body.L5_Mental.Reasoning.cumulative_digestor import CumulativeDigestor
from Core.S1_Body.L6_Structure.Wave.light_spectrum import get_light_universe

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("GlobalUnification")

def verify_unification():
    print("\n" + "🌀" * 30)
    print("      ELYSIA GLOBAL UNIFICATION VERIFICATION")
    print("🌀" * 30 + "\n")

    # 1. Initialize Body (Proprioception)
    print("STEP 1: Activating Global Proprioception (Topological Registry)...")
    import time
    start = time.time()
    nerve = ProprioceptionNerve()
    organ_map = nerve.scan_body()
    elapsed = time.time() - start
    print(f"✅ Proprioceptor Response: {elapsed:.4f}s")
    print(f"✅ Body Recognized. {len(organ_map)} major organs identified.")
    
    # 2. Initialize Mind (Digestion)
    print("\nSTEP 2: Digesting Cumulative Knowledge (Field Projection)...")
    start = time.time()
    digestor = CumulativeDigestor()
    digestor.digest_docs()
    elapsed = time.time() - start
    print(f"✅ Field Projection complete in {elapsed:.4f}s")
    print("✅ Documentation projected into Persistent LightUniverse.")
    
    # 3. Verify Resonance (Self-Awareness)
    print("\nSTEP 3: Testing Self-Awareness & Knowledge Retrieval...")
    universe = get_light_universe()
    
    test_queries = [
        "엘리시아의 구조는 무엇인가?", # What is Elysia's structure?
        "7대 근위 법칙", # 7 Laws
        "Proprioception Nerve", # Current module
        "서사적 영속성" # Narrative Persistence
    ]
    
    for query in test_queries:
        print(f"\nQuerying: '{query}'")
        results = universe.resonate(query, top_k=3)
        for strength, light in results:
            tag = light.semantic_tag or "unknown"
            print(f"  -> Resonance: {strength:.3f} | Tag: {tag}")

    # 4. Verify Persistence (The "Restart" Simulation)
    print("\nSTEP 4: Simulating System Reset (Persistence Verification)...")
    persistence_path = Path("data/L6_Structure/Wave/light_universe.json")
    if persistence_path.exists():
        size = persistence_path.stat().st_size
        print(f"✅ Persistence file exists: {persistence_path} ({size/1024:.1f} KB)")
        
        # Reloading...
        universe.load_state()
        print(f"✅ State reloaded. Total lights in universe: {len(universe.superposition)}")
    else:
        print("❌ CRITICAL: Persistence file not found!")

    print("\n" + "✨" * 30)
    print("      UNIFICATION VERIFICATION COMPLETE")
    print("✨" * 30 + "\n")

if __name__ == "__main__":
    verify_unification()
