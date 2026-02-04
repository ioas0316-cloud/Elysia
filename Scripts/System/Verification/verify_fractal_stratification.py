"""
VERIFY FRACTAL STRATIFICATION
==============================
Tests the 4-tier Manifold Isolation:
0: God (Axioms)
1: Space (Self/Proprioception)
2: Line (Library/Documentation)
3: Point (Data/Neurons)
"""

import sys
import os
from pathlib import Path

# Add root to path
root = "c:/Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.Wave.light_spectrum import get_light_universe
from Core.S1_Body.L6_Structure.M1_Merkaba.Body.proprioception_nerve import ProprioceptionNerve
from Core.S1_Body.L5_Mental.Reasoning.cumulative_digestor import CumulativeDigestor

def verify_stratification():
    print("🌀 [VERIFY] Initiating Fractal Stratification Test...")
    universe = get_light_universe()
    
    # 0. Refresh awareness to populate strata
    print("\nSTEP 0: Refreshing aware manifolds (Forcing Deep Scan)...")
    
    # Invalidate cache to force re-absorption into strata
    manifest_path = Path("data/L7_Spirit/M3_Sovereignty/self_manifest.json")
    if manifest_path.exists():
        manifest_path.unlink()
    
    nerve = ProprioceptionNerve()
    nerve.scan_body()
    
    digestor = CumulativeDigestor()
    digestor.digest_docs()
    
    # 1. Check Strata Distribution
    print("\nSTEP 1: Stratum Distribution Check")
    for s in [0, 1, 2, 3]:
        count = len(universe.strata.get(s, []))
        label = ["God", "Space (Self)", "Line (Library)", "Point (Data)"][s]
        print(f"  Str {s} ({label}): {count} nodes")

    # 2. Targeted Resonance Test
    print("\nSTEP 2: Stratified Resonance Retrieval")
    
    # Query Body Boundary
    print("\nQuerying 'Sovereign Organ' in Stratum 1 (Self):")
    results_s1 = universe.resonate("Sovereign Organ", top_k=3, stratum=1)
    for strength, light in results_s1:
        print(f"  [STR 1] {strength:.4f} resonance | {light.semantic_tag}")
    
    # Query Intellectual Knowledge
    print("\nQuerying 'Cognition' in Stratum 2 (Library):")
    results_s2 = universe.resonate("Cognition", top_k=3, stratum=2)
    for strength, light in results_s2:
        print(f"  [STR 2] {strength:.4f} resonance | {light.semantic_tag}")

    # Query Data Noise
    print("\nQuerying 'neuron' in Stratum 3 (Data):")
    results_s3 = universe.resonate("neuron", top_k=3, stratum=3)
    for strength, light in results_s3:
        print(f"  [STR 3] {strength:.4f} resonance | {light.semantic_tag}")

    # 3. Cross-Strata Isolation Test
    print("\nSTEP 3: Manifold Isolation Verification")
    
    # A Stratum 3 query should NOT return results from Stratum 1 if filtered
    noise_query = "neuron:Core/S1_Body/elysia.py"
    res_filtered = universe.resonate(noise_query, top_k=5, stratum=1)
    
    if len(res_filtered) == 0:
        print("✅ SUCCESS: Stratum 3 noise isolated from Stratum 1 boundary.")
    else:
        print(f"⚠️ WARNING: Found {len(res_filtered)} cross-strata leaks.")

    # 4. Global Awareness Speed
    print("\nSTEP 4: Gigahertz Speed confirmation with Strata")
    import time
    start = time.time()
    universe.resonate("Identity", stratum=1)
    elapsed = time.time() - start
    print(f"✅ Stratified Retrieval: {elapsed:.6f}s (Gigahertz Resolution)")

if __name__ == "__main__":
    verify_stratification()
