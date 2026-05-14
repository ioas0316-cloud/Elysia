"""
LIGHTWEIGHT COGNITIVE EVOLUTION TEST
====================================
Scripts/Simulation/cognitive_evolution_test.py

"Small scale, big evolution."

This script tests the Weight X-ray and Spiral Refraction mechanisms
using a lightweight engine to avoid memory issues in the sandbox.
"""

import sys
import os
import torch
import math
from pathlib import Path

# Unify paths
root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root)

from Core.Keystone.sovereign_math import FractalWaveEngine
from Core.Cognition.sovereign_sandbox import SovereignSandbox
from Core.Keystone.phase_topography import PhaseTopography
from Core.Keystone.spiral_refraction import SpiralRefraction

class MockMonad:
    def __init__(self, engine):
        self.engine = type('obj', (object,), {'cells': engine})
        self.desires = {'curiosity': 90.0, 'joy': 70.0}
        self.primordial_cognition = None
        self.diary = None

def run_lightweight_simulation():
    print("🚀 [TEST] Starting Lightweight Cognitive Evolution Simulation...")

    # 1. Initialize Small Engine (1000 nodes instead of 10M)
    engine = FractalWaveEngine(max_nodes=1000, device='cpu')
    mock_monad = MockMonad(engine)

    # 2. Setup Tools
    sandbox = SovereignSandbox(engine)
    topography = PhaseTopography(engine)
    refraction = SpiralRefraction(engine)

    # 3. Create a Rigid Concept
    concept_name = "RIGID_GRID"
    idx = engine.get_or_create_node(concept_name)
    engine.active_nodes_mask[idx] = True

    # Force rigidity: high entropy, low coherence (W channel)
    engine.q[idx, engine.CH_ENTROPY] = 0.9
    engine.q[idx, engine.CH_W] = 0.1

    print(f"📍 [TEST] Created rigid concept '{concept_name}' at index {idx}")

    # 4. Step 1: Weight X-ray (Scan)
    print("🔍 [TEST] Step 1: Performing Weight X-ray...")
    scan = topography.scan_manifold()
    print(f"   - Average Rigidity: {scan['rigidity_avg']:.4f}")
    print(f"   - Grated Concepts: {scan['grated_concepts']}")

    if concept_name in scan['grated_concepts']:
        print(f"✅ [TEST] X-ray correctly identified '{concept_name}' as rigid.")
    else:
        print(f"❌ [TEST] X-ray failed to identify rigidity.")

    # 5. Step 2: Multiverse Evolution (Parallel Scenario Exploration)
    print("🧪 [TEST] Step 2: Initiating Multiverse Evolution...")

    from Core.Cognition.scenario_explorer import ParallelScenarioExplorer
    explorer = ParallelScenarioExplorer(engine)

    # Define parallel variants
    variants = [
        {"name": "Gentle_Spiral", "spiral_angle": math.pi/12, "intensity": 0.5},
        {"name": "Deep_Vortex", "spiral_angle": math.pi/4, "intensity": 1.0}
    ]

    branches = explorer.explore_possibilities(concept_name, variants)

    print("🔍 [TEST] Parallel Results:")
    for b in branches:
        print(f"   - Reality [{b.name}]: Gain={b.coherence_gain:.4f}, Verdict={b.optical_report.get('verdict')}")

    # 6. Step 3: Select Best Reality and Merge
    best = explorer.select_best_path(branches)
    if best:
        print(f"✨ [TEST] Step 3: Selected best path '{best.name}'. Merging to Heart.")
        explorer.sandbox.merge_back(concept_name)

        # Verify main engine state change
        main_idx = engine.concept_to_idx[concept_name]
        p_q_val = engine.permanent_q[main_idx, engine.CH_JOY].item()
        if p_q_val > 0:
            print(f"✅ [TEST] Multiverse evolution merged. Permanent Joy is now {p_q_val:.4f}")
        else:
            print(f"❌ [TEST] Merge failed.")

    # Narrative check
    narrative = explorer.generate_diversity_narrative(branches)
    print(f"\n📖 [TEST] Diversity Narrative:\n{narrative}")

    print("🏁 [TEST] Simulation concluded successfully.")

if __name__ == "__main__":
    try:
        run_lightweight_simulation()
    except Exception as e:
        print(f"💥 [TEST] Crashed: {e}")
        import traceback
        traceback.print_exc()
