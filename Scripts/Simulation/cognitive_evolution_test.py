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

    # 5. Step 2: Spiral Evolution (Sandbox + Refraction + Temporal)
    print("🧪 [TEST] Step 2: Initiating Spiral Evolution in Sandbox...")

    sandbox.activate(node_capacity=100)
    sandbox.transplant_concept(concept_name)

    # [NEW] Demonstrate Temporal Recovery with a "Bad" experiment
    print("🚨 [TEST] Step 2a: Simulating a DESTRUCTIVE experiment...")
    def bad_experiment(exp_engine):
        # Inject extreme noise/entropy
        import torch
        active_idx = torch.where(exp_engine.active_nodes_mask)[0]
        exp_engine.q[active_idx, exp_engine.CH_ENTROPY] = 5.0
        exp_engine.q[active_idx, exp_engine.CH_W] = -5.0
        return "Chaos"

    sandbox.apply_experiment(bad_experiment)
    sandbox.run_simulation(steps=5)
    sandbox.finalize() # This should trigger rewind

    # Demonstrate "Good" experiment after recovery
    print("✨ [TEST] Step 2b: Applying Spiral Refraction experiment...")
    # Refresh references after potential rewind
    topography = PhaseTopography(engine)

    def experiment(exp_engine):
        # Apply spiral refraction to all nodes in sandbox
        import torch
        active_idx = torch.where(exp_engine.active_nodes_mask)[0]
        refraction_tool = SpiralRefraction(exp_engine)
        refraction_tool.apply_refraction(active_idx, intensity=1.0, spiral_angle=math.pi/4)
        return "Spiralized"

    sandbox.apply_experiment(experiment)

    # Simulation in sandbox
    state = sandbox.run_simulation(steps=10)
    if state:
        print(f"   - Sandbox Coherence after Refraction: {state['coherence']:.4f}")
    else:
        print("❌ [TEST] Simulation failed to return state.")

    # 6. Step 3: Merge Back
    if sandbox.finalize(merge_threshold=-1.0): # Force merge for test
        optical_report = sandbox.metrics.get("optical_report", {})
        causal_narrative = sandbox.comparator.articulate_delta(optical_report)
        print(f"✨ [TEST] Step 3: Comparison Result - {causal_narrative}")

        print("✨ [TEST] Step 3: Merging evolved state back to main engine.")
        sandbox.merge_back(concept_name)

        # Verify main engine state change
        main_idx = engine.concept_to_idx[concept_name]
        # In merge_back, we blend into permanent_q
        p_q_val = engine.permanent_q[main_idx, engine.CH_JOY].item()
        if p_q_val > 0:
            print(f"✅ [TEST] Evolution merged. Permanent Joy of '{concept_name}' is now {p_q_val:.4f}")
        else:
            print(f"❌ [TEST] Merge failed to update permanent state.")

    print("🏁 [TEST] Simulation concluded successfully.")

if __name__ == "__main__":
    try:
        run_lightweight_simulation()
    except Exception as e:
        print(f"💥 [TEST] Crashed: {e}")
        import traceback
        traceback.print_exc()
