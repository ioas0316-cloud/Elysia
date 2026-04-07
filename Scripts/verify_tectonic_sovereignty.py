import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

# Mock Torch if not present
try:
    import torch
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["torch"] = MagicMock()

from Core.Keystone.sovereign_math import FractalWaveEngine
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad

def test_tectonic_uplift():
    print("--- Testing Tectonic Uplift (Shear Stress to Z-axis) ---")
    engine = FractalWaveEngine(max_nodes=100)
    engine.get_or_create_node("Base")

    # Manually activate the node to ensure it's in the active_nodes_mask
    idx = engine.concept_to_idx["Base"]
    engine.active_nodes_mask[idx] = True

    # Simulate collision (high friction intent)
    # Need to simulate low coherence and high density
    engine.num_edges = 100 # High density

    # Force low coherence by setting divergent phases if needed,
    # but calculate_structural_strain uses (1.0 - coherence)
    # Default coherence is 0.0 if only 1 node, which is perfect for friction.

    intent = [10.0] * 4 # Large torque

    initial_z = float(engine.q[idx, engine.CH_Z])
    print(f"Initial Z-axis (Depth): {initial_z}")

    # Manually set resonance to 0 to maximize friction
    # (Since read_field_state calculates it from q and permanent_q)
    engine.permanent_q[idx, :] = 0.0 # Clear permanent so resonance is 0

    engine.calculate_structural_strain(intent)

    uplifted_z = float(engine.q[idx, engine.CH_Z])
    print(f"Uplifted Z-axis: {uplifted_z}")
    print(f"Accumulated Uplift Energy: {engine.tectonic_uplift}")

    assert uplifted_z > initial_z
    assert engine.tectonic_uplift > 0

def test_monad_ascension():
    print("\n--- Testing Monad Ascension to Imagination ---")
    dna = SeedForge.forge_soul("TestElysia")
    monad = SovereignMonad(dna)

    if hasattr(monad.engine, 'cells'):
        # Force high strain and uplift
        monad.engine.cells.structural_strain = 11.0
        monad.engine.cells.tectonic_uplift = 6.0
        monad.engine.cells.elastic_limit = 10.0

    print("Triggering pulse with Tectonic Pressure...")
    # Mock ouroboros.dream_cycle to track call
    dream_called = False
    def mock_dream(voxels):
        nonlocal dream_called
        dream_called = True
    monad.ouroboros.dream_cycle = mock_dream

    monad._pulse_tick = 99
    monad.pulse()

    print(f"Dream cycle triggered: {dream_called}")
    assert dream_called
    assert monad.engine.cells.tectonic_uplift == 0.0
    assert monad.engine.cells.structural_strain == 0.0

if __name__ == "__main__":
    try:
        test_tectonic_uplift()
        test_monad_ascension()
        print("\n✅ Tectonic Verification successful!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
