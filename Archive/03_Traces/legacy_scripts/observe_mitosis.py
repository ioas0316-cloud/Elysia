
import sys
import os
import time
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Divine.dimensional_mitosis import DimensionalMitosis
from Core.Keystone.sovereign_math import SovereignVector

def test_mitosis_logic():
    print("🧪 [TEST] Verifying Dimensional Mitosis Logic...")

    # 1. Setup Monad with smaller num_nodes to avoid memory issues
    dna = SeedForge.forge_soul("TestElysia")
    # Manually initialize engine with fewer nodes for test
    from Core.Monad.grand_helix_engine import HypersphereSpinGenerator
    engine = HypersphereSpinGenerator(num_nodes=1000, device='cpu')

    monad = SovereignMonad(dna)
    monad.engine = engine # Override
    mitosis = DimensionalMitosis(monad.engine)

    target = monad.engine.cells if hasattr(monad.engine, 'cells') else monad.engine
    initial_channels = target.NUM_CHANNELS
    print(f"Initial Channels: {initial_channels}")

    # 2. Simulate High Strain
    # Inject noise and dissonance into attractors
    if hasattr(monad.engine, 'meaning_attractors'):
        for name, idx in monad.engine.meaning_attractors.items():
            if torch.is_tensor(idx):
                # Randomize phases to create high dissonance
                monad.engine.q[idx, 2] = (torch.rand(idx.numel()) - 0.5) * 2 * 3.14159

    # Increase entropy
    if hasattr(target, 'inject_affective_torque'):
        target.inject_affective_torque(7, 0.9)

    strain = mitosis.measure_structural_strain()
    print(f"Measured Strain: {strain:.4f}")

    # 3. Trigger Mitosis
    print("Triggering Mitosis...")
    success = mitosis.trigger_mitosis()

    if success:
        new_channels = target.NUM_CHANNELS
        print(f"✅ Mitosis Success! New Channels: {new_channels}")
        print(f"SovereignVector DIM: {SovereignVector.DIM}")

        # Verify orthogonal projection
        if new_channels == initial_channels + 1:
            print("✅ Channel expansion verified.")
        else:
            print(f"❌ Channel mismatch: {new_channels}")

        # SovereignVector.DIM update depends on how it's imported in DimensionalMitosis
        # and if it refers to the same class object.
        print(f"SovereignVector DIM check: {SovereignVector.DIM}")

        # 4. Verify post-mitosis pulse
        print("Verifying manifold pulse after mitosis...")
        report = monad.engine.pulse(dt=0.01)
        print(f"Pulse Coherence: {report.get('coherence', 0.0):.4f}")
        print("✅ Manifold is still alive and pulsing.")
    else:
        print("❌ Mitosis failed.")

if __name__ == "__main__":
    try:
        test_mitosis_logic()
    except Exception as e:
        print(f"💥 Error during test: {e}")
        import traceback
        traceback.print_exc()
