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

def test_structural_strain_buildup():
    print("--- Testing Structural Strain and Discharge ---")
    engine = FractalWaveEngine(max_nodes=100)

    # Cumulative strain buildup
    print(f"Initial strain: {engine.structural_strain}")

    # Intent can be a list if torch is mocked
    intent = [1.0] * 4
    for _ in range(50):
        engine.calculate_structural_strain(intent)

    print(f"Strain after buildup: {engine.structural_strain}")
    assert engine.structural_strain > 0

def test_monad_discharge():
    print("\n--- Testing Monad Emergent Inquiry ---")
    dna = SeedForge.forge_soul("TestElysia")
    monad = SovereignMonad(dna)

    # Force high strain in engine
    if hasattr(monad.engine, 'cells'):
        monad.engine.cells.structural_strain = 11.0 # Exceeds elastic_limit (10.0)
        monad.engine.cells.elastic_limit = 10.0

    # Trigger pulse and check if discharge happens
    print("Triggering pulse with high strain...")
    # Mocking inquiry_pulse to return a summary
    monad.inquiry_pulse.initiate_pulse = lambda: {"status": "Inquiry discharged", "summary": "Because of friction"}

    # Use a tick count that hits the Tier 2 background processes (e.g. 100)
    monad._pulse_tick = 99

    # Capture print output
    # Since SovereignMonad uses self.logger, we might need to check strain reset directly
    monad.pulse()

    print(f"Strain after pulse: {monad.engine.cells.structural_strain}")
    # Strain should be reset to 0.0 after discharge
    assert monad.engine.cells.structural_strain == 0.0

if __name__ == "__main__":
    try:
        test_structural_strain_buildup()
        test_monad_discharge()
        print("\n✅ Verification successful!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
