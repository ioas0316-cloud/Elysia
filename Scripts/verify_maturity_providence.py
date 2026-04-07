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

from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Divine.cognitive_field import CognitiveField

def test_maturity_trigger():
    print("--- Testing Maturity-Driven Re-evaluation ---")
    dna = SeedForge.forge_soul("TestMaturity")
    monad = SovereignMonad(dna)

    # 1. Inject a concept into the lexicon and force its maturity
    crystal = monad.lexicon.ingest("RAIN", "Water falling from the sky", "Sensation")
    crystal.strength = 0.95 # Above threshold (0.9)

    print(f"Concept '{crystal.name}' strength: {crystal.strength}")

    # 2. Mock inquiry pulse to track call
    re_eval_called = False
    def mock_initiate(focus_context=None):
        nonlocal re_eval_called
        if focus_context and "Providential 'Why' of RAIN" in focus_context:
            re_eval_called = True
        return {"status": "Complete", "summary": "Maturity test"}

    monad.inquiry_pulse.initiate_pulse = mock_initiate

    # Trigger Tier 2 tick
    monad._pulse_tick = 99
    monad.pulse()

    print(f"Maturity-driven re-evaluation triggered: {re_eval_called}")
    assert re_eval_called
    assert "RAIN" in monad.matured_concepts

def test_environmental_casting():
    print("\n--- Testing Environmental Casting (Providence) ---")
    field = CognitiveField()

    # Initial atmosphere
    initial_atmos = field.get_semantic_atmosphere()

    # 1. Mature a specific monad (e.g. LOVE/AGAPE)
    if "LOVE/AGAPE" in field.monads:
        love_monad = field.monads["LOVE/AGAPE"]
        love_monad.charge = 0.9 # Maturity threshold is 0.8 in get_semantic_atmosphere

        matured_atmos = field.get_semantic_atmosphere()

        # The matured atmosphere should be closer to LOVE/AGAPE
        res_initial = initial_atmos.resonance_score(love_monad.current_vector)
        res_matured = matured_atmos.resonance_score(love_monad.current_vector)

        print(f"Initial resonance with LOVE: {res_initial:.3f}")
        print(f"Matured resonance with LOVE: {res_matured:.3f}")

        assert res_matured > res_initial

if __name__ == "__main__":
    try:
        # Install numpy if needed (handled in bash earlier)
        test_maturity_trigger()
        test_environmental_casting()
        print("\n✅ Ontological Maturity Verification successful!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
