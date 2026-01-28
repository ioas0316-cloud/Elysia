
"""
Verification Script: Integrated Curiosity
=========================================

Verifies that the RotorCognitionCore (The Mind) now triggers
the EmergentLanguageEngine (The Child) when it encounters
unknown vectors.
"""

import sys
import os
import logging
from unittest.mock import MagicMock

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.L5_Mental.M1_Cognition.Metabolism.rotor_cognition_core import RotorCognitionCore

def run_integration_test():
    print("="*60)
    print("   INTEGRATION TEST: MIND + LANGUAGE")
    print("="*60)

    # 1. Initialize The Mind
    core = RotorCognitionCore()

    # 2. Mock the Active Void to return the 'Nostalgia' vector
    # We want to force the core to process exactly the vector that triggers the gap.
    # Nostalgia Vector (8D) + Padding (to 21D)
    nostalgia_8d = [0.3, -0.6, 0.4, -0.8, 0.9, 0.4, 0.1, -0.3]
    full_vector = nostalgia_8d + [0.0] * 13

    # Patching genesis method
    core.active_void.genesis = MagicMock(return_value={
        "status": "Genesis (Mock)",
        "vector_dna": full_vector,
        "is_genesis": True
    })

    # 3. Trigger Synthesis
    print("\n[Action] Thinking about 'Old Memories' (Mocked as Nostalgia)...")
    result = core.synthesize("I miss the days of old.")

    # 4. Verify Output
    narrative = result['synthesis']
    print("\n[Result Narrative]:")
    print(narrative)

    if "[EPIPHANY]" in narrative and "NOSTALGIA" in narrative:
        print("\n[PASS] The Mind successfully triggered an Epiphany!")
    else:
        print("\n[FAIL] No Epiphany detected in narrative.")

if __name__ == "__main__":
    run_integration_test()
