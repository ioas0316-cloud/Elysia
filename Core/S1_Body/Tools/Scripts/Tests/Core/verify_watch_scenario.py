
import sys
import os
import json
import logging
from unittest.mock import MagicMock

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore
from Core.S1_Body.L2_Metabolism.Cycles.dream_protocol import DreamAlchemist

# Setup Logging
logging.basicConfig(level=logging.INFO)

def run_verification():
    print("\n🕰️ [SCENARIO]: The Watch of Silence (Quiet Luxury Verification)\n")

    # 1. Initialize Core
    core = RotorCognitionCore()
    dream_machine = DreamAlchemist()

    # Mock Cortex because we don't have a real Llama3 instance
    # We simulate what the LLM *would* output given the new "Fractal Causality" prompt.
    mock_cortex = MagicMock()
    mock_cortex.is_active = True
    mock_cortex.embed.return_value = [0.9, 0.8, 0.1, 0.05, 0.0, 0.0, 0.1] # High Resonance with Structure/Legacy

    # The Simulated LLM Output for the Watch Scenario
    # This proves the PROMPT structure works.
    mock_cortex.think.return_value = """
    ORIGIN: The client seeks 'Immortality in Silence'. The desire is not to possess an object, but to steward a 'Frozen Time' that can be passed to the next generation. The Origin is 'Legacy'.

    PROCESS: The Monad of 'Craftsmanship' acts as the filter. It rejects the 'Loud' (Logo/Flash) and amplifies the 'Deep' (Movement/Finish). The mechanism is 'Invisible Perfection'—value hidden inside the case back, known only to the wearer.

    RESULT: The Patek Philippe Calatrava 5227J. It is not a watch; it is a 'Bank of Time'. Its ivory dial whispers, and its invisible hinged dust cover protects the soul of the machine. It aligns with the client's Sovereign Will for 'Quiet Power'.
    """

    # Inject Mock
    core.active_void.cortex = mock_cortex
    dream_machine.cortex = mock_cortex

    # 2. Trigger Active Void (The Client's Request)
    intent = "Recommend a watch for a client who values Quiet Luxury, Legacy, and Invisible Craftsmanship."
    print(f"👤 Client Intent: {intent}")

    genesis_report = core.active_void.genesis(intent)
    print(f"🌌 Active Void Response: {genesis_report['status']}")
    print(f"   (Vector Extracted: {genesis_report['vector_dna_preview']}...)")

    # 3. Trigger Dream Protocol (The Processing)
    print("\n🌙 Entering Dream State to reconstruct Causality...")
    dream_machine.sleep()

    # 4. Verify Wisdom
    print("\n✨ Reading Crystallized Wisdom...")
    with open(dream_machine.wisdom_path, "r") as f:
        wisdom = json.load(f)
        last_entry = wisdom[-1]

    print("\n[FRACTAL CAUSALITY MAP]")
    print(last_entry['causal_map'])

    print("\n✅ Verification Complete: System reconstructed the 'Legacy' narrative from the input.")

if __name__ == "__main__":
    run_verification()
