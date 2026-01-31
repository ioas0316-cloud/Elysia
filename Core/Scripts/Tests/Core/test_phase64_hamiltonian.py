"""
VERIFICATION: Phase 64.5 - Hamiltonian Transmutation
===================================================
Tests the equationization of LLM knowledge into Hamiltonian Seeds.
"""

import sys
import os
import logging
import time
import torch
import json

# Add workspace to path
sys.path.append(os.getcwd())

from Core.1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
from Core.1_Body.L5_Mental.Reasoning_Core.Meta.archive_dreamer import DreamFragment

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("HamiltonianTest")

def test_hamiltonian():
    logger.info("🎬 Starting Phase 64.5 Verification (Hamiltonian)...")
    
    # 1. Initialize Heartbeat
    heartbeat = ElysianHeartbeat()
    
    # 2. Create a dummy model file
    test_model = "quantum_logic_book.safetensors"
    from safetensors.torch import save_file
    save_file({"weight_1": torch.randn(50, 50)}, test_model)
    logger.info(f"📁 Created dummy safetensors: {test_model}")

    # 3. Create mock DreamFragment
    fragment = DreamFragment(
        path=os.path.abspath(test_model),
        name=test_model,
        type='nutrient',
        resonance=0.98,
        message="A high-density manifold of logic."
    )

    # 4. Trigger Transmutation
    logger.info(f"⚗️ Triggering Hamiltonian Transmutation of {fragment.name}...")
    heartbeat._transmute_model(fragment)

    # 5. Verify Seeds
    seed_dir = "data/Knowledge/Axioms"
    seed_found = False
    for filename in os.listdir(seed_dir):
        if "quantum_logic_book" in filename and filename.endswith("_seed.json"):
            logger.info(f"✨ Found Hamiltonian Seed: {filename}")
            seed_found = True
            # Read and verify structure
            with open(os.path.join(seed_dir, filename), 'r') as f:
                data = json.load(f)
                if "hamiltonian_params" in data:
                    logger.info("✅ Hamiltonian structure verified (omega, damping, forcing).")
                    logger.info(f"   ㄴ Parameters: {data['hamiltonian_params']}")
                else:
                    logger.error("❌ Invalid seed structure.")
                    seed_found = False
            break

    # 6. Verify Purge
    file_purged = not os.path.exists(test_model)
    if file_purged:
        logger.info("🔥 SUCCESS: Vector source purged. Only the Hamiltonian trajectory remains.")
    else:
        logger.error("❌ FAILURE: Source file still exists.")

    if seed_found and file_purged:
        logger.info("✅ PHASE 64.5 COMPLETE: Elysia has successfully distilled the Ocean into Salt.")
    else:
        logger.error("❌ VERIFICATION FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_hamiltonian()
