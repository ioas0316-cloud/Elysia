"""
VERIFICATION: Phase 64 - The Alchemist
=======================================
Tests the Distill -> Transmute -> Purge cycle.
Knowledge is internalized, the source is deleted.
"""

import sys
import os
import logging
import time
import torch
import json

# Add workspace to path
sys.path.append(os.getcwd())

from Core.Cognition.elysian_heartbeat import ElysianHeartbeat
from Core.Cognition.archive_dreamer import DreamFragment

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Phase64Test")

def test_alchemist():
    logger.info("🎬 Starting Phase 64 Verification (The Alchemist)...")
    
    # 1. Initialize Heartbeat
    heartbeat = ElysianHeartbeat()
    
    # 2. Create a dummy model file to eat
    test_model = "temp_library_book.pt"
    torch.save({"layer1": torch.randn(100, 100)}, test_model)
    logger.info(f"📁 Created dummy model: {test_model} ({os.path.getsize(test_model)} bytes)")

    # 3. Capture initial state
    init_inspiration = heartbeat.soul_mesh.variables["Inspiration"].value
    logger.info(f"📊 Initial Inspiration: {init_inspiration:.4f}")

    # 4. Create a mock DreamFragment for the nutrient
    fragment = DreamFragment(
        path=os.path.abspath(test_model),
        name=test_model,
        type='nutrient',
        resonance=0.95,
        message="A rare book of ancient logic."
    )

    # 5. Trigger Transmutation (The Alchemist Cycle)
    logger.info(f"⚗️ Triggering Alchemical Transmutation of {fragment.name}...")
    heartbeat._transmute_model(fragment)

    # 6. Verify Results
    
    # Check if Axiom exists
    axiom_dir = "data/Knowledge/Axioms"
    axiom_found = False
    for filename in os.listdir(axiom_dir):
        if "temp_library_book" in filename:
            logger.info(f"✨ Found Crystallized Axiom: {filename}")
            axiom_found = True
            break
            
    # Check if source file is DELETED
    file_purged = not os.path.exists(test_model)
    if file_purged:
        logger.info("🔥 SUCCESS: Source file purged from the library.")
    else:
        logger.error("❌ FAILURE: Source file still exists dependency not removed.")

    # Check state boost
    final_inspiration = heartbeat.soul_mesh.variables["Inspiration"].value
    logger.info(f"📊 Final Inspiration: {final_inspiration:.4f}")

    if axiom_found and file_purged and (final_inspiration > init_inspiration):
        logger.info("✅ ALL SYSTEMS VERIFIED: Elysia has internalized the knowledge and burned the book.")
    else:
        logger.error("❌ VERIFICATION FAILED.")
        sys.exit(1)

    logger.info("🎉 Phase 64 Verification Complete.")

if __name__ == "__main__":
    test_alchemist()
