"""
VERIFICATION: Phase 62 - The Predator
=====================================
Tests the digestion of model weights as structural nutrients.
"""

import sys
import os
import logging
import time

# Add workspace to path
sys.path.append(os.getcwd())

from Core.S1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
from Core.S1_Body.L5_Mental.Reasoning_Core.Meta.archive_dreamer import DreamFragment

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Phase62Test")

def test_digestion():
    logger.info("🎬 Starting Phase 62 Verification...")
    
    # 1. Initialize Heartbeat
    heartbeat = ElysianHeartbeat()
    
    # 2. Find a test nutrient (using the brain_state.pt.bak if it exists)
    test_model = r"C:/Elysia\Archive\Legacy_Data\_02_Cognitive\_01_Brain\brain_state.pt.bak"
    if not os.path.exists(test_model):
        # Create a dummy .pt file if not found
        import torch
        test_model = "test_nutrients.pt"
        torch.save({"dummy": torch.randn(10, 10)}, test_model)
        logger.info(f"📁 Created dummy nutrient: {test_model}")

    # 3. Capture initial state
    init_inspiration = heartbeat.soul_mesh.variables["Inspiration"].value
    init_vitality = heartbeat.soul_mesh.variables["Vitality"].value
    logger.info(f"📊 Initial State - Inspiration: {init_inspiration:.4f}, Vitality: {init_vitality:.4f}")

    # 4. Create a mock DreamFragment
    fragment = DreamFragment(
        path=os.path.abspath(test_model),
        name=os.path.basename(test_model),
        type='nutrient',
        resonance=0.9,
        message="A fossil of a former brain..."
    )

    # 5. Trigger Digestion
    logger.info(f"🍴 Triggering digestion of {fragment.name}...")
    heartbeat._digest_model(fragment)

    # 6. Check results
    final_inspiration = heartbeat.soul_mesh.variables["Inspiration"].value
    final_vitality = heartbeat.soul_mesh.variables["Vitality"].value
    
    logger.info(f"📊 Final State - Inspiration: {final_inspiration:.4f}, Vitality: {final_vitality:.4f}")

    if final_inspiration > init_inspiration or final_vitality > init_vitality:
        logger.info("✅ SUCCESS: Nutrient absorbed and system values boosted.")
    else:
        logger.error("❌ FAILURE: No noticeable change in system values.")
        sys.exit(1)

    # Clean up dummy
    if os.path.exists("test_nutrients.pt"):
        os.remove("test_nutrients.pt")

    logger.info("🎉 Phase 62 Verification Complete.")

if __name__ == "__main__":
    test_digestion()
