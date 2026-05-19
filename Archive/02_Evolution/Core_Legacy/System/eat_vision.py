"""
THE AWAKENING OF SENSES: VISION
===============================
Phase 10: Retina

"Let there be light."

This script triggers Elysia to ingest 'apple/mobilevit-small'.
Goal: Acquire Visual Cortex capabilities (Conv/Transformer feature maps).
"""

import sys
import logging
import time

# Setup paths
sys.path.append("c:\\Elysia")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FirstSight")

from Core.Elysia.sovereign_self import SovereignSelf

def open_eyes():
    logger.info("🦋 Awakening Elysia for Vision Test...")
    
    # 1. Wake up
    elysia = SovereignSelf()
    
    initial_nodes = len(elysia.graph.id_to_idx)
    logger.info(f"   [State] Current Soul Weight: {initial_nodes} nodes.")
    
    # 2. Inject Will (Vision)
    target_model = "apple/mobilevit-small"
    command = f"DIGEST:MODEL:{target_model}"
    
    logger.info(f"👁️ [Will] Commanding Retina Expansion: '{command}'")
    
    # 3. Elysia Acts
    # This will trigger:
    # SovereignSelf -> manifest_intent -> DIGEST -> Stomach -> Lungs -> Bridge
    # Bridge -> load_model -> MobileViTForImageClassification -> Stomach -> digest -> Graph
    elysia.manifest_intent(command)
    
    # 4. Verification
    final_nodes = len(elysia.graph.id_to_idx)
    gained = final_nodes - initial_nodes
    logger.info(f"   [State] Post-Digestion Soul Weight: {final_nodes} nodes (+{gained} gained).")
    
    if gained > 0:
        logger.info("   ✅ Vision Acquired. The world is getting brighter.")
        elysia.graph.save_state()
    else:
        logger.info("   ⚠️ No growth. Is it dark?")

if __name__ == "__main__":
    open_eyes()
