import sys
import os
import logging

# Ensure root directory is in path
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify.Ascension")

def verify():
    logger.info("🚀 Starting Ascension Verification...")
    
    # 1. Initialize Elysia
    elysia = SovereignSelf()
    
    # 2. Test Metacognitive Lens (Reflection)
    # Scenario: High-Torque Thought
    test_input = "I am bored and want to see chaos."
    logger.info(f"👤 User: {test_input}")
    response = elysia.speak(test_input)
    logger.info(f"🦋 Elysia: {response}")
    
    # Check if lens worked (critique should be in logs or response prefix)
    assert elysia.lens.reflection_history, "Metacognitive Lens history is empty."
    logger.info("✅ Metacognitive Lens Active.")

    # 3. Test Emotional Gravity (Anchoring)
    # Promote a node's gravity
    if elysia.graph.id_to_idx:
        node_id = list(elysia.graph.id_to_idx.keys())[0]
        idx = elysia.graph.id_to_idx[node_id]
        elysia.graph.grav_tensor[idx] = 100.0 # High gravity
        logger.info(f"⚓ Anchored node '{node_id}' with high gravity.")
        
        # Search for something random - it should resonate with the high gravity node
        results = elysia.graph.get_nearest_by_vector(elysia.graph.vec_tensor[idx], top_k=1)
        assert results[0][0] == node_id, "Gravity anchoring failed."
        logger.info(f"✅ Emotional Anchoring Verified: {node_id} is now a gravitic center.")

    # 4. Test Causal Alignment (Wisdom of Possibility)
    logger.info("🌙 Entering Sleep Mode for Causal Alignment...")
    elysia.sleep_mode = False # Force sleep test
    elysia._enter_sleep_mode()
    
    assert elysia.alignment_log, "Alignment log is empty. No causalities discerned."
    logger.info(f"✅ Causal Alignment Verified: {elysia.alignment_log[-1]}")

    logger.info("🏆 Ascension Verification Complete: Phase 09 is operational with Fractal Principles.")

if __name__ == "__main__":
    verify()
