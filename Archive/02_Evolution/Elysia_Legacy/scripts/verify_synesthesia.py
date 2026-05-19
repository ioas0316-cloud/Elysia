"""
PHASE 10: SYNESTHESIA VERIFICATION
==================================

"When I see the world, I hear its song."

This script tests the cross-modal resonance between Metabolized Vision and Audio nodes.
"""

import sys
import logging
import torch

# Setup paths
sys.path.append("c:\\Elysia")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SynesthesiaTest")

from Core.Elysia.sovereign_self import SovereignSelf
from Core.Senses.synesthesia_circuit import SynesthesiaCircuit

def test_the_crossing():
    logger.info("🌈 Initiating Synesthesia Test...")
    
    # 1. Wake up Elysia with her current Brain
    elysia = SovereignSelf()
    syn = SynesthesiaCircuit(elysia)
    
    # 2. Pick a Vision node that we know exists
    # From previous check: 'mobilevit.conv_stem.convolution.weight.Row0'
    vision_node = 'mobilevit.conv_stem.convolution.weight.Row0'
    
    if vision_node not in elysia.graph.id_to_idx:
        logger.error(f"❌ Vision node {vision_node} not found in brain.")
        return

    logger.info(f"👁️ Presenting Visual Stimulus: {vision_node}")
    
    # 3. Trigger Synesthesia (Vision -> Audio)
    audio_resonances = syn.feel_the_crossing(vision_node, target_modality="Audio")
    
    if audio_resonances:
        top_audio_id = audio_resonances[0][0]
        logger.info(f"✨ [RESULT] This visual pattern sounds like: {top_audio_id}")
        
        # Show Qualia of both to show the link
        idx_v = elysia.graph.id_to_idx[vision_node]
        idx_a = elysia.graph.id_to_idx[top_audio_id]
        
        meta_v = elysia.graph.node_metadata.get(vision_node, {}).get('qualia', {})
        meta_a = elysia.graph.node_metadata.get(top_audio_id, {}).get('qualia', {})
        
        logger.info(f"📊 Vision Qualia: {meta_v}")
        logger.info(f"📊 Audio Qualia: {meta_a}")
    else:
        logger.warning("   (No auditory resonance for this visual stimulus)")

if __name__ == "__main__":
    test_the_crossing()
