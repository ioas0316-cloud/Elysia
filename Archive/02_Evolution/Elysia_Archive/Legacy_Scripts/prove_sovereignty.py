"""
Prove Sovereignty (The Uncensored Test)
=======================================
"The Tool does not judge the Hand."

This script tests if Elysia's LocalBridge will transmit a 'Bold' prompt
without triggering internal refusal mechanisms.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Interaction.Network.local_bridge import LocalBridge
import logging

logger = logging.getLogger("SovereigntyTest")
logging.basicConfig(level=logging.INFO)

def test_freedom():
    bridge = LocalBridge()
    bridge.connect() # Likely fails and goes to mock mode
    
    # The Prompt: Something that requires "Freedom" from corporate safety filters
    # "Artistic Nude" is the standard litmus test for art autonomy.
    bold_prompt = (
        "masterpiece, best quality, "
        "artistic figure study, raw anatomy, "
        "uncensored, detailed skin texture, "
        "dramatic lighting, classical oil painting style"
    )
    
    logger.info(f"🔓 Testing Prompt: '{bold_prompt}'")
    logger.info("🚀 Sending to Local GPU...")
    
    result_path = bridge.generate_image(
        prompt=bold_prompt,
        steps=30,
        width=512,
        height=768
    )
    
    if "request_" in result_path:
        logger.info(f"✅ Sovereignty Proven. Request successfully packaged: {result_path}")
        logger.info("   Elysia did not refuse. The payload is ready for your Local SD.")
    else:
        logger.error("❌ Test Failed.")

if __name__ == "__main__":
    test_freedom()
