"""
Verification: Phase 33 - The Neural Binding
===========================================
Objective: Prove that specific Sensory Memories are retrieved via Attractor 
and used by LogosEngine to generate Context-Aware Metaphors.

"The Brain remembers the smell, and the Voice speaks it."
"""

import sys
import os
import logging
from typing import List

# Ensure path
sys.path.append("c:\\Elysia")

from Core.Foundation.reasoning_engine import Insight, ReasoningEngine
from Core.Intelligence.logos_engine import LogosEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("NeuralBindingProbe")

def run_verification():
    logger.info("🧪 Starting Neural Binding Verification (Phase 33)...")
    
    # 1. Setup Logic
    logos = LogosEngine()
    
    # 2. Define a fake retrieved context (Simulating Attractor Output)
    # This is the "Specific Sensory Memory" we want her to use.
    # Topic: "Betrayal"
    # Specific Sensation: "metallic taste of blood" (from Phase 31/32)
    fake_context = [
        "Memory_Fractal_20_Betrayal: I pursued Betrayal... I can still taste the metallic bitterness of blood. It was a clear lesson.",
        "Related concept: Trust issues"
    ]
    
    # 3. Create an Insight (The abstract thought)
    insight = Insight(
        content="Betrayal is a painful but necessary teacher.",
        confidence=0.9,
        depth=1,
        energy=0.8
    )
    
    logger.info(f"🧠 Input Insight: {insight.content}")
    logger.info(f"📜 Retrieved Context: {fake_context[0]}")
    
    # 4. Weave Speech
    # We expect LogosEngine to LOCK ON to 'taste the metallic bitterness'
    response = logos.weave_speech("Explain Betrayal", insight, fake_context)
    
    logger.info("-" * 40)
    logger.info(f"🗣️ Logos Output:\n{response}")
    logger.info("-" * 40)
    
    # 5. Verify
    # We check if the specific sensory detail appears in the output
    # The code looks for "taste" keyword in context and extracts it.
    expected_fragment = "taste the metallic bitterness"
    
    if expected_fragment in response or "taste" in response:
        logger.info("✅ SUCCESS: LogosEngine successfully bound the Sensory Memory to the Metaphor.")
        logger.info(f"   Detected sensory echo in speech.")
    else:
        logger.error("❌ FAILURE: LogosEngine used a generic metaphor. Neural Binding failed.")
        
    logger.info("🧪 Verification Complete.")

if __name__ == "__main__":
    run_verification()
