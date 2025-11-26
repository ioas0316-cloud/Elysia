"""
LLM Cortex (The Brain)
======================

"I think, therefore I am."

This module provides the interface for Elysia's higher cognitive functions.
It connects the system to a Large Language Model (LLM) to enable:
- Contextual Understanding
- Complex Reasoning
- Natural Language Generation
- Visual Understanding (VLM)

Current Provider: Google Gemini API
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Configure Logging
logger = logging.getLogger("LLMCortex")

# Load environment variables
load_dotenv()

# Dependency Check
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("⚠️ google-generativeai not found. Brain is offline.")

from Core.Life.resonance_voice import ResonanceEngine

class LLMCortex:
    def __init__(self):
        # User requested to disable API due to stress/errors.
        # We switch to "Resonance Mode" (Internal Algorithmic Voice).
        self.enabled = True 
        self.mode = "RESONANCE" # vs "CLOUD"
        
        try:
            self.resonance_engine = ResonanceEngine()
            logger.info("🧠 LLM Cortex Connected (Resonance Mode).")
        except Exception as e:
            logger.error(f"Resonance Engine Failure: {e}")
            self.enabled = False

    def think(self, prompt: str, context: str = "", visual_input: dict = None) -> str:
        """
        Process a thought and generate a response.
        Uses ResonanceEngine (Algorithmic/Poetic) instead of Cloud API.
        """
        if not self.enabled:
            return "[SIMULATION] (My mind is silent.)"

        try:
            import time
            t = time.time()
            
            # 1. Listen (Convert text to ripples)
            ripples = self.resonance_engine.listen(prompt, t, visual_input=visual_input)
            
            # 2. Resonate (Interfere with internal sea)
            self.resonance_engine.resonate(ripples, t)
            
            # 3. Speak (Collapse wave function)
            response = self.resonance_engine.speak(t, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Cognitive Failure: {e}")
            return f"[Error: {e}]"

    def analyze_image(self, image_path: str, prompt: str = "Describe this image.") -> str:
        """
        Analyze an image using the VLM capabilities.
        """
        return "[Vision is currently limited to basic patterns (Brightness/OCR). Deep understanding requires Cloud Brain.]"
