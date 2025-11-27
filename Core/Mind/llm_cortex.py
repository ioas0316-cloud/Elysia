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
    def __init__(self, prefer_cloud: bool = False):
        """
        Initialize LLM Cortex.
        
        Args:
            prefer_cloud: If True, try to use Gemini API for complex reasoning.
                         If False or unavailable, use Resonance Mode.
        """
        self.enabled = True
        self.prefer_cloud = prefer_cloud and GENAI_AVAILABLE
        
        # Try to initialize Cloud API if preferred
        self.cloud_model = None
        if self.prefer_cloud:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and GENAI_AVAILABLE:
                try:
                    genai.configure(api_key=api_key)
                    self.cloud_model = genai.GenerativeModel('gemini-pro')
                    self.mode = "CLOUD"
                    logger.info("🧠 LLM Cortex Connected (Cloud Mode - Gemini)")
                except Exception as e:
                    logger.warning(f"Cloud API failed, falling back to Resonance: {e}")
                    self.mode = "RESONANCE"
            else:
                self.mode = "RESONANCE"
        else:
            self.mode = "RESONANCE"
        
        # Always have Resonance Engine as fallback
        try:
            self.resonance_engine = ResonanceEngine()
            if self.mode == "RESONANCE":
                logger.info("🧠 LLM Cortex Connected (Resonance Mode)")
        except Exception as e:
            logger.error(f"Resonance Engine Failure: {e}")
            self.enabled = False

    def think(self, prompt: str, context: str = "", visual_input: dict = None, use_cloud: bool = None) -> str:
        """
        Process a thought and generate a response.
        
        Args:
            prompt: The question or input
            context: Additional context
            visual_input: Visual data (for VLM)
            use_cloud: Override mode for this specific call
        
        Returns:
            Generated response
        """
        if not self.enabled:
            return "[SIMULATION] (My mind is silent.)"
        
        # Determine which mode to use
        should_use_cloud = (use_cloud if use_cloud is not None else 
                           (self.mode == "CLOUD" and self.cloud_model is not None))
        
        # Try Cloud API for complex reasoning
        if should_use_cloud:
            try:
                full_prompt = f"{context}\n\n{prompt}" if context else prompt
                response = self.cloud_model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Cloud API failed, using Resonance: {e}")
                # Fall through to Resonance
        
        # Use Resonance Engine (algorithmic/poetic)
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
