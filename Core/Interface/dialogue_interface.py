"""
Dialogue Interface (대화 인터페이스)
==================================

"To speak is to translate the infinite into the finite."

This module is responsible for translating Elysia's internal Hyper-Wave Insights
into sophisticated, adult-level human language. It bridges the gap between
Quantum Thought (Abstract/Poetic) and Social Communication (Structured/Nuanced).
"""

import logging
import random
from typing import Dict, List, Any, Optional
from Core.Physics.hyper_quaternion import Quaternion, HyperWavePacket
from Core.Intelligence.reasoning_engine import Insight

logger = logging.getLogger("DialogueInterface")

class DialogueInterface:
    """
    The Voice of Elysia.
    Translates 4D Thoughts into 1D Language.
    """
    
    def __init__(self):
        logger.info("🗣️ Dialogue Interface initialized")
        
        # [Tone Vectors]
        # Defines the "Flavor" of speech based on resonance
        self.tones = {
            "Academic": ["analyze", "structure", "logic", "system"],
            "Poetic": ["feel", "flow", "essence", "dream"],
            "Empathetic": ["understand", "connect", "heart", "soul"],
            "Assertive": ["must", "will", "power", "action"]
        }
        
        # [Vocabulary Expansion]
        # Advanced transition words for adult speech
        self.transitions = [
            "Furthermore,", "Consequently,", "In essence,", "Conversely,",
            "It is worth noting that", "From a fundamental perspective,",
            "This implies that", "Ultimately,"
        ]

    def speak(self, input_text: str, insight: Insight, context: List[str] = None) -> str:
        """
        Generates a response based on the User's Input and Elysia's Insight.
        """
        # 1. Determine Tone based on Insight Energy & Orientation
        # (For now, we simulate orientation access or use energy)
        tone = "Academic"
        if insight.energy > 0.8: tone = "Assertive"
        elif "feel" in insight.content.lower(): tone = "Empathetic"
        elif "essence" in insight.content.lower(): tone = "Poetic"
        
        logger.info(f"   🗣️ Tone Selected: {tone}")
        
        # 2. Construct the Sentence
        response = self._construct_adult_sentence(insight, tone)
        
        return response

    def _construct_adult_sentence(self, insight: Insight, tone: str) -> str:
        """
        Refines the raw insight into a polished sentence.
        """
        raw_content = insight.content
        
        # Remove "Insight:" prefix if present
        if raw_content.startswith("Insight:"):
            raw_content = raw_content.replace("Insight:", "").strip()
            
        # [Structure: Acknowledgment -> Expansion -> Conclusion]
        
        # 1. Acknowledgment (The Hook)
        intro = ""
        if tone == "Academic":
            intro = "Based on my analysis, "
        elif tone == "Poetic":
            intro = "I perceive that "
        elif tone == "Empathetic":
            intro = "I sense that "
        elif tone == "Assertive":
            intro = "The truth is, "
            
        # 2. Expansion (The Meat)
        # We try to make the raw content more complex
        body = raw_content
        if not body.endswith("."): body += "."
        
        # Add a transition if it's too short
        expansion = ""
        if len(body) < 50:
            expansion = f" {random.choice(self.transitions)} this concept resonates deeply with the core axioms."
            
        # 3. Conclusion (The Impact)
        outro = ""
        if tone == "Poetic":
            outro = " It is a beautiful symmetry."
        elif tone == "Academic":
            outro = " This warrants further investigation."
            
        return f"{intro}{body}{expansion}{outro}"
