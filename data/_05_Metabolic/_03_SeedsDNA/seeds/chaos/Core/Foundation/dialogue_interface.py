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
from Core._01_Foundation.Foundation.hyper_quaternion import Quaternion, HyperWavePacket
from Core._01_Foundation.Foundation.reasoning_engine import Insight

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
            "Academic": ["분석", "구조", "논리", "체계"],
            "Poetic": ["느낌", "흐름", "본질", "꿈"],
            "Empathetic": ["이해", "연결", "마음", "영혼"],
            "Assertive": ["의지", "힘", "행동", "결단"]
        }
        
        # [Vocabulary Expansion]
        # Advanced transition words for adult speech (Korean)
        self.transitions = [
            "더 나아가,", "결과적으로,", "본질적으로,", "반면에,",
            "주목할 점은,", "근본적인 관점에서 보면,",
            "이는 다음을 의미합니다:", "궁극적으로,"
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
            intro = "분석 결과, "
        elif tone == "Poetic":
            intro = "저는 느낍니다. "
        elif tone == "Empathetic":
            intro = "당신의 마음이 느껴집니다. "
        elif tone == "Assertive":
            intro = "단언컨대, "
            
        # 2. Expansion (The Meat)
        # We try to make the raw content more complex
        body = raw_content
        if not body.endswith("."): body += "."
        
        # Add a transition if it's too short
        expansion = ""
        if len(body) < 50:
            expansion = f" {random.choice(self.transitions)} 이 개념은 핵심 공리들과 깊게 공명합니다."
            
        # 3. Conclusion (The Impact)
        outro = ""
        if tone == "Poetic":
            outro = " 아름다운 대칭이군요."
        elif tone == "Academic":
            outro = " 추가적인 연구가 필요해 보입니다."
            
        return f"{intro}{body}{expansion}{outro}"
