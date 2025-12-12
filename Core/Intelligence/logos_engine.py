"""
Logos Engine (The Rhetorical Bridge)
====================================
"In the beginning was the Word, and the Word was with God."

The Logos Engine is responsible for the *Art of Speech*.
It takes raw, abstract insights from the ReasoningEngine and transforms them
into sophisticated, culturally rich, and metaphorically resonant language.

It acts as the "Harmonizer" between:
1. Logic (CausalNarrativeEngine) - The Skeleton
2. Metaphor (InternalUniverse) - The Flesh
3. Expression (DialogueInterface) - The Voice
"""

import logging
import random
from typing import List, Optional, Union
from Core.Foundation.internal_universe import InternalUniverse
from Core.Foundation.reasoning_engine import Insight
from Core.Foundation.Math.wave_tensor import WaveTensor

logger = logging.getLogger("LogosEngine")

class LogosEngine:
    def __init__(self):
        self.universe = InternalUniverse()
        logger.info("🗣️ Logos Engine Initialized: The Gift of Tongues")
        
        # Rhetorical Templates
        self.transition_matrix = {
            "thesis": ["근본적으로,", "우선,", "핵심을 짚어보자면,"],
            "antithesis": ["허나,", "그럼에도 불구하고,", "반면,", "하지만 깊이 들여다보면,"],
            "synthesis": ["결국,", "따라서,", "이러한 모순 속에서 저는 깨닫습니다.", "균형은 그 사이에 있습니다."]
        }

    def weave_speech(self, desire: str, insight: Union[Insight, str], context: List[str], wave: Optional[WaveTensor] = None) -> str:
        """
        The Master Function.
        Weaves Logic, Metaphor, and Narrative into a coherent response.
        Accepts raw string intuition or structured Insight objects.
        """
        # Handle simple string insights from Prism/Cognition
        content = insight.content if hasattr(insight, 'content') else str(insight)
        
        # 1. Analyze the Core Axis (Logic vs Emotion vs Value)
        # Use Wave properties if available for better axis detection
        axis = self._determine_axis(content, wave)
        
        # 2. Neural Binding: Check context for sensory anchors
        sensory_anchor = self._scan_for_sensory_anchor(context)
        
        # 3. Find a Root Metaphor (The Flesh)
        if sensory_anchor:
            logger.info(f"   🔗 Neural Binding: Locking onto sensory memory '{sensory_anchor}'")
            metaphor = f"마치 {sensory_anchor} 처럼,"
        else:
            # Enhanced Metaphor Mining using Wave Physics
            metaphor = self._mine_metaphor(axis, content, wave)
        
        # 4. Construct the Dialectic Argument (The Skeleton)
        argument = self._construct_dialectic(desire, content, axis)
        
        # 5. Narrative Polish (The Voice)
        response = f"{argument['thesis']} {metaphor} {argument['antithesis']} {argument['synthesis']}"
        
        return response

    def _scan_for_sensory_anchor(self, context: List[str]) -> Optional[str]:
        """
        Scans retrieved memories for sensory descriptions.
        """
        if not context:
            return None
            
        # We look for phrases injected by InternalUniverse or SensoryCortex
        # "scent of", "taste of", "feeling of Green High Pitch", etc.
        
        for memory in context:
            # Check for specific sensory markers we generated in Phase 31/32
            if "scent of" in memory:
                return self._extract_fragment(memory, "scent of")
            if "taste" in memory:
                return self._extract_fragment(memory, "taste")
            if "sounded like" in memory:
                return self._extract_fragment(memory, "sounded like")
            if "feeling of" in memory:
                # e.g., "A feeling of Green High Pitch" -> "그 초록빛 고음의 감각" (Transcreated)
                return "그 강렬한 감각" # Simplifying for naturalness, or extract detail
                
        return None

    def _extract_fragment(self, text: str, keyword: str) -> str:
        """Extracts the relevant sensory phrase."""
        try:
            # Simple extraction: take the keyword and the next 5 words
            parts = text.split(keyword)
            if len(parts) > 1:
                fragment = keyword + parts[1].split('.')[0]
                return fragment.strip()
        except:
            pass
        return text[:20]

    def _determine_axis(self, content: str, wave: Optional[WaveTensor] = None) -> str:
        """Determines if the thought is Logical, Emotional, or Ethical."""
        # Wave-based override
        if wave:
            # High Entropy/Dissonance -> Emotion/Chaos
            # Low Entropy/Harmonic -> Logic/Order
            if wave.total_energy > 4.0: return "Will" # High Energy
            
        text = content.lower()
        if any(w in text for w in ["feel", "sad", "joy", "pain", "love", "감정", "마음", "슬픔"]):
            return "Emotion"
        elif any(w in text for w in ["logic", "reason", "because", "structure", "논리", "이유", "구조"]):
            return "Logic"
        elif any(w in text for w in ["should", "must", "right", "wrong", "가치", "옳은", "도덕"]):
            return "Value"
        return "Balance"

    def _mine_metaphor(self, axis: str, content: str, wave: Optional[WaveTensor] = None) -> str:
        """
        Consults the Internal Universe to find a resonator (Fallback).
        Uses Wave Frequency to select metaphor register if available.
        """
        # Wave Frequency Mapping
        register = "Balance"
        if wave and wave.active_frequencies:
            dom_freq = wave.active_frequencies[0]
            if dom_freq < 200: register = "Earth" # Low/Deep
            elif dom_freq < 500: register = "Water" # Mid/Warm
            elif dom_freq < 800: register = "Air" # High/Clear
            else: register = "Fire" # Very High/Intense
        
        metaphors = {
            "Emotion": [
                "마치 겨울 바다의 파도처럼,", 
                "심장 깊은 곳에서 울리는 종소리처럼,",
                "비 온 뒤의 젖은 흙내음처럼,"
            ],
            # ... (Existing lists) ...
            "Earth": ["대지에 뿌리 내린 고목처럼,", "깊은 동굴의 울림처럼,", "단단한 바위처럼,"],
            "Water": ["유유히 흐르는 강물처럼,", "깊은 호수의 침묵처럼,", "새벽 이슬처럼,"],
            "Air": ["바람에 실려가는 구름처럼,", "맑은 하늘의 새처럼,", "투명한 유리처럼,"],
            "Fire": ["타오르는 혜성처럼,", "번개처럼 강렬하게,", "태양의 열기처럼,"],
            
            "Logic": [
                "정교하게 맞물린 시계태엽처럼,", 
                "차가운 대리석 조각처럼,",
                "별들의 궤도처럼 명확하게,"
            ],
            "Value": [
                "오래된 나무의 뿌리처럼,",
                "새벽의 첫 빛처럼,",
                "변하지 않는 북극성처럼,"
            ],
            "Will": [
                "타오르는 불꽃처럼,",
                "바위를 뚫는 물방울처럼,",
                "폭풍 속의 등대처럼,"
            ],
            "Balance": [
                "흐르는 강물처럼,",
                "고요한 호수처럼,",
                "바람에 흔들리는 갈대처럼,"
            ]
        }
        
        # Priority: Register (Physics) > Axis (Semantic)
        choices = metaphors.get(register, metaphors.get(axis, metaphors["Balance"]))
        chosen = random.choice(choices)
        return f"{chosen}"

    def _construct_dialectic(self, desire: str, raw_thought: str, axis: str) -> dict:
        """
        Splits the raw thought into a structured argument.
        """
        # Extract keywords from raw thought
        # Example raw_thought: "'Love' is painful but necessary."
        
        # Thesis: The initial assertion
        thesis_start = random.choice(self.transition_matrix["thesis"])
        thesis = f"{thesis_start} {raw_thought}"
        
        # Antithesis: The deeper nuance or contradiction (Paradox)
        antithesis_start = random.choice(self.transition_matrix["antithesis"])
        
        if axis == "Emotion":
            antithesis_content = "그 감정의 무게가 때로는 저를 짓누르기도 합니다."
        elif axis == "Logic":
            antithesis_content = "허나 논리만으로는 설명되지 않는 영역이 존재합니다."
        elif axis == "Value":
            antithesis_content = "하지만 절대적인 정답은 없을지도 모릅니다."
        else:
            antithesis_content = "보이는 것이 전부는 아닐 것입니다."
            
        antithesis = f"{antithesis_start} {antithesis_content}"
        
        # Synthesis: The resolution
        synthesis_start = random.choice(self.transition_matrix["synthesis"])
        synthesis = f"{synthesis_start} 저는 그 속에서 의미를 찾고 있습니다."
        
        return {
            "thesis": thesis,
            "antithesis": antithesis,
            "synthesis": synthesis
        }
