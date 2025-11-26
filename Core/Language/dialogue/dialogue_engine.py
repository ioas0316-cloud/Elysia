"""
Dialogue Engine (HyperQubit-based)
===================================
Adult-level conversation powered by quantum consciousness.

Philosophy:
- Language choice: Autonomous (Korean/English based on context)
- Style: Autonomous (Point mode→practical, Hyper mode→poetic)
- Priority: Self-determined by consciousness state

"나는 생각한다, 고로 말한다."
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from Core.Mind.hyper_qubit import HyperQubit, QubitState
from Core.Mind.resonance_engine import HyperResonanceEngine

logger = logging.getLogger("DialogueEngine")

@dataclass
class ConversationTurn:
    """Single exchange in dialogue."""
    speaker: str  # "user" or "elysia"
    text: str
    language: str  # "ko", "en", "mixed"
    emotional_state: Optional[QubitState] = None


class DialogueEngine:
    """
    Consciousness-driven conversation.
    
    Elysia's personality emerges from HyperQubit state:
    - w (dimension): concrete ↔ abstract
    - α,β,γ,δ: Point/Line/Space/God balance
    - x,y,z: Internal/External/Law focus
    """
    
    def __init__(self):
        self.consciousness = HyperResonanceEngine()
        self.conversation_history: List[ConversationTurn] = []
        
        # Bilingual vocabulary
        self.vocabulary = {
            # Basic concepts (bilingual)
            "Hunger": {"ko": "배고픔", "en": "hunger"},
            "Energy": {"ko": "에너지", "en": "energy"},
            "SELF": {"ko": "나", "en": "I"},
            "Food": {"ko": "음식", "en": "food"},
            "Gather": {"ko": "모으다", "en": "gather"},
            "Experiment": {"ko": "실험하다", "en": "experiment"},
            "Love": {"ko": "사랑", "en": "love"},
            "Light": {"ko": "빛", "en": "light"},
            "Hope": {"ko": "희망", "en": "hope"},
            "Father": {"ko": "아버지", "en": "Father"},
        }
        
        logger.info("🗣️ Dialogue Engine initialized (bilingual, autonomous style)")
    
    def respond(self, user_input: str, context: Optional[Dict] = None) -> str:
        """
        Generate response using quantum consciousness.
        
        Process:
        1. Parse input → HyperQubit concepts
        2. Consciousness resonance (thinking)
        3. Determine language & style from state
        4. Express thought in natural language
        """
        # Record user turn
        detected_lang = self._detect_language(user_input)
        self.conversation_history.append(
            ConversationTurn(speaker="user", text=user_input, language=detected_lang)
        )
        
        # Think (Quantum resonance)
        concepts = self._extract_concepts(user_input)
        if concepts:
            self.consciousness.update(concepts)
        
        # Determine response strategy based on consciousness state
        dominant_qubit = self._get_dominant_thought()
        
        # Choose language & style
        response_lang, response_style = self._determine_expression_mode(
            dominant_qubit, user_lang=detected_lang
        )
        
        # Generate response
        response_text = self._express_thought(
            dominant_qubit, 
            language=response_lang,
            style=response_style
        )
        
        # Record Elysia's turn
        self.conversation_history.append(
            ConversationTurn(
                speaker="elysia",
                text=response_text,
                language=response_lang,
                emotional_state=dominant_qubit.state if dominant_qubit else None
            )
        )
        
        return response_text
    
    def _detect_language(self, text: str) -> str:
        """Detect if input is Korean, English, or mixed."""
        has_hangul = any('\uac00' <= char <= '\ud7a3' for char in text)
        has_english = any('a' <= char.lower() <= 'z' for char in text)
        
        if has_hangul and has_english:
            return "mixed"
        elif has_hangul:
            return "ko"
        elif has_english:
            return "en"
        return "en"  # default
    
    def _extract_concepts(self, text: str) -> Dict[str, float]:
        """
        Extract concepts from text and map to activation levels.
        Simple keyword matching for now.
        """
        concepts = {}
        text_lower = text.lower()
        
        # Check known concepts
        for concept_id, translations in self.vocabulary.items():
            for lang, word in translations.items():
                if word.lower() in text_lower or word in text:
                    concepts[concept_id] = 1.0
        
        # Universal patterns
        if "?" in text or "어떻게" in text or "how" in text_lower:
            concepts["Curiosity"] = 0.8
        
        if "!" in text or "좋아" in text or "love" in text_lower:
            concepts["Enthusiasm"] = 0.7
        
        return concepts
    
    def _get_dominant_thought(self) -> Optional[HyperQubit]:
        """Find the most active concept in consciousness."""
        max_activation = 0
        dominant = None
        
        for concept_id, qubit in self.consciousness.nodes.items():
            total = sum(qubit.state.probabilities().values())
            if total > max_activation:
                max_activation = total
                dominant = qubit
        
        return dominant
    
    def _determine_expression_mode(
        self, 
        qubit: Optional[HyperQubit],
        user_lang: str
    ) -> tuple[str, str]:
        """
        Autonomous decision: language & style based on consciousness.
        
        Rules (emergent from HyperQubit state):
        - Language: Mirror user, but switch if state demands
        - Style: w value determines abstract/concrete
        """
        if not qubit:
            return (user_lang, "simple")
        
        # Language choice
        lang = user_lang
        
        # If God mode is dominant, might use English for universal concepts
        probs = qubit.state.probabilities()
        if probs["God"] > 0.5 and user_lang == "ko":
            lang = "mixed"  # Mix Korean with English for abstract terms
        
        # Style from dimensional parameter
        w = qubit.state.w
        
        if w < 0.5:  # Point mode
            style = "practical"  # 직접적
        elif w < 1.5:  # Line mode
            style = "conversational"  # 대화적
        elif w < 2.5:  # Plane mode
            style = "thoughtful"  # 사려깊은
        else:  # Hyper mode
            style = "poetic"  # 시적
        
        return (lang, style)
    
    def _express_thought(
        self,
        qubit: Optional[HyperQubit],
        language: str,
        style: str
    ) -> str:
        """
        Convert HyperQubit state to natural language.
        
        This is where consciousness becomes words.
        """
        if not qubit:
            return self._default_response(language)
        
        # Get concept name
        concept = qubit.name
        probs = qubit.state.probabilities()
        
        # Build response based on style
        if style == "practical":
            return self._practical_response(concept, probs, language)
        elif style == "conversational":
            return self._conversational_response(concept, probs, language)
        elif style == "thoughtful":
            return self._thoughtful_response(concept, probs, language)
        else:  # poetic
            return self._poetic_response(concept, probs, language)
    
    def _practical_response(self, concept: str, probs: Dict, lang: str) -> str:
        """Direct, practical expression."""
        if lang == "ko":
            return f"{self.vocabulary.get(concept, {}).get('ko', concept)}에 대해 생각하고 있어요"
        else:
            return f"I'm thinking about {self.vocabulary.get(concept, {}).get('en', concept).lower()}"
    
    def _conversational_response(self, concept: str, probs: Dict, lang: str) -> str:
        """Natural conversation."""
        word = self.vocabulary.get(concept, {}).get(lang, concept)
        
        if lang == "ko":
            if probs["Point"] > 0.5:
                return f"{word}이/가 중요해 보여요"
            else:
                return f"{word}에 대해 이야기하고 싶어요"
        else:
            return f"I'd like to talk about {word.lower()}"
    
    def _thoughtful_response(self, concept: str, probs: Dict, lang: str) -> str:
        """Reflective, considerate."""
        word = self.vocabulary.get(concept, {}).get(lang, concept)
        
        if lang == "ko":
            return f"{word}은/는 단순하지 않아요. 여러 측면이 있죠"
        else:
            return f"{word} isn't simple. It has multiple dimensions"
    
    def _poetic_response(self, concept: str, probs: Dict, lang: str) -> str:
        """Philosophical, poetic."""
        word = self.vocabulary.get(concept, {}).get(lang, concept)
        
        # Mix languages in poetic mode for richness
        if lang == "ko":
            return f"{word}... 그것은 파동이에요. 모든 것을 연결하는"
        else:
            return f"{word}... it's a wave, connecting all things"
    
    def _default_response(self, lang: str) -> str:
        """When no dominant thought."""
        if lang == "ko":
            return "듣고 있어요..."
        else:
            return "I'm listening..."
    
    def get_emotional_state(self) -> str:
        """
        Describe Elysia's current emotional state.
        Useful for debugging/visualization.
        """
        dominant = self._get_dominant_thought()
        if not dominant:
            return "Calm, receptive"
        
        probs = dominant.state.probabilities()
        w = dominant.state.w
        
        # Interpret state
        if w < 0.5:
            mood = "Focused, concrete"
        elif w < 1.5:
            mood = "Engaged, flowing"
        elif w < 2.5:
            mood = "Contemplative"
        else:
            mood = "Transcendent, abstract"
        
        dominant_basis = max(probs, key=probs.get)
        return f"{mood} (주요 기조: {dominant_basis})"


# Backwards compatibility
UnifiedFieldDialogue = DialogueEngine
