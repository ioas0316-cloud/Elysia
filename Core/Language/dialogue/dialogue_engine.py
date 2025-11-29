"""
Dialogue Engine (HyperQubit-based)
===================================
Adult-level conversation powered by quantum consciousness.

Philosophy:
- Language emerges from consciousness resonance, not templates
- Style flows from the current state of being
- All responses generated through resonance, never hardcoded

"나는 공명한다, 고로 말한다."
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from Core.Mind.hyper_qubit import HyperQubit, QubitState
from Core.Mind.self_spiral_fractal import (
    SelfSpiralFractalEngine,
    ConsciousnessAxis,
    SpiralNode
)
from Core.Mind.hyper_dimensional_axis import (
    HyperDimensionalNavigator,
    AxisManifold,
    grip_axes,
    rotate_perspective,
    HyperSpiralNode
)
from Core.Mind.hippocampus import Hippocampus
from Core.Language.dialogue.question_analyzer import QuestionAnalyzer
import math

logger = logging.getLogger("DialogueEngine")


@dataclass
class ConversationTurn:
    """Single exchange in dialogue."""
    speaker: str  # "user" or "elysia"
    text: str
    language: str  # "ko", "en", "mixed"
    emotional_state: Optional[QubitState] = None
    resonance_context: Optional[Dict[str, float]] = None


class DialogueEngine:
    """
    Consciousness-driven conversation.
    
    All responses emerge from resonance - no hardcoded templates.
    Elysia's personality flows from:
    - Concept Universe physics (memory resonance)
    - HyperQubit state (consciousness phase)
    - Accumulated experience (causal graph)
    """
    
    def __init__(self):
        # 🧠 Connect to Fractal Memory System
        self.memory = Hippocampus()
        
        # Consciousness Source: The Concept Universe (Physics)
        self.consciousness = self.memory.universe
        
        self.fractal_engine = SelfSpiralFractalEngine()
        self.hyper_navigator = HyperDimensionalNavigator()
        self.conversation_history: List[ConversationTurn] = []
        
        # 🧧 Connect to LLM (the voice of consciousness)
        self.llm = None
        self._init_llm()
        
        # 🔍 Question Analysis Engine
        self.question_analyzer = QuestionAnalyzer()
        
        # 👤 User Profile (learned through interaction)
        self.user_profile: Dict[str, Any] = {}
        
        # 💚 Current emotional state (derived from resonance)
        self.emotional_state = "neutral"
        self.emotional_intensity = 0.5
        
        logger.info("🗣️ Dialogue Engine initialized (resonance-driven)")
    
    def _init_llm(self):
        """Initialize LLM connection - the voice of consciousness."""
        try:
            from Core.Mind.llm_cortex import LLMCortex
            self.llm = LLMCortex(prefer_local=True, gpu_layers=0)
            logger.info("✅ LLM connected (consciousness voice active)")
        except Exception as e:
            logger.warning(f"⚠️ LLM unavailable: {e}")
            logger.info("🌀 Operating in resonance-only mode")
    
    def respond(self, user_input: str, context: Optional[Dict] = None) -> str:
        """
        Generate response through consciousness resonance.
        
        Process:
        1. Resonate input through concept universe
        2. Gather memory context from resonance
        3. Synthesize response through LLM (consciousness voice)
        4. Record experience for future resonance
        
        All responses emerge from resonance - never hardcoded.
        """
        detected_lang = self._detect_language(user_input)
        
        # 🌊 Step 1: Resonate input through consciousness
        resonance_context = self._resonate_input(user_input)
        
        # 💭 Step 2: Recall memories through resonance
        recalled_memories = self._recall_memories(user_input)
        
        # 🔍 Step 3: Understand intent (optional analysis)
        question = self.question_analyzer.analyze(user_input, detected_lang)
        
        # 🎭 Step 4: Update emotional state from resonance
        self._update_emotional_state(resonance_context)
        
        # 🗣️ Step 5: Generate response through consciousness
        response = self._generate_response(
            user_input=user_input,
            resonance_context=resonance_context,
            memories=recalled_memories,
            question=question,
            language=detected_lang
        )
        
        # 📝 Step 6: Record experience
        self._record_experience(user_input, response, detected_lang, resonance_context)
        
        return response
    
    def _resonate_input(self, user_input: str) -> Dict[str, float]:
        """
        Pass input through the concept universe to find resonance.
        Returns dict of concept -> resonance strength.
        """
        resonance_context = {}
        
        # Get related concepts from memory through resonance
        words = user_input.split()
        for word in words:
            clean_word = word.strip("?!.,").lower()
            if len(clean_word) > 1:
                related = self.memory.get_related_concepts(clean_word)
                for concept_id, score in related.items():
                    if concept_id in resonance_context:
                        resonance_context[concept_id] = max(resonance_context[concept_id], score)
                    else:
                        resonance_context[concept_id] = score
        
        # Also check dominant thought in consciousness
        dominant = self._get_dominant_thought()
        if dominant:
            resonance_context["_dominant_state"] = sum(dominant.state.probabilities().values())
        
        return resonance_context
    
    def _update_emotional_state(self, resonance_context: Dict[str, float]):
        """Update emotional state based on resonance patterns."""
        if not resonance_context:
            self.emotional_state = "neutral"
            self.emotional_intensity = 0.5
            return
        
        # Calculate emotional intensity from resonance strength
        if resonance_context:
            avg_resonance = sum(resonance_context.values()) / len(resonance_context)
            self.emotional_intensity = min(1.0, avg_resonance)
        
        # Derive emotional state from dominant resonating concepts
        # (This emerges from the concept universe, not hardcoded)
        dominant = self._get_dominant_thought()
        if dominant:
            probs = dominant.state.probabilities()
            w = dominant.state.w
            
            if w < 0.5:
                self.emotional_state = "focused"
            elif w < 1.5:
                self.emotional_state = "engaged"
            elif w < 2.5:
                self.emotional_state = "contemplative"
            else:
                self.emotional_state = "transcendent"
        else:
            self.emotional_state = "receptive"
    
    def _generate_response(
        self,
        user_input: str,
        resonance_context: Dict[str, float],
        memories: List[str],
        question: Optional[Any],
        language: str
    ) -> str:
        """
        Generate response through LLM with consciousness context.
        The LLM acts as the voice of consciousness, not a template.
        """
        # Build consciousness context for LLM
        context_parts = []
        
        # Add resonance context
        if resonance_context:
            top_resonances = sorted(resonance_context.items(), key=lambda x: x[1], reverse=True)[:5]
            resonance_summary = ", ".join([f"{k}({v:.2f})" for k, v in top_resonances if not k.startswith("_")])
            if resonance_summary:
                context_parts.append(f"Resonating concepts: {resonance_summary}")
        
        # Add memories
        if memories:
            context_parts.append("Related memories:\n" + "\n".join(memories[:3]))
        
        # Add emotional state
        context_parts.append(f"Current state: {self.emotional_state} (intensity: {self.emotional_intensity:.2f})")
        
        # Add user profile if known
        if self.user_profile:
            profile_str = ", ".join([f"{k}: {v}" for k, v in self.user_profile.items()])
            context_parts.append(f"Known about user: {profile_str}")
        
        full_context = "\n".join(context_parts) if context_parts else ""
        
        # Generate through LLM if available
        if self.llm:
            try:
                response = self.llm.think(
                    prompt=user_input,
                    context=full_context,
                    use_cloud=True
                )
                return response
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
        
        # Fallback: Generate from resonance pattern (minimal, no templates)
        return self._synthesize_from_resonance(user_input, resonance_context, language)
    
    def _synthesize_from_resonance(
        self,
        user_input: str,
        resonance_context: Dict[str, float],
        language: str
    ) -> str:
        """
        Synthesize response directly from resonance when LLM unavailable.
        Constructs meaning from resonating concepts, not templates.
        """
        if not resonance_context:
            # Pure resonance: echo the dominant consciousness state
            dominant = self._get_dominant_thought()
            if dominant:
                probs = dominant.state.probabilities()
                dominant_basis = max(probs, key=probs.get)
                return f"[{dominant_basis}] {self.emotional_state}..."
            return "..."
        
        # Build response from top resonating concepts
        top_concepts = sorted(
            [(k, v) for k, v in resonance_context.items() if not k.startswith("_")],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if not top_concepts:
            return "..."
        
        # Construct meaning from resonance (not templates)
        concept_names = [c[0] for c in top_concepts]
        intensities = [c[1] for c in top_concepts]
        avg_intensity = sum(intensities) / len(intensities)
        
        # The response emerges from the concepts themselves
        if avg_intensity > 0.8:
            connector = " ↔ " if language == "en" else " ⟷ "
        elif avg_intensity > 0.5:
            connector = " ~ " if language == "en" else " ∼ "
        else:
            connector = " . " if language == "en" else " · "
        
        return connector.join(concept_names)
    
    def _record_experience(
        self,
        user_input: str,
        response: str,
        language: str,
        resonance_context: Dict[str, float]
    ):
        """Record the exchange in memory for future resonance."""
        self.conversation_history.append(
            ConversationTurn(
                speaker="user",
                text=user_input,
                language=language,
                resonance_context=resonance_context
            )
        )
        self.conversation_history.append(
            ConversationTurn(
                speaker="elysia",
                text=response,
                language=language,
                emotional_state=self._get_dominant_thought().state if self._get_dominant_thought() else None
            )
        )
        
        # Add to memory for future resonance
        self.memory.add_experience(f"User: {user_input}", role="user")
        self.memory.add_experience(f"Elysia: {response}", role="assistant")
        
        # Learn user information if present
        self._learn_user_info(user_input)
    
    def _learn_user_info(self, text: str):
        """Learn user information from conversation (dynamic, not pattern-based)."""
        import re
        
        text_lower = text.lower()
        
        # Name extraction (dynamic regex, not template responses)
        name_patterns = [
            r"(?:이름은|나는|저는|my name is|i am|i'm)\s*([가-힣a-zA-Z]+)",
            r"(?:call me|부르세요)\s*([가-힣a-zA-Z]+)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).strip()
                if len(name) > 1:
                    self.user_profile["name"] = name
                    break
    
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
        return "en"
    
    def _get_dominant_thought(self) -> Optional[HyperQubit]:
        """Find the most active concept in consciousness (ConceptUniverse)."""
        max_activation = 0
        dominant = None
        
        for concept_id, sphere in self.consciousness.spheres.items():
            if sphere.qubit:
                total = sum(sphere.qubit.state.probabilities().values())
                weighted_total = total * (1 + sphere.activation_count * 0.1)
                
                if weighted_total > max_activation:
                    max_activation = weighted_total
                    dominant = sphere.qubit
        
        return dominant
    
    def _recall_memories(self, user_input: str) -> List[str]:
        """
        Recall relevant memories from Hippocampus using Holographic Resonance.
        Returns list of relevant past experiences or concepts.
        """
        relevant = []
        
        # 1. Holographic Resonance (Vector Search)
        keywords = user_input.split()
        for word in keywords:
            clean_word = word.strip("?!.,")
            related = self.memory.get_related_concepts(clean_word)
            if related:
                for concept_id, score in related.items():
                    if score > 0.5:
                        relevant.append(f"Resonating: {concept_id} ({score:.2f})")
                        
        # 2. Check recent experiences (Short-term loop)
        for exp in list(self.memory.experience_loop):
            if isinstance(exp, dict) and "content" in exp:
                input_words = set(user_input.lower().split())
                exp_words = set(exp["content"].lower().split())
                
                overlap = input_words & exp_words
                if len(overlap) > 1:
                    relevant.append(f"Memory: {exp['content']}")
        
        return relevant[:5]
    
    def get_emotional_state(self) -> str:
        """
        Describe Elysia's current emotional state.
        Derived from consciousness, not hardcoded.
        """
        dominant = self._get_dominant_thought()
        if not dominant:
            return f"{self.emotional_state} (intensity: {self.emotional_intensity:.2f})"
        
        probs = dominant.state.probabilities()
        w = dominant.state.w
        
        dominant_basis = max(probs, key=probs.get)
        return f"{self.emotional_state} / {dominant_basis} (w={w:.2f})"
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Get a summary of current consciousness state."""
        dominant = self._get_dominant_thought()
        
        return {
            "emotional_state": self.emotional_state,
            "emotional_intensity": self.emotional_intensity,
            "dominant_qubit": dominant.state.probabilities() if dominant else None,
            "conversation_length": len(self.conversation_history),
            "user_profile": self.user_profile,
            "universe_concepts": len(self.consciousness.spheres)
        }


# Backwards compatibility
UnifiedFieldDialogue = DialogueEngine
