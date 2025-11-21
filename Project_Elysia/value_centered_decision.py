import re
import math
import random
from typing import Optional, List

# Cross-project import for the standardized Thought data structure
from Project_Sophia.core.thought import Thought
from Project_Sophia.core.tensor_wave import Tensor3D
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.emotional_engine import EmotionalState
from tools.kg_manager import KGManager

class ValueCenteredDecision:
    """
    Value-Centered Decision (VCD) module, upgraded to process Thought objects.

    This module evaluates candidate Thoughts based on a holistic score including
    value alignment (resonance), confidence, energy, context, and novelty,
    dynamically adjusted by the current emotional state.
    """
    def __init__(self, kg_manager: KGManager, wave_mechanics: WaveMechanics, core_value: str = 'love'):
        """
        Initializes the VCD module with dependencies.
        """
        self.kg_manager = kg_manager
        self.wave_mechanics = wave_mechanics
        self.core_value = core_value.lower()

        # Base weights for the multi-faceted scoring function
        self.base_weights = {
            'resonance': 1.5,
            'confidence': 1.0,
            'energy': 0.8,
            'context': 0.5,
            'freshness': 1.0,
            'richness': 1.2  # New weight for harmonic complexity
        }

        self.recent_history = []  # To penalize repetition

    def _find_mentioned_entities(self, text: str) -> List[str]:
        """
        Finds entities from the KG mentioned in a message using a robust,
        length-sorted substring search to handle multi-word entities correctly.
        """
        mentioned_entities = []
        nodes = self.kg_manager.kg.get('nodes', [])
        if not nodes:
            return []

        node_ids = {node.get('id') for node in nodes if node.get('id')}

        # Sort by length descending to match longer names first (e.g., "black hole" before "hole")
        sorted_node_ids = sorted(list(node_ids), key=len, reverse=True)

        lower_text = text.lower()
        for entity_id in sorted_node_ids:
            if entity_id.lower() in lower_text:
                mentioned_entities.append(entity_id)

        return list(set(mentioned_entities))

    def _calculate_value_alignment(self, thought: Thought) -> float:
        """Calculates alignment with the core value using WaveMechanics and Tensor Resonance."""

        # 1. Tensor Resonance (Primary)
        # If the thought has a tensor state, compare it directly with the Core Value's tensor
        if thought.tensor_state:
            thought_tensor = Tensor3D.from_dict(thought.tensor_state)

            # Fetch Core Value Tensor (Love)
            core_node = self.kg_manager.get_node(self.core_value)
            if core_node and core_node.get('tensor_state'):
                core_tensor = Tensor3D.from_dict(core_node.get('tensor_state'))
            else:
                # Fallback: Assume 'Love' is high emotion/identity
                core_tensor = Tensor3D(0.2, 1.0, 0.9)

            # Calculate Alignment (Dot Product of normalized tensors)
            alignment = thought_tensor.normalize().dot(core_tensor.normalize())
            return max(0.0, alignment)

        # 2. KG Scalar Resonance (Secondary/Fallback)
        if not thought.content:
            return 0.0

        mentioned_entities = self._find_mentioned_entities(thought.content)
        if not mentioned_entities:
            return 0.25 # Neutral score if no specific entities are found

        total_resonance = 0.0
        for entity in mentioned_entities:
            resonance = self.wave_mechanics.get_resonance_between(entity, self.core_value)
            total_resonance += resonance

        avg_resonance = total_resonance / (len(mentioned_entities)**0.5) if mentioned_entities else 0.0
        return max(0.0, min(1.0, avg_resonance))

    def _calculate_context_fit(self, text: str, context: Optional[List[str]]) -> float:
        """Calculates how well the text fits the recent conversational context."""
        if not context:
            return 0.5
        ctx = ' '.join(context).lower()
        text_l = text.lower()
        tokens = set(re.findall(r"\w+", ctx))
        if not tokens:
            return 0.5
        found = sum(1 for t in tokens if t in text_l)
        return min(1.0, found / len(tokens))

    def _calculate_freshness(self, text: str) -> float:
        """Penalizes repetition of recently expressed thoughts."""
        penalty = 0.0
        history_len = len(self.recent_history)
        for i, prev in enumerate(reversed(self.recent_history[-10:])):
            if prev == text:
                # The more recent the repetition, the higher the penalty
                penalty += 0.9 * ((history_len - i) / history_len)
        return max(0.0, 1.0 - penalty)

    def _normalize_energy(self, energy: float) -> float:
        """Normalizes energy using a logarithmic scale to handle large variations."""
        # log1p is used to handle energy=0 gracefully (log(1)=0)
        return math.log1p(energy) / 10.0 # Scaled down to be in a similar range as other scores

    def _get_emotionally_adjusted_weights(self, emotional_state: Optional[EmotionalState]) -> dict:
        """Adjusts scoring weights based on the current primary emotion."""
        weights = self.base_weights.copy()
        if not emotional_state:
            return weights

        primary_emotion = emotional_state.primary_emotion

        if primary_emotion == 'joy':
            weights['energy'] *= 1.3 # Value dynamic, energetic thoughts more
            weights['confidence'] *= 0.8 # Be less cautious
            weights['richness'] *= 1.1 # Appreciate complexity
        elif primary_emotion == 'sadness':
            weights['resonance'] *= 1.2 # Seek more value-aligned, comforting thoughts
            weights['energy'] *= 0.7 # Less interested in high-energy thoughts
            weights['richness'] *= 1.5 # Deeply value complex/meaningful thoughts over simple ones
        elif primary_emotion == 'fear':
            weights['confidence'] *= 1.4 # Prioritize certainty and safety
            weights['energy'] *= 0.6
            weights['richness'] *= 0.5 # Prefer simple, clear solutions
        elif primary_emotion == 'calm':
            weights['confidence'] *= 1.2
            weights['freshness'] *= 0.8 # Less need for novelty

        return weights

    def score_thought(self, candidate: Thought, context: Optional[List[str]] = None, emotional_state: Optional[EmotionalState] = None) -> float:
        """Scores a single Thought object based on multiple criteria, adjusted for emotion."""

        weights = self._get_emotionally_adjusted_weights(emotional_state)

        # 1. Value Alignment (Soul) - Now Tensor-aware
        resonance_score = self._calculate_value_alignment(candidate)

        # 2. Intrinsic Quality (Mind)
        confidence_score = candidate.confidence
        energy_score = self._normalize_energy(candidate.energy)

        # 3. Richness (Texture/Complexity)
        richness_score = candidate.richness

        # 4. Conversational Flow (Environment)
        context_score = self._calculate_context_fit(candidate.content, context)
        freshness_score = self._calculate_freshness(candidate.content)

        # 5. Final Weighted Score
        final_score = (
            weights['resonance'] * resonance_score +
            weights['confidence'] * confidence_score +
            weights['energy'] * energy_score +
            weights['richness'] * richness_score +
            weights['context'] * context_score +
            weights['freshness'] * freshness_score
        )

        # --- Wisdom Bonus for thoughts from the 'Bone' ---
        if candidate.source == 'bone':
            wisdom_bonus = 0.5  # A significant bonus for foundational knowledge
            final_score += wisdom_bonus

        final_score += random.random() * 0.01  # Small random tie-breaker
        return final_score

    def select_thought(self, candidates: List[Thought], context: Optional[List[str]] = None, emotional_state: Optional[EmotionalState] = None, guiding_intention: Optional[str] = None) -> Optional[Thought]:
        """
        Selects the best Thought from a list of candidates by scoring each one,
        considering the current emotional state and a guiding intention.
        """
        if not candidates:
            return None

        scored_candidates = [
            (self.score_thought(c, context, emotional_state), c) for c in candidates
        ]

        # Sort by score in descending order
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_thought = scored_candidates[0][1]

        # Update history with the content of the chosen thought
        self.recent_history.append(best_thought.content)
        if len(self.recent_history) > 10: # Keep history trimmed
            self.recent_history.pop(0)

        return best_thought
