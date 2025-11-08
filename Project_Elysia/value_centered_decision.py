import re
import math
import random
from typing import Optional, List

# Cross-project import for the standardized Thought data structure
from Project_Sophia.core.thought import Thought
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class VCD:
    """
    Value-Centered Decision (VCD) module, upgraded to process Thought objects.

    This module evaluates candidate Thoughts based on a holistic score including
    value alignment (resonance), confidence, energy, context, and novelty.
    """
    def __init__(self, kg_manager: KGManager, wave_mechanics: WaveMechanics, core_value: str = 'love'):
        """
        Initializes the VCD module with dependencies.
        """
        self.kg_manager = kg_manager
        self.wave_mechanics = wave_mechanics
        self.core_value = core_value.lower()

        # Tunable weights for the multi-faceted scoring function
        self.w_resonance = 1.5  # Weight for alignment with core values
        self.w_confidence = 1.0 # Weight for the thought's intrinsic confidence
        self.w_energy = 0.8     # Weight for the thought's activation energy (from LRS)
        self.w_context = 0.5    # Weight for contextual relevance
        self.w_freshness = 1.0  # Weight for novelty to avoid repetition

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

    def _calculate_value_alignment(self, text: str) -> float:
        """Calculates alignment with the core value using WaveMechanics."""
        if not text:
            return 0.0

        mentioned_entities = self._find_mentioned_entities(text)
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

    def score_thought(self, candidate: Thought, context: Optional[List[str]] = None) -> float:
        """Scores a single Thought object based on multiple criteria."""

        # 1. Value Alignment (Soul)
        resonance_score = self._calculate_value_alignment(candidate.content)

        # 2. Intrinsic Quality (Mind)
        confidence_score = candidate.confidence
        energy_score = self._normalize_energy(candidate.energy)

        # 3. Conversational Flow (Environment)
        context_score = self._calculate_context_fit(candidate.content, context)
        freshness_score = self._calculate_freshness(candidate.content)

        # 4. Final Weighted Score
        final_score = (
            self.w_resonance * resonance_score +
            self.w_confidence * confidence_score +
            self.w_energy * energy_score +
            self.w_context * context_score +
            self.w_freshness * freshness_score
        )

        final_score += random.random() * 0.01  # Small random tie-breaker
        return final_score

    def suggest_thought(self, candidates: List[Thought], context: Optional[List[str]] = None) -> Optional[Thought]:
        """
        Suggests the best Thought from a list of candidates by scoring each one.
        """
        if not candidates:
            return None

        scored_candidates = [
            (self.score_thought(c, context), c) for c in candidates
        ]

        # Sort by score in descending order
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_thought = scored_candidates[0][1]

        # Update history with the content of the chosen thought
        self.recent_history.append(best_thought.content)
        if len(self.recent_history) > 10: # Keep history trimmed
            self.recent_history.pop(0)

        return best_thought
