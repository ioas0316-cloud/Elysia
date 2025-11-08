import re
import random
from typing import Optional, List

from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class VCD:
    """
    Value-Centered Decision (VCD) module, now driven by WaveMechanics.

    This module evaluates candidate actions (strings) based on their conceptual
    resonance with a core value (e.g., 'love') within the knowledge graph.
    """
    def __init__(self, kg_manager: KGManager, wave_mechanics: WaveMechanics, core_value: str = 'love'):
        """
        Initializes the VCD module with dependencies.

        Args:
            kg_manager: An instance of KGManager to interact with the knowledge graph.
            wave_mechanics: An instance of WaveMechanics to calculate conceptual resonance.
            core_value: The central KG node ID to align with (defaults to 'love').
        """
        self.kg_manager = kg_manager
        self.wave_mechanics = wave_mechanics
        self.core_value = core_value.lower()

        # Tunable weights for scoring components
        self.alpha = 1.0  # value alignment (resonance)
        self.beta = 0.7   # context fit
        self.gamma = 1.5  # freshness / novelty

        self.recent_history = []  # To penalize repetition

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """A simple method to find mentioned KG entities in a text."""
        # This is a simplified version. A more robust implementation might use
        # the logic from CognitionPipeline._find_mentioned_entities.
        mentioned_entities = []
        node_ids = {node.get('id') for node in self.kg_manager.kg.get('nodes', [])}

        # Simple substring search for now
        for entity_id in node_ids:
            if entity_id and re.search(re.escape(entity_id), text, re.IGNORECASE):
                mentioned_entities.append(entity_id)

        return list(set(mentioned_entities))

    def value_alignment(self, text: str) -> float:
        """
        Calculates alignment with the core value using WaveMechanics.
        It measures the average resonance of mentioned entities with 'love'.
        """
        if not text:
            return 0.0

        mentioned_entities = self._extract_entities_from_text(text)
        if not mentioned_entities:
            # If no specific entities are found, we can't measure resonance.
            # Return a neutral score.
            return 0.25

        total_resonance = 0.0
        for entity in mentioned_entities:
            # Calculate the resonance between the found entity and the core value.
            resonance = self.wave_mechanics.get_resonance_between(entity, self.core_value)
            total_resonance += resonance

        # Average the resonance score across all mentioned entities.
        # We normalize by sqrt to give a slight boost to multi-concept sentences.
        avg_resonance = total_resonance / (len(mentioned_entities)**0.5) if mentioned_entities else 0.0

        # Clamp the value to a [0.0, 1.0] range for consistent scoring.
        return max(0.0, min(1.0, avg_resonance))

    def context_fit(self, text: str, context: Optional[List[str]]) -> float:
        """Naive context fit: fraction of context tokens present in text."""
        if not context:
            return 0.5
        ctx = ' '.join(context).lower()
        text_l = text.lower()
        tokens = set(re.findall(r"\w+", ctx))
        if not tokens:
            return 0.5
        found = sum(1 for t in tokens if t in text_l)
        return min(1.0, found / len(tokens))

    def freshness(self, text: str) -> float:
        """Penalize repetition: if recently used, lower score."""
        penalty = 0.0
        history_len = len(self.recent_history[-10:])
        for i, prev in enumerate(reversed(self.recent_history[-10:])):
            if prev == text:
                penalty += 0.9 * (history_len - i)
        return max(0.0, 1.0 - penalty)

    def score_action(self, candidate: str, context: Optional[List[str]] = None) -> float:
        """Scores a single candidate action."""
        va = self.value_alignment(candidate)
        cf = self.context_fit(candidate, context)
        fr = self.freshness(candidate)

        score = self.alpha * va + self.beta * cf + self.gamma * fr
        score += random.random() * 0.01  # Small random tie-breaker
        return score

    def suggest_action(self, candidates: List[str], context: Optional[List[str]] = None) -> Optional[str]:
        """Suggests the best action from a list of candidates."""
        if not candidates:
            return None

        scored_candidates = [(self.score_action(c, context), c) for c in candidates]
        scored_candidates.sort(reverse=True)

        best_action = scored_candidates[0][1]
        self.recent_history.append(best_action)

        return best_action
