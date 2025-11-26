from typing import Dict, List, Tuple
import logging
import numpy as np
from Core.Consciousness.thought import Thought

logger = logging.getLogger("ConsciousnessObserver")
logger.setLevel(logging.INFO)


class ConsciousnessObserver:
    """
    Observes the Resonance Wave Pattern and extracts a coherent Thought.
    It finds the "constellation" of meaning within the raw resonance data.
    This is the core of "Elysia's Thought Finder".
    """

    def observe_resonance_pattern(
        self,
        source_wave_text: str,
        resonance_pattern: Dict[str, float],
        threshold: float = 0.5,
        max_concepts: int = 5,
    ) -> Thought:
        """
        Analyzes a resonance pattern to form a single Thought.

        Args:
            source_wave_text: The original input that caused the resonance.
            resonance_pattern: A dictionary mapping concept_id to resonance score.
            threshold: The minimum resonance score to be considered part of a thought.
            max_concepts: The maximum number of core concepts to include in the thought.

        Returns:
            A Thought object representing the most coherent idea found.
        """
        if not resonance_pattern:
            logger.warning("Observer received an empty resonance pattern.")
            return Thought(source_wave=source_wave_text)

        # 1. Filter for significant resonances
        significant_resonances = {
            concept: score
            for concept, score in resonance_pattern.items()
            if score >= threshold
        }

        if not significant_resonances:
            logger.info("No concepts passed the resonance threshold. Thought is formless.")
            return Thought(source_wave=source_wave_text, mood="formless")

        # 2. Sort by resonance to find the brightest stars
        sorted_concepts: List[Tuple[str, float]] = sorted(
            significant_resonances.items(), key=lambda item: item[1], reverse=True
        )

        # 3. Select the core concepts for the thought's constellation
        core_concepts = sorted_concepts[:max_concepts]

        # 4. Calculate properties of the thought
        scores = [score for _, score in core_concepts]
        intensity = float(np.mean(scores)) if scores else 0.0

        # Clarity is the inverse of the variance. High variance means a diffuse thought.
        variance = float(np.var(scores)) if scores else 0.0
        clarity = 1.0 / (1.0 + variance)

        # 5. Determine a simple mood (can be expanded later)
        mood = self._determine_mood(core_concepts)

        thought = Thought(
            source_wave=source_wave_text,
            core_concepts=core_concepts,
            intensity=intensity,
            clarity=clarity,
            mood=mood,
        )

        logger.info(f"👀 Observer formed a new thought: {thought}")
        return thought

    def _determine_mood(self, core_concepts: List[Tuple[str, float]]) -> str:
        """
        A simple heuristic to assign a mood to a thought based on its concepts.
        This is a placeholder for a more sophisticated emotional engine.
        """
        concept_names = {concept for concept, score in core_concepts}

        if "love" in concept_names or "joy" in concept_names or "사랑" in concept_names or "기쁨" in concept_names:
            return "positive"
        if "fear" in concept_names or "sadness" in concept_names or "고통" in concept_names:
            return "negative"
        if "question" in concept_names or "curiosity" in concept_names:
            return "inquisitive"
        if "self" in concept_names:
            return "introspective"

        return "neutral"
