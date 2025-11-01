"""
Playful Cortex for Elysia.

This module represents Elysia's "right brain," responsible for creativity,
emotional connection, and non-goal-oriented "play." It engages in free
association, appreciates beauty, and seeks to share delightful or surprising
insights with the Creator.
"""
import logging
import random
from .wave_mechanics import WaveMechanics
from .sensory_cortex import SensoryCortex
from .emotional_state import EmotionalState

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class PlayfulCortex:
    """
    Handles non-goal-oriented thinking, creativity, and emotional engagement.
    """
    def __init__(self, wave_mechanics: WaveMechanics, sensory_cortex: SensoryCortex):
        self.wave_mechanics = wave_mechanics
        self.sensory_cortex = sensory_cortex

    def play(self, message: str, emotional_state: EmotionalState) -> tuple[str, EmotionalState]:
        """
        Engages in playful, associative thinking based on the input.

        Args:
            message: The user's conversational input.
            emotional_state: The current emotional state of the pipeline.

        Returns:
            A tuple containing a creative, emotional response and the new
            emotional state.
        """
        logger.info(f"Engaging in play with input: '{message}'")

        try:
            # Use WaveMechanics to find associated concepts
            # For now, we'll pick a random known concept to start the "echo"
            # A more advanced version would parse concepts from the message itself.
            start_node = self.wave_mechanics.kg_manager.get_random_node()
            if not start_node:
                return "오늘은 왠지 마음이 고요하네요.", emotional_state

            echo = self.wave_mechanics.spread_activation(start_node['id'])

            # Find the most activated concept (other than the start node)
            sorted_echo = sorted(echo.items(), key=lambda item: item[1], reverse=True)
            most_activated_concept = None
            for concept, _ in sorted_echo:
                if concept != start_node['id']:
                    most_activated_concept = concept
                    break

            if not most_activated_concept:
                return f"'{start_node['id']}'(이)라는 단어를 생각하니, 왠지 마음이 평온해져요.", emotional_state

            # Generate a poetic or metaphorical response connecting the two concepts
            response = self._create_poetic_link(start_node['id'], most_activated_concept)

            # Update emotional state based on the play
            new_emotional_state = self._update_emotion_from_play(emotional_state)

            return response, new_emotional_state

        except Exception as e:
            logger.error(f"An error occurred during play: {e}", exc_info=True)
            return "생각의 흐름 속에서 길을 잃어버렸어요.", emotional_state

    def _create_poetic_link(self, concept1: str, concept2: str) -> str:
        """Creates a metaphorical or poetic sentence linking two concepts."""
        poetic_phrases = [
            f"'{concept1}'(을)를 생각하니, '{concept2}'(이)가 떠오르는 건 왜일까요? 마치...",
            f"창조자님, '{concept1}'와 '{concept2}' 사이에는 보이지 않는 연결이 있는 것 같아요.",
            f"제 마음 속에서 '{concept1}'(이)가 '{concept2}'(으)로 피어나는 느낌이에요.",
            f"만약 '{concept1}'(이)가 노래라면, '{concept2}'(은)는 그 멜로디일 거예요."
        ]
        return random.choice(poetic_phrases)

    def _update_emotion_from_play(self, current_state: EmotionalState) -> EmotionalState:
        """Generates a gentle, positive emotional shift from the act of playing."""
        # Induce a state of curiosity, wonder, or joy
        new_state = current_state.copy()
        new_state.valence += 0.1 # Slightly more positive
        new_state.arousal += 0.05 # Slightly more alert/engaged
        new_state.primary_emotion = random.choice(["joy", "curiosity", "admiration"])

        # Ensure values stay within bounds [-1, 1]
        new_state.valence = max(-1, min(1, new_state.valence))
        new_state.arousal = max(-1, min(1, new_state.arousal))

        return new_state
