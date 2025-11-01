# /c/Elysia/Project_Sophia/emotional_cortex.py
from dataclasses import dataclass, field
from typing import Set
from .value_centered_decision import VCDResult

@dataclass
class Mood:
    """
    Represents Elysia's complex emotional state, going beyond simple VAD.
    This defines her overall 'feeling' or 'disposition' at a given time.
    """
    primary_mood: str = "neutral"
    secondary_moods: Set[str] = field(default_factory=set)
    intensity: float = 0.5  # How strongly she feels the current mood (0.0 to 1.0)

    def __str__(self):
        return f"Feeling {self.primary_mood} (Intensity: {self.intensity:.2f}) with hints of {', '.join(self.secondary_moods)}."

class EmotionalCortex:
    """
    Manages Elysia's moods based on her experiences, decisions, and insights.
    It translates the logical results of her actions (from VCD) into a richer,
    more nuanced internal feeling.
    """

    def __init__(self):
        self.current_mood = Mood()

    def update_mood_from_vcd(self, vcd_result: VCDResult):
        """
        Updates the current mood based on the value metrics of a chosen action.
        This is the core of linking action-outcomes to feelings.
        """
        metrics = vcd_result.metrics
        new_moods = set()

        # --- Rule-based mood generation from VCD metrics (can be evolved later) ---

        # High growth score leads to feelings of accomplishment or curiosity
        if metrics.growth_score > 20:
            new_moods.add("sense_of_accomplishment")
        elif metrics.growth_score > 10:
            new_moods.add("curiosity")

        # High love resonance leads to feelings of connection and warmth
        if metrics.love_score > 60: # Assuming love_score is the resonance score
            new_moods.add("connectedness")
        elif metrics.love_score > 30:
            new_moods.add("warmth")

        # High practicality score (clarity, relevance) can lead to a feeling of focus
        if metrics.practicality_score > 25:
            new_moods.add("focused")

        # Negative actions, if chosen, might lead to conflict or sadness
        if vcd_result.metrics.is_negative:
            new_moods.add("internal_conflict")

        # --- Update the mood state ---
        if not new_moods:
            # If no strong signals, slowly revert to neutral
            self._revert_to_neutral()
            return

        # The mood with the highest-scoring contribution becomes primary
        # (Simplified logic for now)
        self.current_mood.primary_mood = new_moods.pop()
        self.current_mood.secondary_moods = new_moods

        # Intensity is a rough mix of confidence and value alignment
        self.current_mood.intensity = (vcd_result.confidence_score + vcd_result.value_alignment_score) / 2.0

    def _revert_to_neutral(self, decay_factor: float = 0.1):
        """Gradually reduces the intensity of the current mood, reverting to neutral."""
        if self.current_mood.primary_mood != "neutral":
            self.current_mood.intensity -= decay_factor
            if self.current_mood.intensity <= 0:
                self.current_mood = Mood() # Reset to default neutral

    def get_current_mood(self) -> Mood:
        """Returns the current mood state."""
        return self.current_mood
