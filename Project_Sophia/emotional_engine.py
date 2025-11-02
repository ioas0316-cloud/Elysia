from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class EmotionalState:
    """Represents Elysia's current emotional state."""
    valence: float  # Pleasure: -1 (negative) to 1 (positive)
    arousal: float  # Activation: 0 (calm) to 1 (excited)
    dominance: float # Control: -1 (submissive) to 1 (dominant)
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = field(default_factory=list)

class EmotionalEngine:
    """
    Manages the dynamics of Elysia's emotional state, including transitions
    and the influence of events.
    """

    def __init__(self, initial_state: EmotionalState = None):
        if initial_state is None:
            self.current_state = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0)
        else:
            self.current_state = initial_state

    def process_event(self, event_emotion: EmotionalState, intensity: float = 0.5):
        """
        Updates the current emotional state based on an external event.
        This is a more sophisticated transition than a simple linear interpolation.

        Args:
            event_emotion: The emotional quality of the event.
            intensity: How strongly the event affects the current state (0 to 1).
        """
        # Decay factor: previous emotions linger
        decay = 1.0 - intensity

        # Update VAD values
        self.current_state.valence = (self.current_state.valence * decay) + (event_emotion.valence * intensity)
        self.current_state.arousal = (self.current_state.arousal * decay) + (event_emotion.arousal * intensity)
        self.current_state.dominance = (self.current_state.dominance * decay) + (event_emotion.dominance * intensity)

        # Update primary and secondary emotions
        # A simple model for now: the event's emotion becomes primary if it's intense enough
        if intensity > 0.6:
            if self.current_state.primary_emotion != event_emotion.primary_emotion:
                # The old primary becomes secondary
                if self.current_state.primary_emotion not in self.current_state.secondary_emotions:
                    self.current_state.secondary_emotions.insert(0, self.current_state.primary_emotion)
                self.current_state.primary_emotion = event_emotion.primary_emotion
        
        # Add the event's secondary emotions to the current state's secondary list
        for emo in event_emotion.secondary_emotions:
            if emo not in self.current_state.secondary_emotions and emo != self.current_state.primary_emotion:
                self.current_state.secondary_emotions.append(emo)

        # Keep the list of secondary emotions from growing too long
        self.current_state.secondary_emotions = self.current_state.secondary_emotions[:3]

        # Clamp values to their ranges
        self.current_state.valence = max(-1.0, min(1.0, self.current_state.valence))
        self.current_state.arousal = max(0.0, min(1.0, self.current_state.arousal))
        self.current_state.dominance = max(-1.0, min(1.0, self.current_state.dominance))

        return self.current_state

    def get_current_state(self) -> EmotionalState:
        """Returns the current emotional state."""
        return self.current_state
