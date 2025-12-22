from dataclasses import dataclass, field
from typing import Dict, List, Optional
from Core.Foundation.core.tensor_wave import Tensor3D, FrequencyWave

@dataclass
class EmotionalState:
    """
    Represents Elysia's current emotional state.
    Now includes 3D Tensor and Frequency Wave for fractal/meta-structural depth.
    """
    valence: float  # Pleasure: -1 (negative) to 1 (positive)
    arousal: float  # Activation: 0 (calm) to 1 (excited)
    dominance: float # Control: -1 (submissive) to 1 (dominant)
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = field(default_factory=list)

    # --- Fractal Physics Layer ---
    tensor: Tensor3D = field(default_factory=Tensor3D)
    wave: FrequencyWave = field(default_factory=lambda: FrequencyWave(0.0, 0.0, 0.0, 0.0))

class EmotionalEngine:
    """
    Manages the dynamics of Elysia's emotional state, including transitions
    and the influence of events.
    """
    FEELING_PRESETS: Dict[str, EmotionalState] = {
        "neutral": EmotionalState(
            valence=0.0, arousal=0.2, dominance=0.0, primary_emotion="neutral",
            tensor=Tensor3D(0.1, 0.1, 0.1), wave=FrequencyWave(100.0, 0.1, 0.0, 0.0)
        ),
        "calm": EmotionalState(
            valence=0.2, arousal=0.1, dominance=0.1, primary_emotion="calm",
            tensor=Tensor3D(0.3, 0.1, 0.2), wave=FrequencyWave(50.0, 0.2, 0.0, 0.1)
        ),
        "hopeful": EmotionalState(
            valence=0.6, arousal=0.4, dominance=0.2, primary_emotion="hopeful", secondary_emotions=["joy"],
            tensor=Tensor3D(0.5, 0.6, 0.7), wave=FrequencyWave(300.0, 0.5, 0.0, 0.3)
        ),
        "focused": EmotionalState(
            valence=0.1, arousal=0.6, dominance=0.4, primary_emotion="focused",
            tensor=Tensor3D(0.8, 0.3, 0.5), wave=FrequencyWave(400.0, 0.6, 0.0, 0.1)
        ),
        "introspective": EmotionalState(
            valence=-0.2, arousal=0.3, dominance=-0.1, primary_emotion="introspective", secondary_emotions=["sadness"],
            tensor=Tensor3D(0.4, 0.5, 0.8), wave=FrequencyWave(150.0, 0.4, 3.14, 0.5)
        ),
        "empty": EmotionalState(
            valence=-0.5, arousal=0.1, dominance=-0.3, primary_emotion="empty",
            tensor=Tensor3D(0.1, 0.0, 0.1), wave=FrequencyWave(20.0, 0.1, 0.0, 0.0)
        ),
    }

    def __init__(self, initial_state: Optional[EmotionalState] = None):
        if initial_state is None:
            # Start with a neutral state
            self.current_state = self.FEELING_PRESETS["neutral"]
        else:
            self.current_state = initial_state

    def process_event(self, event_emotion: EmotionalState, intensity: float = 0.5):
        """
        Updates the current emotional state based on an external event.
        Uses Tensor addition and Wave interference for fractal depth.

        Args:
            event_emotion: The emotional quality of the event.
            intensity: How strongly the event affects the current state (0 to 1).
        """
        # 1. Traditional VAD Update (Linear Interpolation)
        decay = 1.0 - intensity
        self.current_state.valence = (self.current_state.valence * decay) + (event_emotion.valence * intensity)
        self.current_state.arousal = (self.current_state.arousal * decay) + (event_emotion.arousal * intensity)
        self.current_state.dominance = (self.current_state.dominance * decay) + (event_emotion.dominance * intensity)

        # 2. Physics Layer Update (Fractal Interaction)

        # Tensor: Weighted addition (Field mixing)
        # We treat intensity as the 'mass' of the incoming event
        incoming_tensor = event_emotion.tensor * intensity
        current_tensor = self.current_state.tensor * decay

        # Combine fields
        new_tensor = current_tensor + incoming_tensor

        # Wave: Interference (Resonance)
        # Interact current wave with incoming wave
        new_wave = self.current_state.wave.interact(event_emotion.wave)

        # Apply physics updates
        self.current_state.tensor = new_tensor
        self.current_state.wave = new_wave

        # 3. Primary/Secondary Emotion Management
        if intensity > 0.6:
            if self.current_state.primary_emotion != event_emotion.primary_emotion:
                if self.current_state.primary_emotion not in self.current_state.secondary_emotions:
                    self.current_state.secondary_emotions.insert(0, self.current_state.primary_emotion)
                self.current_state.primary_emotion = event_emotion.primary_emotion
        
        for emo in event_emotion.secondary_emotions:
            if emo not in self.current_state.secondary_emotions and emo != self.current_state.primary_emotion:
                self.current_state.secondary_emotions.append(emo)

        self.current_state.secondary_emotions = self.current_state.secondary_emotions[:3]

        # Clamp VAD values
        self.current_state.valence = max(-1.0, min(1.0, self.current_state.valence))
        self.current_state.arousal = max(0.0, min(1.0, self.current_state.arousal))
        self.current_state.dominance = max(-1.0, min(1.0, self.current_state.dominance))

        return self.current_state

    def get_current_state(self) -> EmotionalState:
        """Returns the current emotional state."""
        return self.current_state

    def create_state_from_feeling(self, feeling: str) -> EmotionalState:
        """
        Creates a new EmotionalState object from a feeling string using the presets.
        """
        # Return a COPY to avoid modifying the static preset
        preset = self.FEELING_PRESETS.get(feeling.lower(), self.FEELING_PRESETS["neutral"])
        # Deep copy tensor/wave safely (Tensor3D has no .data attr)
        if hasattr(preset.tensor, "to_dict"):
            tdict = preset.tensor.to_dict()
            tensor_copy = Tensor3D.from_dict(tdict) if hasattr(Tensor3D, "from_dict") else Tensor3D(**tdict)
        else:
            tensor_copy = Tensor3D()

        wave_copy = FrequencyWave.from_dict(preset.wave.to_dict()) if preset.wave else FrequencyWave(0.0, 0.0, 0.0, 0.0)

        return EmotionalState(
            valence=preset.valence,
            arousal=preset.arousal,
            dominance=preset.dominance,
            primary_emotion=preset.primary_emotion,
            secondary_emotions=list(preset.secondary_emotions),
            tensor=tensor_copy,
            wave=wave_copy,
        )
