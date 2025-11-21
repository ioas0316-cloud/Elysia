from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave

@dataclass
class EmotionalState:
    """
    Represents Elysia's current emotional state.
    Now includes 3D Tensor and Frequency Wave for fractal/meta-structural depth.

    Meta-Emotion (Fractal Expansion):
    - `meta_emotions`: A list of emotions *about* this emotion (e.g., "I am ashamed(meta) of my anger(primary)").
    - `target_concept`: The specific thought or object this emotion is directed at (Gravitational Center).
    """
    valence: float  # Pleasure: -1 (negative) to 1 (positive)
    arousal: float  # Activation: 0 (calm) to 1 (excited)
    dominance: float # Control: -1 (submissive) to 1 (dominant)
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = field(default_factory=list)

    # --- Fractal Physics Layer ---
    tensor: Tensor3D = field(default_factory=Tensor3D)
    wave: FrequencyWave = field(default_factory=lambda: FrequencyWave(0.0, 0.0, 0.0, 0.0))

    # --- Recursive Meta-Structure ---
    meta_emotions: List['EmotionalState'] = field(default_factory=list)
    target_concept: Optional[str] = None # What is this emotion orbiting?

    def calculate_total_mass(self) -> float:
        """Calculates the mass of this emotion including its meta-emotions."""
        base_mass = self.tensor.calculate_mass()
        meta_mass = sum(e.calculate_total_mass() * 0.3 for e in self.meta_emotions) # Meta-emotions add weight but less
        return base_mass + meta_mass

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

        Now includes 'Emotional Gravity':
        - If the current state is very heavy (Mass > Threshold), incoming emotions orbit it (become secondary/meta)
          rather than replacing it.
        """

        # 1. Calculate Physics (Mass & Gravity)
        current_mass = self.current_state.calculate_total_mass()
        incoming_mass = event_emotion.calculate_total_mass() * intensity

        # Gravity Threshold: If current state is massive (e.g. Deep Grief), new events just orbit it.
        # Unless incoming event is MORE massive (e.g. Shock), which displaces it.
        if current_mass > 2.0 and incoming_mass < current_mass:
            # --- Orbital Capture (Meta-Emotion Formation) ---
            # The incoming emotion becomes a meta-commentary or secondary feeling
            # preserving the core state.

            # Create a meta-emotion wrapper (e.g., feeling 'annoyed' at the interruption of 'grief')
            # For now, we just append it to secondary or meta list
            if event_emotion.primary_emotion not in self.current_state.secondary_emotions:
                self.current_state.secondary_emotions.append(event_emotion.primary_emotion)

            # Add physical mass to the current state
            self.current_state.tensor = self.current_state.tensor + (event_emotion.tensor * (intensity * 0.2))
            self.current_state.tensor.mass_offset += 0.1 # Increase complexity

            # Wave interference still happens but dampened
            self.current_state.wave = self.current_state.wave.interact(event_emotion.wave)

            return self.current_state

        # 2. Standard Interaction (Displacement/Mixing)

        # Traditional VAD Update (Linear Interpolation)
        decay = 1.0 - intensity
        self.current_state.valence = (self.current_state.valence * decay) + (event_emotion.valence * intensity)
        self.current_state.arousal = (self.current_state.arousal * decay) + (event_emotion.arousal * intensity)
        self.current_state.dominance = (self.current_state.dominance * decay) + (event_emotion.dominance * intensity)

        # Physics Layer Update (Fractal Interaction)
        # Tensor: Weighted addition (Field mixing)
        incoming_tensor = event_emotion.tensor * intensity
        current_tensor = self.current_state.tensor * decay
        new_tensor = current_tensor + incoming_tensor

        # Wave: Interference (Resonance)
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
        return EmotionalState(
            valence=preset.valence,
            arousal=preset.arousal,
            dominance=preset.dominance,
            primary_emotion=preset.primary_emotion,
            secondary_emotions=list(preset.secondary_emotions),
            tensor=Tensor3D(tensor=preset.tensor.data.copy()), # Deep copy tensor
            wave=FrequencyWave.from_dict(preset.wave.to_dict()) # Deep copy wave
        )
