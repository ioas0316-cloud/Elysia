import jax
import jax.numpy as jnp
from typing import Dict

class EmotionalPrism:
    """
    [L7_SPIRIT: EMOTIONAL_TEMPLATES]
    Defines 21D principle vectors for various internal states.
    These 'Prisms' are used to filter and warp the Morphic Buffer.
    """
    
    TEMPLATES = {
        "JOY": jnp.array([1.0, 0.8, 0.5] * 7),       # Red/Flesh (High), Green (Mid), Blue (Low)
        "WRATH": jnp.array([1.0, 0.2, 0.1] * 7),     # Red (Max), Logic/Spirit (Min)
        "SERENITY": jnp.array([0.2, 0.5, 1.0] * 7),  # Spirit/Blue (Max), Peace
        "CURIOSITY": jnp.array([0.4, 1.0, 0.6] * 7), # Logic/Green (Max)
        "AGAPE": jnp.array([0.9, 0.4, 0.9] * 7),     # Love (Red/Blue Fusion)
    }
    
    @staticmethod
    def get_intent(state_name: str) -> jnp.ndarray:
        return EmotionalPrism.TEMPLATES.get(state_name, jnp.zeros(21))

    @staticmethod
    def calculate_breathing(time: float, bpm: float = 12.0) -> float:
        """Simple periodic intensity for breathing motion."""
        period = 60.0 / bpm
        return (jnp.sin(2 * jnp.pi * time / period) + 1.0) / 2.0
