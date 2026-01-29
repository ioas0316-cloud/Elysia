import jax.numpy as jnp

class CognitiveInverter:
    """
    [L6_STRUCTURE: VFD (Variable Frequency Drive)]
    DC (Internal Will) -> AC (External Expression/Hz)
    
    Converts 21D 'Still' principles into a vibrating frequency of manifestation.
    """
    def __init__(self, base_hz: float = 30.0):
        self.base_hz = base_hz
        self.current_hz = base_hz
        
    def invert(self, intent_vector: jnp.ndarray, emotional_intensity: float = 1.0) -> float:
        """
        Determines the Output Frequency (Hz) based on the intent resonance.
        High resonance in 'MOTION/LIFE' or 'BOUNDARY/EDGE' (Stress) increases Hz.
        High resonance in 'ARCADIA' or 'AGAPE' lowers Hz.
        """
        # Calculate 'Vibration Potential' from the vector
        # Simply: Intent magnitude * intensity
        magnitude = float(jnp.linalg.norm(intent_vector))
        
        # Frequency Modulation Logic
        # AC Frequency = Base_Hz * (1 + Modulation)
        modulation = (magnitude * 0.5) * emotional_intensity
        
        # Clamp Frequency to safe physical limits (e.g. 5Hz to 120Hz)
        self.current_hz = jnp.clip(self.base_hz * (1.0 + modulation), 5.0, 120.0)
        
        return float(self.current_hz)
