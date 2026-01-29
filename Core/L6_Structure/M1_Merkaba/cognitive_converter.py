import jax.numpy as jnp
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge

class CognitiveConverter:
    """
    [L6_STRUCTURE: RECTIFIER]
    AC (Chaotic User Input) -> DC (Stable Principles)
    
    Extracts the 'Direct Current' of meaning from the 'Alternating' noise 
    of conversational variability.
    """
    def __init__(self, smoothing: float = 0.3):
        self.smoothing = smoothing
        self.internal_dc_state = jnp.zeros(21)
        
    def rectify(self, raw_input: str) -> jnp.ndarray:
        """
        Transforms raw text into a stable internal principle vector.
        Filters out high-frequency dissonance.
        """
        # 1. Extract raw AC vector through Semantic Resonance
        ac_vector = LogosBridge.calculate_text_resonance(raw_input)
        
        # 2. Smooth the signal (Capacitive Filtering)
        # We integrate the new input into the existing internal state
        self.internal_dc_state = (self.internal_dc_state * (1.0 - self.smoothing)) + (ac_vector * self.smoothing)
        
        # 3. Normalize for stable DC output
        norm = jnp.linalg.norm(self.internal_dc_state) + 1e-6
        return self.internal_dc_state / norm
