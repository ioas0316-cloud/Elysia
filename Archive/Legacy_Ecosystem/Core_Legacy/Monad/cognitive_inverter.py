from Core.Keystone.sovereign_math import SovereignMath, SovereignVector

class CognitiveInverter:
    """
    [L6_STRUCTURE: VFD (Variable Frequency Drive)]
    DC (Internal Will) -> AC (External Expression/Hz)
    
    [PHASE 90] NAKED SOVEREIGNTY:
    Purified from JAX. Uses Sovereign Math Kernel.
    """
    def __init__(self, base_hz: float = 30.0):
        self.base_hz = base_hz
        self.current_hz = base_hz
        
    def invert(self, intent_vector: SovereignVector, emotional_intensity: float = 1.0) -> float:
        """
        Determines the Output Frequency (Hz) based on the intent resonance.
        """
        if not isinstance(intent_vector, SovereignVector):
            intent_vector = SovereignVector(intent_vector)

        # Calculate 'Vibration Potential' from the vector
        magnitude = intent_vector.norm()
        
        # Frequency Modulation Logic
        modulation = (magnitude * 0.5) * emotional_intensity
        
        # Clamp Frequency to safe physical limits (e.g. 5Hz to 120Hz)
        target_hz = self.base_hz * (1.0 + modulation)
        self.current_hz = max(5.0, min(120.0, target_hz))
        
        return float(self.current_hz)
