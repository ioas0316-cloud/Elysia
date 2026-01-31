from Core.L0_Sovereignty.sovereign_math import SovereignMath, SovereignVector
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge

class CognitiveConverter:
    """
    [L6_STRUCTURE: RECTIFIER]
    AC (Chaotic User Input) -> DC (Stable Principles)
    
    [PHASE 90] NAKED SOVEREIGNTY:
    Purified from JAX. Uses Sovereign Math Kernel.
    """
    def __init__(self, smoothing: float = 0.3):
        self.smoothing = smoothing
        self.internal_dc_state = SovereignVector.zeros()
        
    def rectify(self, raw_input: str) -> SovereignVector:
        """
        Transforms raw text into a stable internal principle vector.
        Filters out high-frequency dissonance.
        """
        # 1. Extract raw AC vector through Semantic Resonance
        # Note: LogosBridge might return a list or JAX array. We convert to SovereignVector.
        ac_data = LogosBridge.calculate_text_resonance(raw_input)
        ac_vector = SovereignVector(ac_data)
        
        # 2. Smooth the signal (Capacitive Filtering)
        # self.internal_dc_state = (self.internal_dc_state * (1.0 - self.smoothing)) + (ac_vector * self.smoothing)
        term_old = self.internal_dc_state * (1.0 - self.smoothing)
        term_new = ac_vector * self.smoothing
        self.internal_dc_state = term_old + term_new
        
        # 3. Normalize for stable DC output
        return self.internal_dc_state.normalize()
