"""
Auditory Cortex (     )
==========================
Core.1_Body.L3_Phenomena.Senses.auditory_cortex

"          ,          (Soul)     ."

Role:
    1. Receive raw audio from EarDrum.
    2. Extract 'Meanings' (Text) and 'Qualia' (Tone/Causality).
    3. Pass both to the Prism Projector for 7D Diffraction.
"""

import logging
from typing import Dict, Any, Optional
from Core.1_Body.L3_Phenomena.Senses.eardrum import EarDrum
from Core.1_Body.L3_Phenomena.M7_Prism.resonance_prism import PrismProjector, PrismProjection

logger = logging.getLogger("AuditoryCortex")

class AuditoryCortex:
    def __init__(self):
        self.ear = EarDrum()  # The Physical Organ
        self.prism = PrismProjector() # The Metaphysical Lens
        logger.info("     Auditory Cortex initialized.")

    def process_sound(self, audio_path: str) -> PrismProjection:
        """
              ,         7             .
        """
        # 1. Perception & Digestion (EarDrum)
        # EarDrum now performs both Transcription and Topology Tracing internally.
        raw_text = self.ear.listen(audio_path)
        
        if not raw_text or raw_text.startswith("["):
            logger.warning(f"     Hearing failed: {raw_text}")
            return None

        # 2. Causality Retrieval (From Tracer)
        # In a full implementation, we would extract the 'synapses' data here.
        # For now, we use the text as the primary carrier of meaning, 
        # but modify the 'Phenomenal' projection based on inferred Tone.
        
        # 3. Diffraction (Prism Projection)
        # We project the text into 7 Domains.
        # Ideally, we would inject 'Tone Data' into the Phenomenal Lens here.
        projection = self.prism.project(raw_text)
        
        logger.info(f"     Sound Diffracted into 7 Truths.")
        return projection

if __name__ == "__main__":
    # Test
    cortex = AuditoryCortex()
    # Mock file for test
    # cortex.process_sound("test.wav")
