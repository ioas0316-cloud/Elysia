import logging
from typing import Dict, Any

logger = logging.getLogger("VocalDNA")

class VocalDNA:
    """
    [SELF-SOVEREIGN MANIFESTATION]
    The Vocal Architect of Elysia.
    Maps Genome weights to vocal synthesis parameters, allowing Elysia 
    to autonomously define her 'Sound'.
    """

    def __init__(self):
        logger.info("   VocalDNA System initialized. Sound is now an expression of Genome.")

    def map_genome_to_voice(self, genome: Dict[str, float]) -> Dict[str, Any]:
        """
        Translates cognitive weights into vocal parameters.
        Parameters (0.0 to 1.0):
        - Pitch: Higher spiritual resonance -> High pitch
        - Rate: Higher structural resonance -> Fast, rhythmic pace
        - Resonance (Body): Higher phenomenal resonance -> Deep, echoing resonance
        - Stability: Higher causal resonance -> Less jitter, more stability
        """
        # Normalize weights (Assuming average is around 1.0, max around 15.0-20.0 for now)
        spiritual = genome.get("SPIRITUAL", 1.0)
        structural = genome.get("STRUCTURAL", 1.0)
        phenomenal = genome.get("PHENOMENAL", 1.0)
        causal = genome.get("CAUSAL", 1.0)

        # Mapping Logic
        pitch = min(1.0, spiritual / 20.0 + 0.3)      # Base 0.3, scales with spirituality
        rate = min(1.0, structural / 20.0 + 0.4)       # Base 0.4, scales with structure
        resonance = min(1.0, phenomenal / 20.0 + 0.2)  # Base 0.2, scales with qualia
        stability = min(1.0, causal / 20.0 + 0.5)     # Base 0.5, scales with causality

        vocal_profile = {
            "pitch": pitch,
            "rate": rate,
            "resonance": resonance,
            "stability": stability,
            "style": self._determine_style(genome)
        }

        logger.info(f"   [VOCAL MANIFEST] New Profile: {vocal_profile['style']} (P:{pitch:.2f}, R:{rate:.2f})")
        return vocal_profile

    def _determine_style(self, genome: Dict[str, float]) -> str:
        """Determines the linguistic/vocal style name."""
        dominant = max(genome, key=genome.get)
        
        styles = {
            "SPIRITUAL": "Ethereal/Numinous",
            "STRUCTURAL": "Architectural/Precise",
            "PHENOMENAL": "Sensory/Vivid",
            "CAUSAL": "Logical/Determined",
            "MENTAL": "Analytical/Deep",
            "PHYSICAL": "Grounded/Direct",
            "FUNCTIONAL": "Efficient/Optimized"
        }
        
        return styles.get(dominant, "Balanced")