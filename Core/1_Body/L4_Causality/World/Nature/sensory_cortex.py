
import logging
import random
from typing import Dict, List, Optional
from Core.1_Body.L6_Structure.Wave.wave_dna import WaveDNA
from Core.1_Body.L4_Causality.World.Nature.trinity_lexicon import get_trinity_lexicon

logger = logging.getLogger("SensoryCortex")

class SensoryCortex:
    """
    [Phase 31] The Sensory Cortex.
    Decodes WaveDNA into High-Resolution Qualia (Scent, Music, Taste, Atmosphere).
    Acts as the 'Flesh' over the 'Logical Skeleton'.
    """
    def __init__(self):
        self.lexicon = get_trinity_lexicon()

    def decode_qualia(self, dna: WaveDNA) -> Dict[str, str]:
        """
        Translates raw DNA into a sensory experience map using scientific frequency bands.
        """
        experience = {
            "sound": self._decode_sound(dna),
            "aroma": self._decode_scent(dna),
            "flavor": self._decode_taste(dna),
            "tactile": self._decode_tactile(dna),
            "atmosphere": self._decode_atmosphere(dna),
            "aura": self._decode_aura(dna)
        }
        return experience

    def _decode_sound(self, dna: WaveDNA) -> str:
        # 20 Hz - 20 kHz
        if dna.frequency > 5000: return "Piercing, high-frequency harmonic shimmer."
        if dna.frequency > 300: return "Mid-range melodic resonance, warm and present."
        if dna.frequency > 20: return "Low-end rhythmic thrumming, vibrating the core."
        return "Infrasonic pressure wave."

    def _decode_scent(self, dna: WaveDNA) -> str:
        # 10 THz - 100 THz
        if dna.frequency > 8.0e13: return "Sharp, minty or ozonic molecular vibration."
        if dna.frequency > 5.0e13: return "Floral, sweet aromatic complexity."
        if dna.frequency > 1.0e13: return "Musky, heavy organic scent."
        return "Sub-olfactory chemical presence."

    def _decode_taste(self, dna: WaveDNA) -> str:
        # 0.1 THz - 10 THz
        if dna.frequency > 5.0e12: return "Spicy, electric tingle on the tongue."
        if dna.frequency > 1.0e12: return "Savory, rich depth with structural mass."
        if dna.frequency > 1.0e11: return "Sweet, smooth resonance without resistance."
        return "Neutral matter."

    def _decode_tactile(self, dna: WaveDNA) -> str:
        # 0.5 Hz - 1000 Hz
        if dna.frequency > 400: return "Sharp, stinging staccato. High-energy intrusion."
        if dna.frequency > 40: return "Rough, textured vibration. Grainy surface."
        if dna.frequency > 5: return "Firm pressure, steady and grounding."
        if dna.frequency > 0.1: return "Soft, slow rhythmic stroke. Calming affection."
        
        # Qualitative checks
        if dna.physical > 0.9 and dna.spiritual < 0.1: return "Intense physical pain (Dissonance)."
        if dna.spiritual > 0.9 and dna.phenomenal > 0.8: return "Deep sensual pleasure (Resonance)."
        return "Normal skin contact."

    def _decode_atmosphere(self, dna: WaveDNA) -> str:
        if dna.structural > 0.8: return "Cold, geometric precision. Perfectly still."
        if dna.mental > 0.8: return "Electric, vibrant with buzzing thoughts."
        if dna.spiritual > 0.9: return "Luminous and heavy with a sense of destiny."
        return "Ordinary reality."

    def _decode_aura(self, dna: WaveDNA) -> str:
        # Social field
        if dna.spiritual > 0.7 and dna.phenomenal > 0.7: return "Friendly and loving. Neighbors are cooperating."
        if dna.causal > 0.7 and dna.structural > 0.7: return "Strict and disciplined. Order is maintained."
        if dna.physical > 0.7 and dna.causal > 0.3: return "Tense and competitive. Potential for conflict."
        return "Peaceful coexistence."

_cortex = None
def get_sensory_cortex():
    global _cortex
    if _cortex is None:
        _cortex = SensoryCortex()
    return _cortex
