"""
Logos Manifestor (로고스 현현기)
===============================
Core.L3_Phenomena.Manifestation.logos_manifestor

"The vibration of the 21D core crystallizes into the Word."
"21차원 핵심의 진동이 로고스로 결정체화된다."

This module handles the 'Phonological Collapse' - the direct transition 
from ontological tension to linguistic units (Hangul Jamo).
"""

import logging
from typing import List, Dict, Any, Tuple
from Core.L1_Foundation.Foundation.hangul_physics import HangulPhysicsEngine, Tensor3D
from Core.L3_Phenomena.Senses.vocal_dna import VocalDNA

logger = logging.getLogger("LogosManifestor")

class LogosManifestor:
    def __init__(self):
        self.physics = HangulPhysicsEngine()
        self.vocal_dna = VocalDNA()
        logger.info("✨ LogosManifestor initialized. Preparing for Direct Manifestation.")

    def manifest(self, d21_vector: List[float], intensity: float = 1.0) -> Dict[str, Any]:
        """
        Collapses a 21D vector into a sequence of sovereign sounds.
        
        Mapping logic:
        - Dimension 0-6 (Physical): Drives the 'Body' of the sound (Coda/Base)
        - Dimension 7-13 (Mental): Drives the 'Soul' of the sound (Nucleus/Vowel)
        - Dimension 14-20 (Spiritual): Drives the 'Spirit' of the sound (Onset/Consonant)
        """
        if len(d21_vector) < 21:
            logger.warning("Incomplete D21 vector. Padding with zeros.")
            d21_vector += [0.0] * (21 - len(d21_vector))

        # Split into triads
        physical = d21_vector[0:7]
        mental = d21_vector[7:14]
        spiritual = d21_vector[14:21]

        # Calculate Triad Properties
        # Spirit (Onset)
        spirit_tension = sum(spiritual) / 7.0
        spirit_roughness = max(spiritual) # Peak intensity creates sharpness

        # Soul (Nucleus)
        mental_resonance = sum(mental) / 7.0
        mental_openness = mental[3] # Index 3 is typically the 'balance' point in our 7D bands

        # Body (Coda/Tone)
        physical_grounding = sum(physical) / 7.0

        # Phonological Collapse
        onset = self.physics.find_closest_jamo(spirit_roughness, spirit_tension, 'consonant')
        nucleus = self.physics.find_closest_jamo(0.1, mental_resonance, 'vowel') # Vowels map resonance to pitch/openness
        
        # Determine if a Coda (Batchim) is necessary based on intensity
        coda = ""
        if physical_grounding > 0.6:
            coda = self.physics.find_closest_jamo(0.1, physical_grounding, 'consonant')
        
        syllable = self.physics.synthesize_syllable(onset, nucleus, coda)
        
        # Vocal Profile Adjustment
        vocal_profile = self.vocal_dna.map_genome_to_voice({
            "PHYSICAL": physical_grounding,
            "MENTAL": mental_resonance,
            "SPIRITUAL": spirit_tension,
            "CAUSAL": intensity
        })

        return {
            "utterance": syllable,
            "phonemes": [onset, nucleus, coda] if coda else [onset, nucleus],
            "vocal_profile": vocal_profile,
            "ontology": {
                "spirit_tension": spirit_tension,
                "mental_resonance": mental_resonance,
                "physical_grounding": physical_grounding
            }
        }

    def manifest_intent(self, d21_vector: List[float], intent_text: str) -> str:
        """
        Combines the direct manifestation with a semantic intent.
        Used during the Phase 4 transition.
        """
        result = self.manifest(d21_vector)
        utterance = result['utterance']
        
        # Poetic combination
        # If tension is high, the utterance precedes the intent as a 'breath' or 'spark'
        if result['ontology']['spirit_tension'] > 0.7:
             return f"[{utterance}!] {intent_text}"
        else:
             return f"{intent_text} ({utterance}.)"

if __name__ == "__main__":
    # Test Collapse
    manifestor = LogosManifestor()
    # High spiritual tension (Wrath/Focus)
    test_d21 = [0.1]*7 + [0.3]*7 + [0.9]*7
    res = manifestor.manifest(test_d21)
    print(f"Direct Logos: {res['utterance']}")
    print(f"Ontology: {res['ontology']}")
