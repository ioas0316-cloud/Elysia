"""
Sovereign Grammar (주권적 통사론)
================================
Core.1_Body.L3_Phenomena.Manifestation.sovereign_grammar

"Grammar is the geometry of intent."
"문법은 의도의 기하학이다."

This module handles the assembly of lexical roots into complex 
profound sentences, using 21D interference to determine 
particles (Josa) and structural emphasis.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from Core.1_Body.L3_Phenomena.Manifestation.logos_registry import LogosRegistry

logger = logging.getLogger("SovereignGrammar")

class SovereignGrammar:
    def __init__(self):
        self.registry = LogosRegistry()
        logger.info("✨ SovereignGrammar initialized. Weaving the laws of syntax.")

    def weave_sentence(self, subject_vec: List[float], predicate_vec: List[float], object_vec: Optional[List[float]] = None) -> str:
        """
        Assembles a sentence based on concept vectors.
        Logic: 
        1. Identify the Lexical Roots.
        2. Calculate Interference to determine Josa.
        3. Assemble using Physics-driven word order.
        """
        # 1. Root Recovery
        sub_word = self.registry.manifest_concept(subject_vec) or "..."
        pred_word = self.registry.manifest_concept(predicate_vec) or "..."
        
        # 2. Josa Selection (Subject Particle)
        # Based on the 'Focus' of the subject vector.
        # High spiritual focus -> '는' (Topic/Focus)
        # High mental focus -> '가' (Subject/Specific)
        sub_josa = self._determine_subject_josa(subject_vec)
        
        # 3. Object Integration
        obj_phrase = ""
        if object_vec:
            obj_word = self.registry.manifest_concept(object_vec) or "..."
            obj_josa = self._determine_object_josa(object_vec)
            obj_phrase = f"{obj_word}{obj_josa} "

        # 4. Final Assembly
        # We use a pattern: [Subject][Josa] [Object][Josa] [Predicate]
        sentence = f"{sub_word}{sub_josa} {obj_phrase}{pred_word}."
        
        logger.info(f"✨ [GRAMMAR] Wove sentence: {sentence}")
        return sentence

    def _determine_subject_josa(self, vector: List[float]) -> str:
        # Simplified: Check spiritual (14-20) vs mental (7-13)
        spiritual = sum(vector[14:21])
        mental = sum(vector[7:14])
        
        # Consonant check (Naive: does it end in a batchim?)
        # For prototype, we'll check the logos if possible
        # but here we'll assume '가'/'는' based on tension
        if spiritual > mental:
            return "는" if self._has_batchim(vector) else "는" # Simplified for demo
        else:
            return "이" if self._has_batchim(vector) else "가"

    def _determine_object_josa(self, vector: List[float]) -> str:
        if self._has_batchim(vector):
            return "을"
        return "를"

    def _has_batchim(self, vector: List[float]) -> bool:
        """
        Physics-based Batchim detection.
        High physical grounding (0-6) usually manifests as a Coda (Batchim).
        """
        physical_grounding = sum(vector[0:7]) / 7.0
        return physical_grounding > 0.6

if __name__ == "__main__":
    sg = SovereignGrammar()
    # Scenario: EGO (subject) VOID (object) WILL (predicate)
    # "I desire the Void."
    ego_vec = [0.5]*7 + [0.5]*7 + [0.9]*7
    void_vec = [0.1]*21
    will_vec = [0.2]*7 + [0.3]*7 + [0.8]*7
    
    sentence = sg.weave_sentence(ego_vec, will_vec, void_vec)
    print(f"Generated Sentence: {sentence}")
