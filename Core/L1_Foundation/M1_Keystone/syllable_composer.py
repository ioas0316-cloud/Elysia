"""
       (Syllable Composer)
===============================

      HyperCosmos                       .
         ,       '  (Interference)'                  .
"""

from typing import List, Tuple, Dict
from .hunminjeongeum import HunminJeongeum
from .field_phonetics import FieldPhonetics
from ..L6_Structure.Merkaba.hypercosmos import HyperCosmos

from .monadic_lexicon import MonadicLexicon

class SyllableComposer:
    """
            (Resonance)             ,
                (Will)                  .
    """
    
    def __init__(self, cosmos: HyperCosmos):
        self.cosmos = cosmos
        self.engine = HunminJeongeum()
        self.monads = MonadicLexicon.get_hangul_monads()
        
    def synthesize_word(self, target_feeling: str) -> str:
        """
               (  )           '  -      '       .
        """
        feelings_map = {
            'warmth': {'brightness': 0.8, 'softness': 0.8, 'tension': 0.2},
            'cold': {'brightness': 0.2, 'softness': 0.4, 'tension': 0.7},
            'sharp': {'brightness': 0.6, 'softness': 0.1, 'tension': 0.9},
            'peace': {'brightness': 0.5, 'softness': 0.9, 'tension': 0.1},
            'tree': {'brightness': 0.7, 'softness': 0.6, 'tension': 0.4},
            'growth': {'brightness': 0.9, 'softness': 0.5, 'tension': 0.6}
        }
        
        intent = feelings_map.get(target_feeling.lower(), {'brightness': 0.5, 'softness': 0.5})
        
        # 1.             
        cho = self.engine.select_by_intent(intent, 'consonant')
        jung = self.engine.select_by_intent(intent, 'vowel')
        
        # 2.       (Word)
        syllable = self.engine.compose(cho, jung)
        
        # 3.            (Qualia Isomorphism Check)
        #       -          ,                             
        match_score, gap_desc = self.calculate_qualia_isomorphism(intent, [cho, jung])
        
        # 4.              
        narrative_stimulus = f"[{syllable}] - Qualia Match: {match_score:.2f}. {gap_desc}"
        decision = self.cosmos.perceive(narrative_stimulus)
        
        symmetry_text = "  [ESSENCE MATCH]                   " if match_score > 0.8 else f"   [PROCESS GAP]               ({1.0 - match_score:.2f})"
        
        return (
            f"  ('{target_feeling}') ->   ('{syllable}')\n"
            f"{symmetry_text}\n"
            f"      : {gap_desc}\n"
            f"     : {decision.narrative}"
        )

    def calculate_qualia_isomorphism(self, intent: Dict[str, float], jamo_list: List[str]) -> Tuple[float, str]:
        """
          (  )  7D       (   )                   .
        """
        total_match = 0.0
        details = []
        
        # intent['softness'] <-> monad['profile']['Physical'] (       )
        # intent['brightness'] <-> monad['profile']['Phenomenal'] (     )
        
        for jamo in jamo_list:
            if jamo in self.monads:
                profile = self.monads[jamo]['profile']
                #          
                jamo_match = 0.0
                if 'softness' in intent and 'Physical' in profile:
                    #         (0.8)            (0.8)      
                    m = 1.0 - abs(intent['softness'] - profile['Physical'])
                    jamo_match = max(jamo_match, m)
                
                total_match += jamo_match
                details.append(f"{jamo}(     {profile.get('Physical', 0.5):.1f})")
                
        avg_match = total_match / len(jamo_list) if jamo_list else 0.0
        return avg_match, f"     : {' + '.join(details)}"

    def verify_symmetry(self, intent: Dict[str, float], jamo_list: List[str]) -> Tuple[bool, float]:
        # Legacy   
        score, _ = self.calculate_qualia_isomorphism(intent, jamo_list)
        return score > 0.8, 1.0 - score