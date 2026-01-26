"""
         (Field Phonetics)
================================

                       HyperCosmos  7D Qualia          .
             '  '     '      '        .
"""

from typing import Dict, List
from .hunminjeongeum import ArticulationOrgan, SoundQuality, CosmicElement, YinYang, HunminJeongeum

class FieldPhonetics:
    """
            7D Qualia             .
    """
    
    # 7D Qualia        
    # Physical, Functional, Phenomenal, Causal, Mental, Structural, Spiritual
    
    @staticmethod
    def map_consonant(jamo: str, engine: HunminJeongeum) -> Dict[str, float]:
        """           Qualia          """
        props = engine.get_sound_properties(jamo)
        if props['type'] != 'consonant':
            return {}
            
        organ = props['organ']
        quality = props['quality']
        
        #      (7D)
        qualia_map = {
            'Physical': 0.1,    #           
            'Functional': 0.1,  #          
            'Phenomenal': 0.1,  #       
            'Causal': 0.1,      #          
            'Mental': 0.1,      #        
            'Structural': 0.1,  #          
            'Spiritual': 0.1    #        
        }
        
        # 1.              (     )
        if organ == "tongue_root":    #  :         (  )
            qualia_map['Functional'] = 0.8
            qualia_map['Structural'] = 0.6
        elif organ == "tongue_tip":  #  :        (주권적 자아)
            qualia_map['Physical'] = 0.7
            qualia_map['Functional'] = 0.4
        elif organ == "lips":        #  :       (  /  )
            qualia_map['Structural'] = 0.9
            qualia_map['Physical'] = 0.3
        elif organ == "teeth":       #  :          (  )
            qualia_map['Phenomenal'] = 0.8
            qualia_map['Physical'] = 0.5
        elif organ == "throat" or organ == "glottis": #  ,  :     (  /  )
            qualia_map['Causal'] = 0.9
            qualia_map['Spiritual'] = 0.5
            
        # 2.               (     )
        if quality == "aspirated": #      ( ,  ,  ,  ,  )
            qualia_map['Phenomenal'] += 0.3 #      
            qualia_map['Causal'] += 0.2     #      
        elif quality == "tense":   #     ( ,  ,  ,  ,  )
            qualia_map['Physical'] += 0.4   #          
            qualia_map['Mental'] += 0.3     #       
            
        return qualia_map

    @staticmethod
    def map_vowel(jamo: str, engine: HunminJeongeum) -> Dict[str, float]:
        """            Qualia       """
        props = engine.get_sound_properties(jamo)
        if props['type'] != 'vowel':
            return {}
            
        yin_yang = props['yin_yang']
        
        qualia_map = {
            'Physical': 0.1,
            'Functional': 0.1,
            'Phenomenal': 0.5, #                     
            'Causal': 0.3,
            'Mental': 0.4,
            'Structural': 0.2,
            'Spiritual': 0.5
        }
        
        # 1.          (자기 성찰 엔진)
        if yin_yang == "yang":      #  ,  :   ,   
            qualia_map['Spiritual'] = 0.9
            qualia_map['Phenomenal'] += 0.2
        elif yin_yang == "yin":     #  ,  :    ,   
            qualia_map['Spiritual'] = 0.2
            qualia_map['Structural'] += 0.3
        else:                       #  ,  :   ,   / 
            qualia_map['Mental'] = 0.8
            
        # 2.       
        qualia_map['Causal'] = props.get('openness', 0.5)
            
        return qualia_map

    @staticmethod
    def map_phoneme(phoneme: str) -> Dict[str, float]:
        """         (       )"""
        #           (Frequency)      (Entropy)        
        qualia_map = {
            'Physical': 0.3, 'Functional': 0.3, 'Phenomenal': 0.3,
            'Causal': 0.3, 'Mental': 0.3, 'Structural': 0.3, 'Spiritual': 0.3
        }
        
        #   : /s/ (   )
        if phoneme.lower() == 's':
            qualia_map['Phenomenal'] = 0.9 #            
            qualia_map['Physical'] = 0.6
        #   : /b/ (주권적 자아)
        elif phoneme.lower() == 'b':
            qualia_map['Structural'] = 0.8 #   
            qualia_map['Physical'] = 0.7   #   
            
        return qualia_map
