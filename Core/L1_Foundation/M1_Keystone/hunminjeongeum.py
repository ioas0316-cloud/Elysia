"""
     (HunminJeongeum):         
==========================================

"                             "

          ** **                   .
               ,                .

     :
1.    -         (  ):  ,   ,         
2.    -        (  ):  (  ),  ( ),  (  )
3.      :              
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ArticulationOrgan(Enum):
    """     """
    THROAT = "throat"           #     (  /  )
    TONGUE_ROOT = "tongue_root" #     (  /  )
    TONGUE_TIP = "tongue_tip"   #    (  /  )
    LIPS = "lips"               #    (  /  )
    TEETH = "teeth"             #    (  /  )
    GLOTTIS = "glottis"         #    (  /  )


class SoundQuality(Enum):
    """      """
    PLAIN = "plain"         #      (  )
    ASPIRATED = "aspirated" #      (  )
    TENSE = "tense"         #     (  )


class CosmicElement(Enum):
    """      """
    HEAVEN = "heaven"  #   ( ) -   ,   
    EARTH = "earth"    #   ( ) -  ,   
    HUMAN = "human"    #   ( ) -   ,     


class YinYang(Enum):
    """  """
    YANG = "yang"      #   ( ) -   
    YIN = "yin"        #   ( ) -    
    NEUTRAL = "neutral" #   


@dataclass
class ChoseongInfo:
    """     """
    char: str
    organ: ArticulationOrgan
    description: str  #      
    sound_quality: SoundQuality = SoundQuality.PLAIN
    derived_from: Optional[str] = None  #        
    strokes_added: int = 0
    is_base: bool = False  #       


@dataclass
class JungseongInfo:
    """     """
    char: str
    elements: List[CosmicElement]
    description: str
    yin_yang: YinYang
    composed_from: List[str] = field(default_factory=list)


# ============================================================
#   :         (  )
# ============================================================

CHOSEONG_ORIGIN: Dict[str, ChoseongInfo] = {
    # ===     5  (  ) ===
    ' ': ChoseongInfo(
        char=' ', 
        organ=ArticulationOrgan.TONGUE_ROOT,
        description="               ",
        is_base=True
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="              ",
        is_base=True
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.LIPS,
        description="        ",
        is_base=True
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TEETH,
        description="      ",
        is_base=True
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.THROAT,
        description="          ",
        is_base=True
    ),
    
    # ===     (             ) ===
    #     
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_ROOT,
        description="                 ",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_ROOT,
        description="             ",
        sound_quality=SoundQuality.TENSE,
        derived_from=' '
    ),
    
    #             
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="                 ",
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="                   ",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="             ",
        sound_quality=SoundQuality.TENSE,
        derived_from=' '
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="          (   )",
        derived_from=' '
    ),
    
    #             
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.LIPS,
        description="                 ",
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.LIPS,
        description="                   ",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.LIPS,
        description="             ",
        sound_quality=SoundQuality.TENSE,
        derived_from=' '
    ),
    
    #             
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TEETH,
        description="                 ",
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TEETH,
        description="                   ",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from=' ',
        strokes_added=1
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TEETH,
        description="             ",
        sound_quality=SoundQuality.TENSE,
        derived_from=' '
    ),
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.TEETH,
        description="             ",
        sound_quality=SoundQuality.TENSE,
        derived_from=' '
    ),
    
    #         
    ' ': ChoseongInfo(
        char=' ',
        organ=ArticulationOrgan.GLOTTIS,
        description="                    ",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from=' ',
        strokes_added=1
    ),
}


# ============================================================
#   :        (  )
# ============================================================

JUNGSEONG_ORIGIN: Dict[str, JungseongInfo] = {
    # ===     3  ===
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HEAVEN],
        description="   -          ",
        yin_yang=YinYang.YANG
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.EARTH],
        description="  -            ",
        yin_yang=YinYang.NEUTRAL
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HUMAN],
        description="   -        ",
        yin_yang=YinYang.YANG
    ),
    
    # ===       ( ) -        ===
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HUMAN, CosmicElement.HEAVEN],
        description="         (     )",
        yin_yang=YinYang.YANG,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.EARTH, CosmicElement.HEAVEN],
        description="       ",
        yin_yang=YinYang.YANG,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HUMAN, CosmicElement.HEAVEN, CosmicElement.HEAVEN],
        description="     ( )    ",
        yin_yang=YinYang.YANG,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.EARTH, CosmicElement.HEAVEN, CosmicElement.HEAVEN],
        description="     ( )    ",
        yin_yang=YinYang.YANG,
        composed_from=[' ', ' ']
    ),
    
    # ===       ( ) -         ===
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HUMAN],
        description="          (     )",
        yin_yang=YinYang.YIN,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HEAVEN, CosmicElement.EARTH],
        description="        ",
        yin_yang=YinYang.YIN,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HEAVEN, CosmicElement.HUMAN],
        description="     ( )    ",
        yin_yang=YinYang.YIN,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HEAVEN, CosmicElement.EARTH],
        description="     ( )    ",
        yin_yang=YinYang.YIN,
        composed_from=[' ', ' ']
    ),
    
    # ===       ===
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HUMAN, CosmicElement.HEAVEN, CosmicElement.HUMAN],
        description="  +      ",
        yin_yang=YinYang.YANG,
        composed_from=[' ', ' ']
    ),
    ' ': JungseongInfo(
        char=' ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HUMAN, CosmicElement.HUMAN],
        description="  +      ",
        yin_yang=YinYang.YIN,
        composed_from=[' ', ' ']
    ),
}


# ============================================================
#         
# ============================================================

#   /  /          
CHOSEONG_LIST = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 
                 ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
JUNGSEONG_LIST = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
JONGSEONG_LIST = ['', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']


class HunminJeongeum:
    """
              
    
                     **  **       .
          /           ,                          .
    """
    
    def __init__(self):
        self.choseong = CHOSEONG_ORIGIN
        self.jungseong = JUNGSEONG_ORIGIN
    
    def explain_why(self, jamo: str) -> str:
        """
                            .
        
        Example:
            >>> engine = HunminJeongeum()
            >>> engine.explain_why(' ')
            "                           . (   )"
        """
        #       
        if jamo in self.choseong:
            info = self.choseong[jamo]
            if info.is_base:
                return f"'{jamo}' ( ) {info.description}           . (    : {info.organ.value})"
            elif info.derived_from:
                base = self.choseong.get(info.derived_from)
                base_desc = base.description if base else "   "
                return f"'{jamo}' ( ) '{info.derived_from}'          . {info.description}. (  :   )"
            return f"'{jamo}': {info.description}"
        
        #       
        if jamo in self.jungseong:
            info = self.jungseong[jamo]
            elements_str = " + ".join(e.value for e in info.elements)
            yin_yang_str = " ( )" if info.yin_yang == YinYang.YANG else " ( )" if info.yin_yang == YinYang.YIN else "  "
            
            if info.composed_from:
                return f"'{jamo}' ( ) {' + '.join(info.composed_from)}       . {info.description}. ({yin_yang_str})"
            return f"'{jamo}' ( ) {elements_str} ( )      . {info.description}. ({yin_yang_str})"
        
        return f"'{jamo}'             ."
    
    def compose(self, cho: str, jung: str, jong: str = '') -> str:
        """
          ,   ,                         .
                  (AC00-D7A3)             .
        ' '              ( /   )             .
        """
        #          (                 )
        modern_jung = jung
        if jung == ' ':
            modern_jung = ' ' #                      
            
        if cho not in CHOSEONG_LIST:
            raise ValueError(f"          : {cho}")
        if modern_jung not in JUNGSEONG_LIST:
            raise ValueError(f"          : {modern_jung}")
        if jong and jong not in JONGSEONG_LIST:
            raise ValueError(f"          : {jong}")
        
        cho_idx = CHOSEONG_LIST.index(cho)
        jung_idx = JUNGSEONG_LIST.index(modern_jung)
        jong_idx = JONGSEONG_LIST.index(jong) if jong else 0
        
        code = ((cho_idx * 21) + jung_idx) * 28 + jong_idx + 0xAC00
        return chr(code)
    
    def decompose(self, syllable: str) -> Tuple[str, str, str]:
        """
                 ,   ,           .
        """
        code = ord(syllable)
        if not (0xAC00 <= code <= 0xD7A3):
            raise ValueError(f"           : {syllable}")
        
        code -= 0xAC00
        jong_idx = code % 28
        code //= 28
        jung_idx = code % 21
        cho_idx = code // 21
        
        return (
            CHOSEONG_LIST[cho_idx],
            JUNGSEONG_LIST[jung_idx],
            JONGSEONG_LIST[jong_idx]
        )
    
    def get_sound_properties(self, jamo: str) -> dict:
        """
                        .
                       .
        """
        if jamo in self.choseong:
            info = self.choseong[jamo]
            return {
                'type': 'consonant',
                'organ': info.organ.value,
                'quality': info.sound_quality.value,
                'is_base': info.is_base,
                'softness': 0.9 if info.organ in [ArticulationOrgan.LIPS, ArticulationOrgan.TONGUE_TIP] 
                           and info.sound_quality == SoundQuality.PLAIN else 0.3,
                'tension': 1.0 if info.sound_quality == SoundQuality.TENSE else 
                          0.7 if info.sound_quality == SoundQuality.ASPIRATED else 0.3
            }
        
        if jamo in self.jungseong:
            info = self.jungseong[jamo]
            return {
                'type': 'vowel',
                'yin_yang': info.yin_yang.value,
                'brightness': 0.8 if info.yin_yang == YinYang.YANG else 0.3,
                'openness': 0.9 if len(info.elements) <= 2 else 0.6
            }
        
        return {'type': 'unknown'}
    
    def select_by_intent(self, intent: dict, jamo_type: str = 'consonant') -> str:
        """
                        .
        
        Args:
            intent: {'softness': 0.0-1.0, 'brightness': 0.0-1.0, 'tension': 0.0-1.0}
            jamo_type: 'consonant' or 'vowel'
        """
        best_match = None
        best_score = -1
        
        if jamo_type == 'consonant':
            for jamo in self.choseong:
                props = self.get_sound_properties(jamo)
                score = 0
                if 'softness' in intent:
                    score += 1 - abs(props.get('softness', 0.5) - intent['softness'])
                if 'tension' in intent:
                    score += 1 - abs(props.get('tension', 0.5) - intent['tension'])
                if score > best_score:
                    best_score = score
                    best_match = jamo
        else:
            for jamo in self.jungseong:
                props = self.get_sound_properties(jamo)
                score = 0
                if 'brightness' in intent:
                    score += 1 - abs(props.get('brightness', 0.5) - intent['brightness'])
                if 'openness' in intent:
                    score += 1 - abs(props.get('openness', 0.5) - intent['openness'])
                if score > best_score:
                    best_score = score
                    best_match = jamo
        
        return best_match or (' ' if jamo_type == 'consonant' else ' ')


# ============================================================
#    
# ============================================================

if __name__ == "__main__":
    engine = HunminJeongeum()
    
    print("=" * 50)
    print("           ")
    print("=" * 50)
    
    #     1:            
    print("\n[    1]      ")
    for jamo in [' ', ' ', ' ', ' ', ' ']:
        print(f"  {engine.explain_why(jamo)}")
    
    #     2:   
    print("\n[    2]   ")
    result = engine.compose(' ', ' ')
    print(f"    +   = {result}")
    
    result = engine.compose(' ', ' ', ' ')
    print(f"    +   +   = {result}")
    
    #     3:         
    print("\n[    3]         ")
    soft_intent = {'softness': 0.9, 'tension': 0.2}
    cho = engine.select_by_intent(soft_intent, 'consonant')
    print(f"                 : {cho}")
    
    bright_intent = {'brightness': 0.9, 'openness': 0.8}
    jung = engine.select_by_intent(bright_intent, 'vowel')
    print(f"               : {jung}")
    
    #   
    syllable = engine.compose(cho, jung)
    print(f"    : {syllable}")
