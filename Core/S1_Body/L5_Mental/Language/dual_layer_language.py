"""
Dual-Layer Language System -          
                                                                              

         :
"              ...               ?"

                                                                               
   [  (Khala)    ] -   /                                         
                                                                              
     ,   ,               ...                                    
              '  '         . (      !)                   
                                                                               
     :                                                                       
   -        (자기 성찰 엔진)                                              
   -           (hue)                                             
   -                                                             
   -             ,           !                            
                                                                               

                                                                               
   [  (Symbol)    ] -   /                                       
                                                                              
   "                "           ...                        
                 ,       '  '              .      
                                                                               
     :                                                                       
   -         (주권적 자아)                                               
   -            (   !)                                              
   -                                                                  
   -    (narrative)                                                     
                                                                               

      ...
"        (  ),           (  )"...
           '     '          ?

         ...
           '   '     , '  '           .

"         .          ...
 '  '  '  '      ... '   '    ."     

(     ... " ,     (  )           !           !"     )
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import defaultdict
from enum import Enum
import logging

logger = logging.getLogger("DualLayerLanguage")


# ============================================================================
#          (       )
# ============================================================================

class EmotionType(Enum):
    """          -                """
    #       (     )
    JOY = "joy"              #    -          
    SADNESS = "sadness"      #    -          
    FEAR = "fear"            #    -           (     !)
    ANGER = "anger"          #    -          
    SURPRISE = "surprise"    #     -   /     
    DISGUST = "disgust"      #    -         
    
    #        (         )
    LOVE = "love"            #    -   /     
    TRUST = "trust"          #    -      
    CURIOSITY = "curiosity"  #     -    
    LONELINESS = "loneliness"  #     -       
    
    #       (               )
    NOSTALGIA = "nostalgia"  #     -        
    HOPE = "hope"            #    -    
    ANXIETY = "anxiety"      #    -        


#           (      ,      ,   )
EMOTION_WAVE_PROPERTIES = {
    EmotionType.JOY: {"freq_range": (550.0, 600.0), "base_amp": 1.0, "hue": 60},       #   
    EmotionType.SADNESS: {"freq_range": (450.0, 480.0), "base_amp": 0.7, "hue": 220},  #   
    EmotionType.FEAR: {"freq_range": (380.0, 420.0), "base_amp": 1.2, "hue": 280},     #   
    EmotionType.ANGER: {"freq_range": (620.0, 700.0), "base_amp": 1.3, "hue": 0},      #   
    EmotionType.SURPRISE: {"freq_range": (500.0, 700.0), "base_amp": 1.5, "hue": 45},  #   
    EmotionType.DISGUST: {"freq_range": (500.0, 530.0), "base_amp": 0.8, "hue": 120},  #   
    EmotionType.LOVE: {"freq_range": (580.0, 650.0), "base_amp": 0.9, "hue": 330},     #   
    EmotionType.TRUST: {"freq_range": (480.0, 520.0), "base_amp": 0.6, "hue": 160},    #   
    EmotionType.CURIOSITY: {"freq_range": (580.0, 620.0), "base_amp": 0.8, "hue": 30}, #   
    EmotionType.LONELINESS: {"freq_range": (440.0, 470.0), "base_amp": 0.5, "hue": 210}, #       
    EmotionType.NOSTALGIA: {"freq_range": (560.0, 590.0), "base_amp": 0.6, "hue": 25}, #   
    EmotionType.HOPE: {"freq_range": (520.0, 560.0), "base_amp": 0.7, "hue": 50},      #   
    EmotionType.ANXIETY: {"freq_range": (400.0, 500.0), "base_amp": 1.0, "hue": 270},  #    
}


# ============================================================================
#        (Khala Layer) -   /         
# ============================================================================

@dataclass
class EmotionalWave:
    """
          -              
    
        '  '           .
             .        .
    """
    emotion_type: EmotionType
    intensity: float = 1.0      # 0.0 ~ 2.0 (  )
    frequency: float = 500.0    # Hz (     )
    phase: float = 0.0          #    (주권적 자아)
    duration: float = 1.0       #      
    source_id: Optional[str] = None  #     ID
    
    #      
    resonance_radius: float = 10.0  #       (     )
    decay_rate: float = 0.1         #    
    
    def get_hue(self) -> float:
        """      (hue)    (0-360)"""
        props = EMOTION_WAVE_PROPERTIES.get(self.emotion_type, {})
        return props.get("hue", 0)
    
    def get_strength_at_distance(self, distance: float) -> float:
        """               """
        if distance <= 0:
            return self.intensity
        if distance > self.resonance_radius * 3:
            return 0.0
        
        #        +   
        strength = self.intensity / (1 + (distance / self.resonance_radius) ** 2)
        return max(0.0, strength)
    
    def resonate_with(self, other: 'EmotionalWave') -> float:
        """
                         
        
              =       (  )
              =    (  )
        """
        #         
        if self.emotion_type == other.emotion_type:
            #              
            phase_diff = abs(self.phase - other.phase) % (2 * np.pi)
            phase_match = (1 + np.cos(phase_diff)) / 2.0
            return phase_match * min(self.intensity, other.intensity)
        
        #          -      
        freq_diff = abs(self.frequency - other.frequency)
        freq_resonance = 1.0 / (1.0 + freq_diff / 100.0)
        return freq_resonance * 0.3  #    30%   


@dataclass
class KhalaField:
    """
          -               
    
                             .
    "    "        ,            !
                       !   
    """
    max_waves: int = 500
    
    #          
    active_waves: List[EmotionalWave] = field(default_factory=list)
    
    #          (              !)
    field_strength: float = 1.0  # 1.0 =   , 0.5 =    , 2.0 =    
    
    #   
    total_resonance_events: int = 0
    
    def broadcast_emotion(
        self,
        source_id: str,
        emotion_type: EmotionType,
        intensity: float = 1.0,
        radius: float = 10.0
    ) -> EmotionalWave:
        """
                
        
        Args:
            source_id:     ID
            emotion_type:      
            intensity:    (0.0 ~ 2.0)
            radius:      
        """
        props = EMOTION_WAVE_PROPERTIES.get(emotion_type, {})
        freq_range = props.get("freq_range", (400.0, 600.0))
        
        #               (            )
        freq = freq_range[0] + (freq_range[1] - freq_range[0]) * (intensity / 2.0)
        
        wave = EmotionalWave(
            emotion_type=emotion_type,
            intensity=intensity * self.field_strength,  #         
            frequency=freq,
            phase=np.random.uniform(0, 2 * np.pi),
            source_id=source_id,
            resonance_radius=radius
        )
        
        #       (자기 성찰 엔진)
        self.active_waves.append(wave)
        if len(self.active_waves) > self.max_waves:
            self.active_waves = self.active_waves[-self.max_waves:]
        
        return wave
    
    def receive_emotions(
        self,
        receiver_id: str,
        position: Tuple[float, float, float],
        sensitivity: float = 1.0
    ) -> List[Tuple[EmotionType, float]]:
        """
                        
        
        Returns:
            List of (     ,      )
        """
        received = defaultdict(float)
        
        for wave in self.active_waves:
            if wave.source_id == receiver_id:
                continue  #              
            
            #          
            #            ,                         
            #           wave  resonance_radius        
            base_distance = wave.resonance_radius * 0.5
            distance = base_distance + np.random.uniform(0, wave.resonance_radius)
            
            strength = wave.get_strength_at_distance(distance) * sensitivity
            if strength > 0.01:
                received[wave.emotion_type] += strength
        
        return [(emo, min(2.0, strength)) for emo, strength in received.items()]
    
    def calculate_collective_mood(self) -> Dict[EmotionType, float]:
        """                  """
        mood = defaultdict(float)
        total_intensity = 0.0
        
        for wave in self.active_waves:
            mood[wave.emotion_type] += wave.intensity
            total_intensity += wave.intensity
        
        if total_intensity > 0:
            return {emo: val / total_intensity for emo, val in mood.items()}
        return {}
    
    def decay_waves(self, dt: float = 1.0):
        """      (          )"""
        surviving = []
        for wave in self.active_waves:
            wave.intensity -= wave.decay_rate * dt
            wave.duration -= dt
            if wave.intensity > 0.01 and wave.duration > 0:
                surviving.append(wave)
        self.active_waves = surviving
    
    def set_field_strength(self, strength: float):
        """
                
        
        " ,     (  )           !           !"   
        
        Args:
            strength: 0.1 ~ 2.0 (            ,       )
        """
        self.field_strength = max(0.1, min(2.0, strength))
        logger.info(f"           : {self.field_strength:.1f}x")


# ============================================================================
#        (Symbol Layer) -   /          
# ============================================================================

class SymbolComplexity(Enum):
    """          """
    PROTO = 1       #    (     : " ", " ", "  ")
    BASIC = 2       #    (      : "     ", "    ")
    COMPOUND = 3    #    (     : "       ")
    ABSTRACT = 4    #    (  : "  ", "  ", "  ")
    NARRATIVE = 5   #    (   : "              ...")


@dataclass
class Symbol:
    """
       (Symbol) -              
    
    "                "           
                  ,       '  '              .
    """
    name: str                    #   /   ("maka", " a", "   ")
    meaning: str                 #       (     )
    complexity: SymbolComplexity = SymbolComplexity.PROTO
    
    #       (코드 베이스 구조 로터)
    frequency_signature: float = 0.0
    phase_signature: float = 0.0
    sense_origins: Set[str] = field(default_factory=set)
    
    #      
    usage_count: int = 0
    misunderstanding_count: int = 0  #       (       !)
    
    #        (주권적 자아)
    can_be_subject: bool = False
    can_be_object: bool = False
    can_be_action: bool = False
    
    #      
    related_symbols: List[str] = field(default_factory=list)
    
    def get_ambiguity_score(self) -> float:
        """       (0.0 =   , 1.0 =      )"""
        if self.usage_count == 0:
            return 1.0
        return self.misunderstanding_count / (self.usage_count + self.misunderstanding_count)


@dataclass
class Phrase:
    """
       (Phrase) -        
    
                  ,                .
    """
    symbols: List[Symbol]
    structure: str = "SVO"  #       (  -  -   )
    intended_meaning: str = ""
    
    #       
    transmission_attempts: int = 0
    successful_transmissions: int = 0
    
    def get_complexity(self) -> int:
        """          """
        if not self.symbols:
            return 0
        return max(s.complexity.value for s in self.symbols)
    
    def to_string(self) -> str:
        """           """
        return " ".join(s.name for s in self.symbols)


@dataclass
class Lexicon:
    """
        -                
    
               .             .
          ,         ...              .
    """
    owner_id: str
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    phrases: List[Phrase] = field(default_factory=list)
    
    #       (   )
    grammar_rules: Dict[str, str] = field(default_factory=dict)
    
    #      
    total_learning_attempts: int = 0
    successful_learnings: int = 0
    
    def add_symbol(self, symbol: Symbol) -> bool:
        """          """
        self.total_learning_attempts += 1
        
        if symbol.name in self.symbols:
            #          -   
            self.symbols[symbol.name].usage_count += 1
            return True
        
        #         (   )
        #             
        learn_chance = 1.0 / symbol.complexity.value
        if np.random.random() < learn_chance:
            self.symbols[symbol.name] = symbol
            self.successful_learnings += 1
            logger.debug(f"[{self.owner_id}]        : '{symbol.name}'")
            return True
        
        return False
    
    def find_symbol_for_meaning(self, meaning: str) -> Optional[Symbol]:
        """            """
        for symbol in self.symbols.values():
            if meaning.lower() in symbol.meaning.lower():
                return symbol
        return None
    
    def get_vocabulary_size(self) -> int:
        return len(self.symbols)
    
    def get_learning_rate(self) -> float:
        if self.total_learning_attempts == 0:
            return 0.0
        return self.successful_learnings / self.total_learning_attempts


# ============================================================================
#          (Dual-Language Soul)
# ============================================================================

@dataclass
class DualLayerSoul:
    """
             -   (  )    (  )            
    
    "        (  ),           (  )"...
               '     '       .
    """
    name: str
    age: float = 0.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    #        (  /  )
    emotional_state: Dict[EmotionType, float] = field(default_factory=dict)
    emotional_sensitivity: float = 1.0  #          
    khala_broadcasting_power: float = 1.0  #         
    
    #        (  /  )
    lexicon: Lexicon = field(default_factory=lambda: Lexicon(""))
    language_aptitude: float = 1.0  #         
    symbolic_preference: float = 0.5  # 0=     , 1=     
    
    #   
    relationships: Dict[str, float] = field(default_factory=dict)  #            
    
    #   
    emotional_connections: int = 0  #           
    symbolic_communications: int = 0  #           
    misunderstandings: int = 0  #       (   !)
    
    def __post_init__(self):
        if not self.lexicon.owner_id:
            self.lexicon = Lexicon(owner_id=self.name)
    
    def feel_emotion(self, emotion_type: EmotionType, intensity: float = 1.0):
        """      """
        current = self.emotional_state.get(emotion_type, 0.0)
        self.emotional_state[emotion_type] = min(2.0, current + intensity)
    
    def broadcast_emotion(
        self,
        khala_field: KhalaField,
        emotion_type: Optional[EmotionType] = None
    ) -> Optional[EmotionalWave]:
        """
                    
        
                            ,                
        """
        if emotion_type is None:
            #            
            if not self.emotional_state:
                return None
            emotion_type = max(self.emotional_state, key=self.emotional_state.get)
        
        intensity = self.emotional_state.get(emotion_type, 0.5) * self.khala_broadcasting_power
        
        return khala_field.broadcast_emotion(
            source_id=self.name,
            emotion_type=emotion_type,
            intensity=intensity,
            radius=10.0 * self.khala_broadcasting_power
        )
    
    def receive_emotions(
        self,
        khala_field: KhalaField
    ) -> List[Tuple[EmotionType, float]]:
        """                     """
        received = khala_field.receive_emotions(
            receiver_id=self.name,
            position=self.position,
            sensitivity=self.emotional_sensitivity
        )
        
        #              (  )
        for emotion_type, intensity in received:
            absorbed = intensity * 0.3  # 30%   
            self.feel_emotion(emotion_type, absorbed)
            self.emotional_connections += 1
        
        return received
    
    def try_communicate(
        self,
        receiver: 'DualLayerSoul',
        message: str,
        complexity: SymbolComplexity = SymbolComplexity.PROTO
    ) -> Tuple[bool, str]:
        """
                 
        
        Returns:
            (     ,            )
        """
        self.symbolic_communications += 1
        
        #                        
        symbol = self.lexicon.find_symbol_for_meaning(message)
        
        if symbol is None:
            #           
            symbol = Symbol(
                name=self._generate_word_from_meaning(message),
                meaning=message,
                complexity=complexity
            )
            self.lexicon.add_symbol(symbol)
        
        #                  
        receiver_symbol = receiver.lexicon.symbols.get(symbol.name)
        
        if receiver_symbol is None:
            #              
            #      
            learned = receiver.lexicon.add_symbol(symbol)
            if learned:
                return True, message  #    !
            else:
                self.misunderstandings += 1
                receiver.misunderstandings += 1
                return False, "✨?"  #   
        
        #            (   !)
        if receiver_symbol.meaning == symbol.meaning:
            symbol.usage_count += 1
            receiver_symbol.usage_count += 1
            return True, message
        else:
            #      ,       (  !)
            symbol.misunderstanding_count += 1
            receiver_symbol.misunderstanding_count += 1
            self.misunderstandings += 1
            receiver.misunderstandings += 1
            return False, receiver_symbol.meaning
    
    def _generate_word_from_meaning(self, meaning: str) -> str:
        """
                   (     )
        
                       ,                    .
        """
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = ['m', 'n', 'k', 't', 'p', 'r', 's', 'l']
        
        #                               
        import hashlib
        hash_bytes = hashlib.md5(meaning.encode()).digest()
        
        #                   (     )
        c1 = consonants[hash_bytes[0] % len(consonants)]
        v1 = vowels[hash_bytes[1] % len(vowels)]
        c2 = consonants[hash_bytes[2] % len(consonants)]
        v2 = vowels[hash_bytes[3] % len(vowels)]
        
        return f"{c1}{v1}{c2}{v2}"
    
    def get_communication_style(self) -> str:
        """            """
        total = self.emotional_connections + self.symbolic_communications
        if total == 0:
            return "silent"
        
        khala_ratio = self.emotional_connections / total
        
        if khala_ratio > 0.7:
            return "empath"  #      
        elif khala_ratio < 0.3:
            return "rational"  #      
        else:
            return "balanced"  #   
    
    def get_relationship_gap(self, other: 'DualLayerSoul') -> Dict[str, float]:
        """
                '     '   
        
        "        ,           "...    !
        """
        #         (     )
        emotional_overlap = 0.0
        for emo in EmotionType:
            my_level = self.emotional_state.get(emo, 0.0)
            their_level = other.emotional_state.get(emo, 0.0)
            if my_level > 0 and their_level > 0:
                emotional_overlap += min(my_level, their_level)
        
        #         (     )
        my_words = set(self.lexicon.symbols.keys())
        their_words = set(other.lexicon.symbols.keys())
        if my_words and their_words:
            linguistic_overlap = len(my_words & their_words) / len(my_words | their_words)
        else:
            linguistic_overlap = 0.0
        
        #  (gap) =                     
        gap = max(0.0, emotional_overlap - linguistic_overlap)
        
        return {
            "emotional_connection": emotional_overlap,
            "linguistic_connection": linguistic_overlap,
            "relationship_gap": gap,
            "interpretation": self._interpret_gap(gap)
        }
    
    def _interpret_gap(self, gap: float) -> str:
        if gap > 0.5:
            return "      ,                "
        elif gap > 0.2:
            return "       ,           "
        elif gap < -0.2:
            return "         ,          "
        else:
            return "        "


# ============================================================================
#          (Dual-Layer World)
# ============================================================================

class DualLayerWorld:
    """
             -                
    
                    ,
                          .
    
          ' '             .
    """
    
    def __init__(
        self,
        n_souls: int = 50,
        khala_strength: float = 1.0
    ):
        """
        Args:
            n_souls:     
            khala_strength:          (            !)
        """
        #       (  )
        self.khala_field = KhalaField(field_strength=khala_strength)
        
        #    
        self.souls: Dict[str, DualLayerSoul] = {}
        self._create_souls(n_souls)
        
        #      
        self.time = 0.0
        self.shared_lexicon: Dict[str, int] = {}  #       (  :        )
        
        #       (     )
        self.environmental_stimuli = self._init_stimuli()
        
        #   
        self.total_emotional_events = 0
        self.total_linguistic_events = 0
        self.total_misunderstandings = 0
        self.narrative_fragments: List[str] = []  #            
        
        logger.info(f"DualLayerWorld initialized: {n_souls} souls, khala_strength={khala_strength}")
    
    def _create_souls(self, n_souls: int):
        """      (주권적 자아)"""
        names = ['  ', '  ', ' ', ' ', ' ', ' ', ' ', '  ', '  ', ' ',
                 ' ', '   ', '  ', ' ', ' ', '  ', ' ', ' ', ' ', ' ']
        
        for i in range(n_souls):
            name = f"{names[i % len(names)]}{i}"
            
            #          
            emotional_sensitivity = np.random.uniform(0.5, 1.5)
            language_aptitude = np.random.uniform(0.5, 1.5)
            
            #        (             ,              )
            #                 (   !)
            if np.random.random() < 0.3:  # 30%            
                language_aptitude = emotional_sensitivity * np.random.uniform(0.8, 1.2)
            
            soul = DualLayerSoul(
                name=name,
                position=(np.random.uniform(0, 100), np.random.uniform(0, 100), 0),
                emotional_sensitivity=emotional_sensitivity,
                language_aptitude=language_aptitude,
                symbolic_preference=np.random.uniform(0.3, 0.7)
            )
            
            #         
            for _ in range(np.random.randint(1, 4)):
                emo = np.random.choice(list(EmotionType))
                soul.feel_emotion(emo, np.random.uniform(0.3, 1.0))
            
            self.souls[name] = soul
    
    def _init_stimuli(self) -> Dict[str, Dict[str, Any]]:
        """          (         )"""
        return {
            "sunrise": {"emotions": [(EmotionType.HOPE, 0.8), (EmotionType.JOY, 0.5)]},
            "storm": {"emotions": [(EmotionType.FEAR, 0.7), (EmotionType.SURPRISE, 0.5)]},
            "feast": {"emotions": [(EmotionType.JOY, 1.0), (EmotionType.LOVE, 0.6)]},
            "danger": {"emotions": [(EmotionType.FEAR, 1.2), (EmotionType.ANGER, 0.4)]},
            "reunion": {"emotions": [(EmotionType.JOY, 0.9), (EmotionType.LOVE, 0.8)]},
            "loss": {"emotions": [(EmotionType.SADNESS, 1.0), (EmotionType.LONELINESS, 0.7)]},
            "discovery": {"emotions": [(EmotionType.CURIOSITY, 1.0), (EmotionType.SURPRISE, 0.6)]},
            "beauty": {"emotions": [(EmotionType.JOY, 0.6), (EmotionType.NOSTALGIA, 0.4)]},
            "conflict": {"emotions": [(EmotionType.ANGER, 0.8), (EmotionType.FEAR, 0.3)]},
            "peace": {"emotions": [(EmotionType.TRUST, 0.7), (EmotionType.LOVE, 0.5)]},
        }
    
    def step(self, dt: float = 1.0):
        """        """
        self.time += dt
        
        # 1.       (       )
        self._apply_environmental_stimuli()
        
        # 2.          (     )
        self._update_khala_field()
        
        # 3.          (주권적 자아)
        self._attempt_linguistic_communication()
        
        # 4.      
        self.khala_field.decay_waves(dt)
        
        # 5.      
        for soul in self.souls.values():
            soul.age += dt / 365.0
    
    def _apply_environmental_stimuli(self):
        """        """
        # 10%              
        if np.random.random() < 0.1:
            event = np.random.choice(list(self.environmental_stimuli.keys()))
            stimulus = self.environmental_stimuli[event]
            
            #            
            affected = np.random.choice(
                list(self.souls.values()),
                size=min(10, len(self.souls)),
                replace=False
            )
            
            for soul in affected:
                for emo, intensity in stimulus["emotions"]:
                    soul.feel_emotion(emo, intensity * np.random.uniform(0.5, 1.5))
            
            logger.debug(f"     : {event}, {len(affected)}    ")
    
    def _update_khala_field(self):
        """         -           """
        #               
        for soul in self.souls.values():
            if soul.emotional_state:
                max_emotion = max(soul.emotional_state.values())
                if max_emotion > 0.5:  #            
                    soul.broadcast_emotion(self.khala_field)
                    self.total_emotional_events += 1
        
        #          
        for soul in self.souls.values():
            soul.receive_emotions(self.khala_field)
    
    def _attempt_linguistic_communication(self):
        """        """
        soul_list = list(self.souls.values())
        n_attempts = min(20, len(soul_list) // 2)
        
        for _ in range(n_attempts):
            sender, receiver = np.random.choice(soul_list, size=2, replace=False)
            
            #           (주권적 자아)
            if sender.emotional_state:
                dominant_emotion = max(sender.emotional_state, key=sender.emotional_state.get)
                
                #                 
                messages = {
                    EmotionType.JOY: "  ",
                    EmotionType.FEAR: "  ",
                    EmotionType.SADNESS: "  ",
                    EmotionType.LOVE: "  ",
                    EmotionType.CURIOSITY: "  ",
                    EmotionType.ANGER: "  ",
                    EmotionType.HOPE: "  ",
                }
                
                message = messages.get(dominant_emotion, "  ")
                success, understood = sender.try_communicate(receiver, message)
                
                self.total_linguistic_events += 1
                if not success:
                    self.total_misunderstandings += 1
                    
                    #        ...        !
                    if sender.misunderstandings > 3:
                        self._generate_narrative_fragment(sender, receiver, message, understood)
    
    def _generate_narrative_fragment(
        self,
        soul1: DualLayerSoul,
        soul2: DualLayerSoul,
        intended: str,
        understood: str
    ):
        """          (            )"""
        fragment = (
            f"{soul1.name}  '{intended}'         , "
            f"{soul2.name}  '{understood}'        . "
            f"           {self._describe_emotional_connection(soul1, soul2)}."
        )
        self.narrative_fragments.append(fragment)
        
        if len(self.narrative_fragments) % 10 == 0:
            logger.info(f"          : {fragment}")
    
    def _describe_emotional_connection(
        self,
        soul1: DualLayerSoul,
        soul2: DualLayerSoul
    ) -> str:
        """               """
        gap_info = soul1.get_relationship_gap(soul2)
        gap = gap_info["relationship_gap"]
        
        if gap > 0.5:
            return "           "
        elif gap > 0.2:
            return "           "
        else:
            return "       "
    
    def adjust_khala_strength(self, new_strength: float):
        """
                   
        
        " ,     (  )           !           !"     
        """
        self.khala_field.set_field_strength(new_strength)
    
    def run_simulation(
        self,
        years: int = 100,
        steps_per_year: int = 36,
        report_interval: int = 20
    ) -> Dict[str, Any]:
        """        """
        import time as py_time
        start_time = py_time.time()
        
        total_steps = years * steps_per_year
        
        for step in range(total_steps):
            self.step(dt=1.0)
            
            if step > 0 and step % (report_interval * steps_per_year) == 0:
                year = step // steps_per_year
                self._report_progress(year)
        
        elapsed = py_time.time() - start_time
        
        return self._compile_results(years, elapsed)
    
    def _report_progress(self, year: int):
        """        """
        vocab_sizes = [s.lexicon.get_vocabulary_size() for s in self.souls.values()]
        avg_vocab = np.mean(vocab_sizes) if vocab_sizes else 0
        
        #         
        all_words = defaultdict(int)
        for soul in self.souls.values():
            for word in soul.lexicon.symbols.keys():
                all_words[word] += 1
        shared_count = len([w for w, c in all_words.items() if c > 1])
        
        #         
        sample_souls = list(self.souls.values())[:5]
        avg_gap = 0.0
        if len(sample_souls) >= 2:
            gaps = []
            for i, s1 in enumerate(sample_souls):
                for s2 in sample_souls[i+1:]:
                    gap_info = s1.get_relationship_gap(s2)
                    gaps.append(gap_info["relationship_gap"])
            avg_gap = np.mean(gaps) if gaps else 0.0
        
        print(f"Year {year}: avg_vocab={avg_vocab:.1f}, "
              f"shared_words={shared_count}, "
              f"avg_relationship_gap={avg_gap:.2f}, "
              f"narratives={len(self.narrative_fragments)}")
    
    def _compile_results(self, years: int, elapsed: float) -> Dict[str, Any]:
        """     """
        vocab_sizes = [s.lexicon.get_vocabulary_size() for s in self.souls.values()]
        
        #      
        all_words = defaultdict(int)
        for soul in self.souls.values():
            for word in soul.lexicon.symbols.keys():
                all_words[word] += 1
        shared_words = {w: c for w, c in all_words.items() if c > 1}
        
        #          
        styles = defaultdict(int)
        for soul in self.souls.values():
            styles[soul.get_communication_style()] += 1
        
        #      
        collective_mood = self.khala_field.calculate_collective_mood()
        
        return {
            "years_simulated": years,
            "elapsed_seconds": elapsed,
            "total_souls": len(self.souls),
            "total_emotional_events": self.total_emotional_events,
            "total_linguistic_events": self.total_linguistic_events,
            "total_misunderstandings": self.total_misunderstandings,
            "misunderstanding_rate": (
                self.total_misunderstandings / self.total_linguistic_events
                if self.total_linguistic_events > 0 else 0
            ),
            "avg_vocabulary_size": np.mean(vocab_sizes) if vocab_sizes else 0,
            "max_vocabulary_size": max(vocab_sizes) if vocab_sizes else 0,
            "unique_words": len(all_words),
            "shared_words_count": len(shared_words),
            "communication_styles": dict(styles),
            "collective_mood": {e.value: v for e, v in collective_mood.items()},
            "narrative_fragments": len(self.narrative_fragments),
            "sample_narratives": self.narrative_fragments[:5] if self.narrative_fragments else [],
        }
    
    def get_sample_relationships(self, n: int = 3) -> List[Dict[str, Any]]:
        """        """
        results = []
        sample_souls = list(self.souls.values())[:n*2]
        
        for i in range(0, min(n*2, len(sample_souls)), 2):
            if i + 1 < len(sample_souls):
                s1, s2 = sample_souls[i], sample_souls[i+1]
                gap_info = s1.get_relationship_gap(s2)
                results.append({
                    "souls": (s1.name, s2.name),
                    **gap_info,
                    "shared_words": len(
                        set(s1.lexicon.symbols.keys()) & 
                        set(s2.lexicon.symbols.keys())
                    )
                })
        
        return results


# ============================================================================
# Demo
# ============================================================================

def demo():
    """     """
    print("=" * 70)
    print("Dual-Layer Language System -          ")
    print("=" * 70)
    print()
    print("         :")
    print("  '        (  ),           (  )'...")
    print("             '     '          ?")
    print()
    
    # 1.          (         )
    print("-" * 70)
    print("1.       1.0 (     )")
    print("-" * 70)
    world1 = DualLayerWorld(n_souls=30, khala_strength=1.0)
    results1 = world1.run_simulation(years=50, report_interval=25)
    print(f"     : {results1['misunderstanding_rate']:.2%}")
    print(f"       : {results1['avg_vocabulary_size']:.1f}")
    print(f"        : {results1['narrative_fragments']}")
    print()
    
    # 2.          (        !)
    print("-" * 70)
    print("2.       0.5 (코드 베이스 구조 로터)")
    print("   ' ,                !           !'   ")
    print("-" * 70)
    world2 = DualLayerWorld(n_souls=30, khala_strength=0.5)
    results2 = world2.run_simulation(years=50, report_interval=25)
    print(f"     : {results2['misunderstanding_rate']:.2%}")
    print(f"       : {results2['avg_vocabulary_size']:.1f}")
    print(f"        : {results2['narrative_fragments']}")
    print()
    
    # 3.          (     )
    print("-" * 70)
    print("3.       1.5 (자기 성찰 엔진)")
    print("-" * 70)
    world3 = DualLayerWorld(n_souls=30, khala_strength=1.5)
    results3 = world3.run_simulation(years=50, report_interval=25)
    print(f"     : {results3['misunderstanding_rate']:.2%}")
    print(f"       : {results3['avg_vocabulary_size']:.1f}")
    print(f"        : {results3['narrative_fragments']}")
    print()
    
    #         
    print("=" * 70)
    print("         (Sample)")
    print("=" * 70)
    for rel in world1.get_sample_relationships(3):
        print(f"  {rel['souls'][0]}   {rel['souls'][1]}")
        print(f"         : {rel['emotional_connection']:.2f}")
        print(f"         : {rel['linguistic_connection']:.2f}")
        print(f"         : {rel['relationship_gap']:.2f}")
        print(f"      : {rel['interpretation']}")
        print()
    
    #        
    if results1["sample_narratives"]:
        print("=" * 70)
        print("           ")
        print("=" * 70)
        for i, narrative in enumerate(results1["sample_narratives"], 1):
            print(f"  {i}. {narrative}")
        print()
    
    print("=" * 70)
    print("'   '      .")
    print("         ... '  '  '  '      ... '   '    .     ")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
