"""
Fluctlight Language System (          )
==============================================

          -          ,        .

  :
- "              ,             "
-                 ,               
-  (  )    (  )    (  )     (  )     ( /  )

       (   ):
-    (Planet) =   /  /   (FluctlightParticle)
-    (Star) =   /      
-    (StarSystem) =   
-   /   (Nebula) =        
-    (Galaxy) =        (Saga)
-     (Milky Way) =      

"  (  )    (  )          "
-   : FluctlightEngine -        
-   : LanguageCrystal -            

      :            ,            
"""

from __future__ import annotations

import numpy as np
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from enum import Enum, auto

#    Elysia           
try:
    from Core.1_Body.L1_Foundation.Foundation.Physics.fluctlight import FluctlightParticle, FluctlightEngine
except ImportError:
    #                 
    FluctlightParticle = None
    FluctlightEngine = None

logger = logging.getLogger("FluctlightLanguage")


# =============================================================================
#       (Configuration Constants)
# =============================================================================

#         (Crystallization thresholds)
RESONANCE_THRESHOLD = 0.3        #                    (코드 베이스 구조 로터)
CRYSTALLIZATION_COUNT = 5        #                     
PATTERN_DECAY_RATE = 0.01        #                  

#         
LANGUAGE_LEVEL_THRESHOLDS = [10, 50, 200, 1000]  #            

#          
POETRY_COMPLEXITY_THRESHOLD = 5  #                       


# =============================================================================
# 1.       (Experience Trace) -        
# =============================================================================

@dataclass
class ExperienceTrace:
    """
           - Fluctlight        
    
                       
    """
    # 8         (Elysia       )
    # [  ,   ,   ,   ,    ,   ,  /  ,   ]
    sensory_vector: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    #       
    timestamp: float = 0.0
    location: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    #           
    resonated_with: Set[int] = field(default_factory=set)
    
    #    (   )
    intensity: float = 1.0
    
    #    ID
    trace_id: int = field(default_factory=lambda: id(object()))
    
    def resonate(self, other: 'ExperienceTrace') -> float:
        """                """
        #          
        dot = np.dot(self.sensory_vector, other.sensory_vector)
        norm_self = np.linalg.norm(self.sensory_vector) + 1e-8
        norm_other = np.linalg.norm(other.sensory_vector) + 1e-8
        similarity = dot / (norm_self * norm_other)
        
        #        (자기 성찰 엔진)
        time_diff = abs(self.timestamp - other.timestamp)
        time_factor = np.exp(-time_diff / 100.0)
        
        #       
        space_diff = np.linalg.norm(self.location - other.location)
        space_factor = np.exp(-space_diff / 50.0)
        
        return similarity * time_factor * space_factor
    
    def decay(self, dt: float = 1.0):
        """         """
        self.intensity *= np.exp(-PATTERN_DECAY_RATE * dt)


# =============================================================================
# 2.       (Proto-Pattern) -             
# =============================================================================

@dataclass
class ProtoPattern:
    """
          -          ,        
    
                       
               Symbol(  )      
    """
    #                  
    traces: List[ExperienceTrace] = field(default_factory=list)
    
    #     "  " -          
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    #      
    occurrence_count: int = 0
    
    #    (코드 베이스 구조 로터)
    strength: float = 0.0
    
    #            (     )
    associations: Dict[int, float] = field(default_factory=dict)
    
    #    ID
    pattern_id: int = field(default_factory=lambda: id(object()))
    
    def add_trace(self, trace: ExperienceTrace):
        """       """
        self.traces.append(trace)
        self.occurrence_count += 1
        self.strength = min(1.0, self.strength + 0.1)
        self._update_centroid()
    
    def _update_centroid(self):
        """        """
        if self.traces:
            vectors = np.array([t.sensory_vector for t in self.traces])
            self.centroid = np.mean(vectors, axis=0)
    
    def distance_to(self, trace: ExperienceTrace) -> float:
        """       """
        return np.linalg.norm(self.centroid - trace.sensory_vector)
    
    def is_crystallizable(self) -> bool:
        """         (          )"""
        return self.occurrence_count >= CRYSTALLIZATION_COUNT


# =============================================================================
# 3.         (Crystallized Symbol) -          
# =============================================================================

class SymbolType(Enum):
    """       -            """
    ENTITY = auto()      #    (         =  ,          =  )
    ACTION = auto()      #    (         =    ,          =   )
    STATE = auto()       #    (      =   ,       =    )
    RELATION = auto()    #    (      =   ,     =    )


@dataclass
class CrystallizedSymbol:
    """
            -              
    
            "  "    
    """
    #      
    source_pattern: ProtoPattern
    
    #       (     )
    symbol_type: SymbolType
    
    #       (8  )
    meaning_vector: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    #        (       )
    korean_projection: Optional[str] = None
    english_projection: Optional[str] = None
    
    #      
    usage_count: int = 0
    
    #           
    associations: Dict[int, float] = field(default_factory=dict)
    
    #    ID
    symbol_id: int = field(default_factory=lambda: id(object()))
    
    @classmethod
    def from_pattern(cls, pattern: ProtoPattern) -> 'CrystallizedSymbol':
        """          """
        #       =       
        meaning = pattern.centroid.copy()
        
        #          (자기 성찰 엔진)
        symbol_type = cls._classify_type(meaning)
        
        return cls(
            source_pattern=pattern,
            symbol_type=symbol_type,
            meaning_vector=meaning
        )
    
    @staticmethod
    def _classify_type(meaning: np.ndarray) -> SymbolType:
        """
                        
        
        [  ,   ,   ,   ,    ,   ,  /  ,   ]
        """
        speed = meaning[3]      #   
        intensity = meaning[5]  #   
        intimacy = meaning[4]   #    
        arousal = meaning[7]    #   
        
        #              
        if abs(speed) > 0.5 or abs(intensity) > 0.5:
            return SymbolType.ACTION
        
        #               
        if abs(intimacy) > 0.5:
            return SymbolType.RELATION
        
        #              
        if abs(arousal) < 0.3:
            return SymbolType.STATE
        
        #         
        return SymbolType.ENTITY
    
    def strengthen_association(self, other_id: int, amount: float = 0.1):
        """              (     )"""
        current = self.associations.get(other_id, 0.0)
        self.associations[other_id] = min(1.0, current + amount)


# =============================================================================
# 4.       (Language Crystal) -        
# =============================================================================

class LanguageCrystal:
    """
          -           
    
        "  "   
      (  )               
    """
    
    def __init__(self):
        #          
        self.traces: List[ExperienceTrace] = []
        
        #          
        self.patterns: Dict[int, ProtoPattern] = {}
        
        #            
        self.symbols: Dict[int, CrystallizedSymbol] = {}
        
        #       (자기 성찰 엔진)
        self.grammar_rules: Dict[Tuple[SymbolType, ...], int] = defaultdict(int)
        
        #      
        self.language_level: int = 0
        
        #   
        self.total_experiences: int = 0
        self.crystallization_count: int = 0
    
    def receive_experience(self, sensory_vector: np.ndarray, 
                          timestamp: float, location: np.ndarray) -> Optional[str]:
        """
              -              
        
        Returns:        (   ),     None
        """
        # 1.      
        trace = ExperienceTrace(
            sensory_vector=sensory_vector.copy(),
            timestamp=timestamp,
            location=location.copy(),
            intensity=1.0
        )
        self.traces.append(trace)
        self.total_experiences += 1
        
        # 2.             
        matched_pattern = self._find_resonating_pattern(trace)
        
        if matched_pattern:
            #          
            matched_pattern.add_trace(trace)
            
            #            
            if matched_pattern.is_crystallizable():
                symbol = self._crystallize_pattern(matched_pattern)
                if symbol:
                    return self._express_symbol(symbol)
        else:
            #        
            new_pattern = ProtoPattern()
            new_pattern.add_trace(trace)
            self.patterns[new_pattern.pattern_id] = new_pattern
        
        # 3.          
        self._decay_traces()
        
        return None
    
    def _find_resonating_pattern(self, trace: ExperienceTrace) -> Optional[ProtoPattern]:
        """          """
        best_pattern = None
        best_resonance = RESONANCE_THRESHOLD
        
        for pattern in self.patterns.values():
            distance = pattern.distance_to(trace)
            resonance = 1.0 / (1.0 + distance)  #            
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_pattern = pattern
        
        return best_pattern
    
    def _crystallize_pattern(self, pattern: ProtoPattern) -> Optional[CrystallizedSymbol]:
        """           """
        #                
        for symbol in self.symbols.values():
            if symbol.source_pattern.pattern_id == pattern.pattern_id:
                symbol.usage_count += 1
                return symbol
        
        #        
        symbol = CrystallizedSymbol.from_pattern(pattern)
        self.symbols[symbol.symbol_id] = symbol
        self.crystallization_count += 1
        
        #           
        self._update_language_level()
        
        logger.info(f"   :         (type={symbol.symbol_type.name}, "
                   f"total={len(self.symbols)})")
        
        return symbol
    
    def _express_symbol(self, symbol: CrystallizedSymbol) -> str:
        """           """
        #              
        if symbol.korean_projection:
            return symbol.korean_projection
        
        #               
        return self._project_to_korean(symbol)
    
    def _project_to_korean(self, symbol: CrystallizedSymbol) -> str:
        """
                     
        
        [  ,   ,   ,   ,    ,   ,  /  ,   ]
        """
        v = symbol.meaning_vector
        
        #          
        if symbol.symbol_type == SymbolType.ENTITY:
            return self._project_entity(v)
        elif symbol.symbol_type == SymbolType.ACTION:
            return self._project_action(v)
        elif symbol.symbol_type == SymbolType.STATE:
            return self._project_state(v)
        elif symbol.symbol_type == SymbolType.RELATION:
            return self._project_relation(v)
        
        return "..."
    
    def _project_entity(self, v: np.ndarray) -> str:
        """       """
        temp, bright, size, _, intimacy, intensity, pleasure, arousal = v
        
        #    +           
        if temp > 0.5 and bright > 0.5:
            return " " if intensity > 0.5 else " "
        if temp < -0.5 and bright < 0:
            return " " if size > 0 else "   "
        if temp < -0.5:
            return "  " if intensity > 0 else " "
        
        #           
        if intimacy > 0.5:
            return "  " if pleasure > 0 else " "
        if intimacy < -0.5:
            return "    "
        
        #       
        if size > 0.5:
            return " " if intensity > 0 else "  "
        if size < -0.5:
            return " " if pleasure > 0 else " "
        
        return "  "
    
    def _project_action(self, v: np.ndarray) -> str:
        """       """
        _, _, _, speed, intimacy, intensity, pleasure, arousal = v
        
        #    +          
        if speed > 0.5 and intensity > 0.5:
            return "   " if arousal > 0 else "   "
        if speed > 0.3:
            return "  " if pleasure > 0 else "    "
        if speed < -0.3:
            return "  " if pleasure > 0 else "   "
        
        #           
        if intimacy > 0.5:
            return "  " if pleasure > 0 else "   "
        if intimacy < -0.5:
            return "   "
        
        #  /      
        if pleasure > 0.5:
            return "  " if arousal > 0 else "  "
        if pleasure < -0.5:
            return "  " if arousal > 0 else "   "
        
        return "  "
    
    def _project_state(self, v: np.ndarray) -> str:
        """       """
        temp, bright, size, _, intimacy, intensity, pleasure, arousal = v
        
        #  /   +              
        if pleasure > 0.5:
            if arousal > 0.5:
                return "   "
            return "    " if intimacy > 0 else "    "
        
        if pleasure < -0.5:
            if arousal > 0.5:
                return "   "
            return "   " if intimacy > 0 else "   "
        
        #          
        if temp > 0.5:
            return "    "
        if temp < -0.5:
            return "   "
        
        #          
        if bright > 0.5:
            return "  "
        if bright < -0.5:
            return "   "
        
        return "   "
    
    def _project_relation(self, v: np.ndarray) -> str:
        """       """
        _, _, _, speed, intimacy, intensity, _, _ = v
        
        if intimacy > 0.5:
            return "  " if intensity > 0 else "  "
        if intimacy < -0.5:
            return "  " if intensity > 0 else "   "
        
        if speed > 0:
            return "  "
        
        return "   "
    
    def _decay_traces(self):
        """         """
        surviving = []
        for trace in self.traces:
            trace.decay(1.0)
            if trace.intensity > 0.1:
                surviving.append(trace)
        self.traces = surviving
    
    def _update_language_level(self):
        """          """
        symbol_count = len(self.symbols)
        for i, threshold in enumerate(LANGUAGE_LEVEL_THRESHOLDS):
            if symbol_count >= threshold:
                self.language_level = i + 1
    
    def compose_utterance(self, symbols: List[CrystallizedSymbol]) -> str:
        """                 """
        if not symbols:
            return "..."
        
        #         
        types = tuple(s.symbol_type for s in symbols)
        self.grammar_rules[types] += 1
        
        #            (     )
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                s1.strengthen_association(s2.symbol_id)
                s2.strengthen_association(s1.symbol_id)
        
        #       
        words = [self._express_symbol(s) for s in symbols]
        
        #          
        return self._apply_grammar(words, types)
    
    def _apply_grammar(self, words: List[str], types: Tuple[SymbolType, ...]) -> str:
        """     """
        if len(words) == 1:
            return words[0]
        
        if len(words) == 2:
            # ENTITY + STATE: "X /  Y"
            if types == (SymbolType.ENTITY, SymbolType.STATE):
                return f"{words[0]}  {words[1]}"
            # ENTITY + ACTION: "X /  Y"
            if types == (SymbolType.ENTITY, SymbolType.ACTION):
                return f"{words[0]}  {words[1]}"
            # RELATION + ENTITY: "X Y"
            if types[0] == SymbolType.RELATION:
                return f"{words[0]} {words[1]}"
        
        if len(words) == 3:
            # ENTITY + RELATION + ENTITY: "X  Z  Y"
            if types == (SymbolType.ENTITY, SymbolType.RELATION, SymbolType.ENTITY):
                return f"{words[0]}  {words[2]}  {words[1]}"
            # ENTITY + ACTION + ENTITY: "X  Z  Y"
            if types == (SymbolType.ENTITY, SymbolType.ACTION, SymbolType.ENTITY):
                return f"{words[0]}  {words[2]}  {words[1]}"
        
        #   :        
        return " ".join(words)
    
    def generate_thought(self, heart_state: np.ndarray) -> str:
        """
                     
        
                         
        """
        #               
        if not self.symbols:
            return self._primitive_expression(heart_state)
        
        #                      
        resonating_symbols = []
        
        for symbol in self.symbols.values():
            resonance = np.dot(heart_state, symbol.meaning_vector)
            norm = np.linalg.norm(heart_state) * np.linalg.norm(symbol.meaning_vector)
            if norm > 0:
                resonance /= norm
            
            if resonance > 0.2:  #       
                resonating_symbols.append((symbol, resonance))
        
        if not resonating_symbols:
            #               
            closest = min(self.symbols.values(), 
                         key=lambda s: np.linalg.norm(s.meaning_vector - heart_state))
            return self._express_symbol(closest)
        
        #          
        resonating_symbols.sort(key=lambda x: -x[1])
        
        #    3           
        top_symbols = [s for s, _ in resonating_symbols[:3]]
        
        return self.compose_utterance(top_symbols)
    
    def _primitive_expression(self, state: np.ndarray) -> str:
        """               (   )"""
        temp, bright, _, _, intimacy, intensity, pleasure, arousal = state
        
        #            
        feelings = [
            (abs(pleasure), "  ..." if pleasure > 0 else "  ..."),
            (abs(arousal), "  ..." if arousal > 0 else "  ..."),
            (abs(temp), "  ..." if temp > 0 else "   ..."),
            (abs(intimacy), "   ..." if intimacy > 0 else "  ..."),
        ]
        feelings.sort(key=lambda x: -x[0])
        return feelings[0][1]
    
    def write_diary(self, experiences: List[np.ndarray], year: int) -> str:
        """
              -             
        """
        if not experiences:
            return f"Year {year}: ..."
        
        #        
        avg_experience = np.mean(experiences, axis=0)
        
        #      
        thought = self.generate_thought(avg_experience)
        
        #                 
        if self.language_level >= 2 and len(self.symbols) > 50:
            #      
            secondary = self._find_contrasting_thought(avg_experience)
            if secondary != thought:
                thought = f"{thought}. {secondary}"
        
        return f"Year {year}: {thought}"
    
    def _find_contrasting_thought(self, state: np.ndarray) -> str:
        """           (       )"""
        #             
        opposite_state = -state
        return self.generate_thought(opposite_state)
    
    def get_statistics(self) -> Dict[str, Any]:
        """     """
        return {
            "total_experiences": self.total_experiences,
            "trace_count": len(self.traces),
            "pattern_count": len(self.patterns),
            "symbol_count": len(self.symbols),
            "grammar_rules": len(self.grammar_rules),
            "language_level": self.language_level,
            "crystallization_count": self.crystallization_count,
        }


# =============================================================================
# 5.        (Fractal Soul) -       
# =============================================================================

class FractalSoul:
    """
           -               
    
    "       "        ,                
    """
    
    def __init__(self, name: str, soul_id: int):
        self.name = name
        self.id = soul_id
        self.age = 0
        
        #    (  /  )
        self.heart_state = np.random.randn(8) * 0.3  # 8        
        
        #    (  )
        self.mind = LanguageCrystal()
        
        #    (     )
        self.experiences: List[np.ndarray] = []
        
        #   
        self.relationships: Dict[int, float] = {}
        
        #   
        self.diary_entries: List[str] = []
    
    def experience(self, sensory_input: np.ndarray, timestamp: float):
        """     -       ,       """
        #           
        self.heart_state = 0.9 * self.heart_state + 0.1 * sensory_input
        
        #      
        self.experiences.append(self.heart_state.copy())
        
        #        (주권적 자아)
        location = np.random.randn(3)  #       (   )
        utterance = self.mind.receive_experience(
            self.heart_state, timestamp, location
        )
        
        return utterance
    
    def think(self) -> str:
        """     -              """
        return self.mind.generate_thought(self.heart_state)
    
    def write_diary(self, year: int) -> str:
        """     """
        if not self.experiences:
            return f"Year {year}: ..."
        
        #              
        recent = self.experiences[-100:]  #    100    
        diary = self.mind.write_diary(recent, year)
        
        self.diary_entries.append(diary)
        return diary
    
    def converse_with(self, other: 'FractalSoul') -> Tuple[str, str]:
        """    """
        #           
        social_input = np.zeros(8)
        social_input[4] = 0.5  #       
        social_input[7] = 0.3  #      
        
        my_thought = self.experience(social_input, self.age)
        other_thought = other.experience(social_input, other.age)
        
        #      
        self.relationships[other.id] = self.relationships.get(other.id, 0) + 0.1
        other.relationships[self.id] = other.relationships.get(self.id, 0) + 0.1
        
        return my_thought or self.think(), other_thought or other.think()
    
    def get_self_description(self) -> str:
        """     """
        thought = self.think()
        stats = self.mind.get_statistics()
        
        return (f"   {self.name}. "
                f"  : {self.age}. "
                f"   {stats['symbol_count']} . "
                f"  : {thought}")


# =============================================================================
# 6.      
# =============================================================================

def run_demo(population: int = 10, years: int = 100, seed: int = 42):
    """
         
    
                            
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print("  Fluctlight Language Demo -         ")
    print("=" * 60)
    
    #      
    names = ["  ", "  ", " ", " ", " ", " ", " ", " ", "  ", " "]
    souls = [FractalSoul(names[i % len(names)] + f"_{i}", i) 
             for i in range(population)]
    
    print(f"\n  {population}         ")
    
    #      
    total_crystallizations = 0
    sample_diaries = []
    sample_conversations = []
    
    #        (                  )
    experience_templates = [
        #         ( ,   )
        np.array([0.7, 0.8, 0.3, 0.1, 0.2, 0.5, 0.6, 0.4]),
        #         (  ,  )
        np.array([-0.6, -0.5, 0.2, -0.2, -0.1, 0.3, -0.3, -0.2]),
        #     (주권적 자아)
        np.array([0.2, 0.3, 0.1, 0.3, 0.8, 0.4, 0.7, 0.5]),
        #     (  )
        np.array([0.0, -0.2, 0.0, -0.3, -0.7, 0.2, -0.5, -0.4]),
        #     (   )
        np.array([0.3, 0.4, 0.2, 0.8, 0.3, 0.7, 0.4, 0.8]),
        #      ( )
        np.array([0.1, 0.2, 0.0, -0.6, 0.2, -0.3, 0.5, -0.5]),
        #       
        np.array([0.4, 0.3, 0.1, -0.1, 0.4, 0.3, 0.8, 0.3]),
        #   
        np.array([-0.2, -0.1, 0.3, -0.2, 0.0, 0.6, -0.7, 0.4]),
    ]
    
    for year in range(years):
        #         
        for day in range(365):
            timestamp = year * 365 + day
            
            #         
            for soul in souls:
                #              (            )
                template_idx = (day + soul.id) % len(experience_templates)
                base_exp = experience_templates[template_idx].copy()
                
                #          
                noise = np.random.randn(8) * 0.15
                env_input = base_exp + noise
                
                #       (  ,   )
                season = (day // 91) % 4
                if season == 0:  #  
                    env_input[0] += 0.15
                    env_input[1] += 0.2
                    env_input[6] += 0.2  #      
                elif season == 1:  #   
                    env_input[0] += 0.4
                    env_input[1] += 0.3
                    env_input[7] += 0.2  #   
                elif season == 2:  #   
                    env_input[0] -= 0.1
                    env_input[6] -= 0.1  #      
                else:  #   
                    env_input[0] -= 0.4
                    env_input[1] -= 0.2
                    env_input[4] += 0.2  #       
                
                soul.experience(env_input, timestamp)
                soul.age = year
            
            #      
            if random.random() < 0.05 and len(souls) >= 2:
                s1, s2 = random.sample(souls, 2)
                conv = s1.converse_with(s2)
                if year >= years - 5:  #     5     
                    sample_conversations.append(
                        f"[{s1.name} & {s2.name}] {conv[0]} / {conv[1]}"
                    )
        
        #      
        for soul in souls:
            diary = soul.write_diary(year)
            if year >= years - 5:  #     5     
                sample_diaries.append(diary)
        
        #        
        for soul in souls:
            total_crystallizations += soul.mind.crystallization_count
        
        #       (10   )
        if (year + 1) % 10 == 0:
            avg_symbols = np.mean([len(s.mind.symbols) for s in souls])
            print(f"  Year {year + 1}:       {avg_symbols:.1f} ")
    
    #      
    print("\n" + "=" * 60)
    print("    ")
    print("=" * 60)
    
    for soul in souls[:3]:  #    3  
        stats = soul.mind.get_statistics()
        print(f"\n{soul.name}:")
        print(f"  -   : {stats['symbol_count']} ")
        print(f"  -   : {stats['pattern_count']} ")
        print(f"  -      : {stats['language_level']}")
        print(f"  -      : {soul.get_self_description()}")
    
    print("\n        (    5 ):")
    for diary in sample_diaries[:10]:
        print(f"  {diary}")
    
    print("\n       :")
    for conv in sample_conversations[:10]:
        print(f"  {conv}")
    
    print("\n" + "=" * 60)
    print("       ")
    print(f"   -     : {sum(s.mind.total_experiences for s in souls):,}")
    print(f"   -      : {sum(s.mind.crystallization_count for s in souls):,}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo(population=10, years=100)
