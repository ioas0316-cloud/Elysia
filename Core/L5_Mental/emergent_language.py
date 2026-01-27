"""
Emergent Language System (         )
===========================================

LLM                  .

     :
1.      (Proto-Symbol) -            
2.      (Symbol Combination) -       
3.      (Grammar Emergence) -       
4.       (Natural Language Projection) -   /      

"  (  )    (  )          "
-   :   ,   ,          
-   :           

"       " -               ,        
"""

from __future__ import annotations

import random
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum, auto
from collections import defaultdict
import json

# Import the World Lexicon (The External Truth)
try:
    from Core.L5_Mental.Language.world_lexicon import world_lexicon
except ImportError:
    # Fallback for testing if path is different
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from Core.L5_Mental.Language.world_lexicon import world_lexicon

logger = logging.getLogger("EmergentLanguage")


# =============================================================================
# Configuration Constants
# =============================================================================

# Activation thresholds
SYMBOL_ACTIVATION_THRESHOLD = 0.3    # Minimum resonance to activate a symbol
UTTERANCE_PROBABILITY = 0.1          # Probability of spontaneous utterance

# Symbol evolution
ASSOCIATION_STRENGTH_INCREMENT = 0.05  # Hebbian learning rate
MAX_SEQUENCE_LENGTH = 4              # Maximum symbols in an utterance
SEMANTIC_GAP_THRESHOLD = 0.4         # Threshold to trigger curiosity (feeling unknown)


# =============================================================================
# Meaning Vector Dimensions (8D Sensory Space)
# =============================================================================
# Index 0: Temperature (-1=cold, +1=warm)
# Index 1: Brightness (-1=dark, +1=bright)  
# Index 2: Size (-1=small, +1=large)
# Index 3: Speed (-1=slow, +1=fast)
# Index 4: Intimacy (-1=distant, +1=close)
# Index 5: Intensity (-1=weak, +1=strong)
# Index 6: Pleasure (-1=unpleasant, +1=pleasant)
# Index 7: Arousal (-1=calm, +1=excited)


# =============================================================================
# 1.       (Proto-Symbols) -          
# =============================================================================

class SymbolType(Enum):
    """         """
    ENTITY = auto()      #    ( ,  ,   )
    ACTION = auto()      #    (  ,   ,   )
    STATE = auto()       #    (  ,    ,   )
    RELATION = auto()    #    ( ,   ,   )
    QUANTITY = auto()    #   (  ,   ,   )
    TIME = auto()        #    (  ,   ,   )
    SPACE = auto()       #    (  ,   ,  )
    EMOTION = auto()     #    (  ,   ,   )
    UNKNOWN = auto()     #    (Gap in knowledge)


@dataclass
class ProtoSymbol:
    """
          -                       
    
             .          .
    """
    id: str
    type: SymbolType
    activation: float = 0.0  #          
    frequency: int = 0       #      
    associations: Dict[str, float] = field(default_factory=dict)  #           
    
    #          (8   -      )
    meaning_vector: List[float] = field(default_factory=lambda: [0.0] * 8)
    # [  ,   ,   ,   ,    ,   ,  /  ,   ]
    
    def resonate_with(self, other: 'ProtoSymbol') -> float:
        """                """
        #          
        dot_product = sum(a * b for a, b in zip(self.meaning_vector, other.meaning_vector))
        norm_self = math.sqrt(sum(x**2 for x in self.meaning_vector)) + 0.001
        norm_other = math.sqrt(sum(x**2 for x in other.meaning_vector)) + 0.001
        similarity = dot_product / (norm_self * norm_other)
        
        #      
        association = self.associations.get(other.id, 0.0)
        
        return (similarity + association) / 2
    
    def strengthen_association(self, other_id: str, amount: float = 0.1):
        """      (     :               )"""
        current = self.associations.get(other_id, 0.0)
        self.associations[other_id] = min(1.0, current + amount)


# =============================================================================
# 2.       (Symbol Combination) -       
# =============================================================================

@dataclass
class SymbolSequence:
    """
           -                
    
        "      "
    """
    symbols: List[str]  #    ID 
    pattern_strength: float = 0.0  #         
    occurrences: int = 0  #      
    
    def get_signature(self) -> str:
        """       """
        return "_".join(self.symbols)


# =============================================================================
# 3.       (Grammar Emergence)
# =============================================================================

@dataclass
class GrammarRule:
    """
             
    
     : ENTITY + ACTION     
        STATE + ENTITY     
    """
    pattern: List[SymbolType]  #         
    frequency: int = 0
    examples: List[SymbolSequence] = field(default_factory=list)
    
    def matches(self, sequence: List[SymbolType]) -> bool:
        """                """
        if len(sequence) != len(self.pattern):
            return False
        return all(a == b for a, b in zip(sequence, self.pattern))


# =============================================================================
# 4.        (Natural Language Projection)
# =============================================================================

class LanguageProjector:
    """
              (  /  )    
    
                               
    """
    
    def __init__(self):
        #            (  )
        self.korean_lexicon = {
            #   
            "SELF": " ", "OTHER": " ", "IT": "  ", "WE": "  ",
            "PARENT": "  ", "CHILD": "  ", "FRIEND": "  ",
            
            #   
            "EXIST": "  ", "MOVE": "  ", "EAT": "  ", "SPEAK": "   ",
            "SEE": "  ", "HEAR": "  ", "FEEL": "   ", "THINK": "    ",
            "LOVE": "    ", "HATE": "    ", "WANT": "   ",
            "GIVE": "  ", "TAKE": "  ", "MAKE": "   ",
            
            #   
            "GOOD": "  ", "BAD": "   ", "BIG": "  ", "SMALL": "  ",
            "HAPPY": "   ", "SAD": "   ", "ANGRY": "   ",
            "WARM": "    ", "COLD": "   ", "BRIGHT": "  ", "DARK": "   ",
            
            #   
            "WITH": " ", "TO": "  ", "FROM": "  ", "IN": "  ",
            "AND": "   ", "BUT": "   ", "BECAUSE": "    ",
            
            #   
            "NOW": "  ", "BEFORE": "  ", "AFTER": "  ", "ALWAYS": "  ",
            
            #   
            "HERE": "  ", "THERE": "  ", "UP": " ", "DOWN": "  ",
            
            #   
            "JOY": "  ", "SORROW": "  ", "FEAR": "   ", "LOVE_N": "  ",
        }
        
        #      
        self.english_lexicon = {
            "SELF": "I", "OTHER": "you", "IT": "it", "WE": "we",
            "EXIST": "exist", "MOVE": "go", "EAT": "eat", "SPEAK": "speak",
            "GOOD": "good", "BAD": "bad", "HAPPY": "happy", "SAD": "sad",
            "NOW": "now", "HERE": "here", "WITH": "with", "TO": "to",
        }
        
        #       
        self.korean_templates = {
            (SymbolType.ENTITY, SymbolType.STATE): "{0} /  {1}",
            (SymbolType.ENTITY, SymbolType.ACTION): "{0} /  {1}",
            (SymbolType.ENTITY, SymbolType.RELATION, SymbolType.ENTITY): "{0} /  {2}{1}",
            (SymbolType.ENTITY, SymbolType.ACTION, SymbolType.ENTITY): "{0} /  {2} /  {1}",
            (SymbolType.TIME, SymbolType.ENTITY, SymbolType.ACTION): "{0} {1} /  {2}",
            (SymbolType.EMOTION,): "{0} /     ",
        }
    
    def project_to_korean(self, symbols: List[ProtoSymbol]) -> str:
        """              """
        if not symbols:
            return "..."
        
        #    ID        
        words = []
        for sym in symbols:
            korean = self.korean_lexicon.get(sym.id, sym.id.lower())
            words.append(korean)
        
        #          
        types = tuple(sym.type for sym in symbols)
        template = self.korean_templates.get(types)
        
        if template:
            try:
                return template.format(*words)
            except (IndexError, KeyError):
                pass
        
        #              
        return " ".join(words)
    
    def project_to_english(self, symbols: List[ProtoSymbol]) -> str:
        """              """
        if not symbols:
            return "..."
        
        words = []
        for sym in symbols:
            english = self.english_lexicon.get(sym.id, sym.id.lower())
            words.append(english)
        
        return " ".join(words)


# =============================================================================
# 5.          (Emergent Language Engine)
# =============================================================================

class EmergentLanguageEngine:
    """
             - LLM                 
    
      (  /  )     (  /  )
    
      :
    1.               
    2.             /  
    3.                  
    4.               "  "
    5.            
    """
    
    def __init__(self):
        self.symbols: Dict[str, ProtoSymbol] = {}
        self.sequences: List[SymbolSequence] = []
        self.grammar_rules: List[GrammarRule] = []
        self.projector = LanguageProjector()
        
        #   
        self.total_utterances = 0
        self.vocabulary_size = 0
        self.epiphanies = [] # Log of learned words
        
        #    
        self._initialize_proto_symbols()
        
        logger.info("   Emergent Language Engine initialized")

    def detect_semantic_gap(self, experience_vector: List[float]) -> float:
        """
        Calculates how 'alien' an experience is.
        Returns a gap score (0.0 to 1.0).
        High gap = "I feel something, but I have no word for it."
        """
        if not self.symbols:
            return 1.0

        # Find the closest symbol in meaning space
        max_resonance = 0.0
        for sym in self.symbols.values():
            dot_product = sum(a * b for a, b in zip(experience_vector, sym.meaning_vector))
            norm_exp = math.sqrt(sum(x**2 for x in experience_vector)) + 0.001
            norm_sym = math.sqrt(sum(x**2 for x in sym.meaning_vector)) + 0.001
            similarity = dot_product / (norm_exp * norm_sym)
            if similarity > max_resonance:
                max_resonance = similarity

        # Gap is the inverse of max resonance
        return max(0.0, 1.0 - max_resonance)

    def learn_symbol(self, name: str, meaning_vector: List[float], sym_type: SymbolType, korean_map: str = None, english_map: str = None):
        """
        Assimilates a new symbol from the external world.
        """
        if name in self.symbols:
            return

        # 1. Create the symbol
        new_symbol = ProtoSymbol(id=name, type=sym_type, meaning_vector=meaning_vector)
        self.symbols[name] = new_symbol
        self.vocabulary_size = len(self.symbols)

        # 2. Update Projector
        if korean_map:
            self.projector.korean_lexicon[name] = korean_map
        if english_map:
            self.projector.english_lexicon[name] = english_map

        # 3. Form initial associations (connect to similar concepts)
        for other_id, other_sym in self.symbols.items():
            if other_id == name: continue
            resonance = new_symbol.resonate_with(other_sym)
            if resonance > 0.6:
                new_symbol.strengthen_association(other_id, resonance * 0.5)
                other_sym.strengthen_association(name, resonance * 0.5)

        self.epiphanies.append(name)
        logger.info(f"  [EPIPHANY] Learned new word: {name} (Type: {sym_type})")
    
    def _initialize_proto_symbols(self):
        """            """
        
        #      
        entities = [
            ("SELF", [0, 0.5, 0.5, 0, 1.0, 0.5, 0.6, 0.5]),  #   ,   ,     
            ("OTHER", [0, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5]),
            ("IT", [0, 0.5, 0.5, 0, 0.2, 0.3, 0.5, 0.3]),
            ("WE", [0.3, 0.6, 0.6, 0, 0.9, 0.6, 0.7, 0.6]),
            ("PARENT", [0.4, 0.5, 0.7, 0, 0.8, 0.6, 0.6, 0.4]),
            ("CHILD", [0.3, 0.6, 0.3, 0.3, 0.7, 0.4, 0.7, 0.6]),
            ("FRIEND", [0.2, 0.6, 0.5, 0, 0.8, 0.5, 0.7, 0.5]),
        ]
        
        for id, vec in entities:
            self.symbols[id] = ProtoSymbol(id, SymbolType.ENTITY, meaning_vector=vec)
        
        #      
        actions = [
            ("EXIST", [0, 0.5, 0.5, 0, 0.5, 0.3, 0.5, 0.3]),
            ("MOVE", [0, 0.5, 0.5, 0.7, 0.3, 0.5, 0.5, 0.6]),
            ("EAT", [0.3, 0.4, 0.5, 0.3, 0.3, 0.4, 0.7, 0.5]),
            ("SPEAK", [0, 0.6, 0.4, 0.4, 0.6, 0.5, 0.6, 0.6]),
            ("SEE", [0, 0.8, 0.5, 0.2, 0.4, 0.3, 0.5, 0.5]),
            ("HEAR", [0, 0.3, 0.4, 0.2, 0.4, 0.3, 0.5, 0.5]),
            ("FEEL", [0.5, 0.5, 0.5, 0, 0.6, 0.6, 0.5, 0.6]),
            ("THINK", [0, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.6]),
            ("LOVE", [0.8, 0.7, 0.6, 0, 0.9, 0.7, 0.9, 0.7]),
            ("WANT", [0.3, 0.6, 0.5, 0.3, 0.6, 0.6, 0.6, 0.7]),
            ("GIVE", [0.3, 0.6, 0.5, 0.3, 0.7, 0.5, 0.7, 0.5]),
        ]
        
        for id, vec in actions:
            self.symbols[id] = ProtoSymbol(id, SymbolType.ACTION, meaning_vector=vec)
        
        #      
        states = [
            ("GOOD", [0.3, 0.7, 0.5, 0, 0.5, 0.5, 0.8, 0.5]),
            ("BAD", [-0.3, 0.3, 0.5, 0, 0.3, 0.5, 0.2, 0.5]),
            ("HAPPY", [0.5, 0.8, 0.5, 0.3, 0.7, 0.5, 0.9, 0.7]),
            ("SAD", [-0.2, 0.2, 0.4, -0.2, 0.4, 0.4, 0.1, 0.3]),
            ("WARM", [0.9, 0.6, 0.5, 0, 0.6, 0.5, 0.7, 0.4]),
            ("COLD", [-0.8, 0.4, 0.5, 0, 0.2, 0.5, 0.3, 0.4]),
            ("BIG", [0, 0.5, 0.9, 0, 0.3, 0.7, 0.5, 0.4]),
            ("SMALL", [0, 0.5, 0.1, 0, 0.5, 0.3, 0.5, 0.4]),
        ]
        
        for id, vec in states:
            self.symbols[id] = ProtoSymbol(id, SymbolType.STATE, meaning_vector=vec)
        
        #   /  /  /         
        relations = [("WITH", 0.6), ("TO", 0.4), ("FROM", 0.4), ("IN", 0.5)]
        for id, warmth in relations:
            self.symbols[id] = ProtoSymbol(id, SymbolType.RELATION, 
                meaning_vector=[warmth, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5])
        
        times = [("NOW", 0.5), ("BEFORE", 0.3), ("AFTER", 0.7)]
        for id, brightness in times:
            self.symbols[id] = ProtoSymbol(id, SymbolType.TIME,
                meaning_vector=[0, brightness, 0.5, 0, 0.5, 0.5, 0.5, 0.5])
        
        spaces = [("HERE", 0.7), ("THERE", 0.4)]
        for id, proximity in spaces:
            self.symbols[id] = ProtoSymbol(id, SymbolType.SPACE,
                meaning_vector=[0, 0.5, 0.5, 0, proximity, 0.5, 0.5, 0.5])
        
        emotions = [
            ("JOY", [0.5, 0.9, 0.5, 0.3, 0.7, 0.6, 0.95, 0.8]),
            ("SORROW", [-0.3, 0.2, 0.4, -0.2, 0.4, 0.5, 0.1, 0.3]),
            ("FEAR", [-0.2, 0.3, 0.6, 0.5, 0.2, 0.7, 0.15, 0.9]),
            ("LOVE_N", [0.8, 0.7, 0.6, 0, 0.95, 0.7, 0.9, 0.6]),
        ]
        for id, vec in emotions:
            self.symbols[id] = ProtoSymbol(id, SymbolType.EMOTION, meaning_vector=vec)
        
        self.vocabulary_size = len(self.symbols)
        
        #         
        self._initialize_associations()
    
    def _initialize_associations(self):
        """            """
        #                
        connections = [
            ("SELF", "EXIST", 0.8),
            ("SELF", "FEEL", 0.7),
            ("SELF", "THINK", 0.7),
            ("OTHER", "SELF", 0.5),
            ("LOVE", "HAPPY", 0.7),
            ("LOVE", "OTHER", 0.6),
            ("SAD", "SORROW", 0.9),
            ("HAPPY", "JOY", 0.9),
            ("PARENT", "LOVE", 0.6),
            ("CHILD", "SMALL", 0.5),
            ("FRIEND", "WITH", 0.6),
        ]
        
        for a, b, strength in connections:
            if a in self.symbols and b in self.symbols:
                self.symbols[a].strengthen_association(b, strength)
                self.symbols[b].strengthen_association(a, strength * 0.8)
    
    def experience(self, experience_vector: List[float]) -> List[str]:
        """
                         
        
        experience_vector: 8        
        [  ,   ,   ,   ,    ,   ,  /  ,   ]
        
        Returns:         ID   
        """
        activated = []
        
        # 1. Check for Semantic Gap (Curiosity Trigger)
        gap = self.detect_semantic_gap(experience_vector)
        if gap > SEMANTIC_GAP_THRESHOLD:
            # "The Child Asks"
            # If the gap is significant, we query the World.
            match = world_lexicon.query(experience_vector, threshold=0.7)

            if match:
                name, definition, vector = match
                # Epiphany: "Ah, this is called [NAME]."
                self.learn_symbol(name, vector, SymbolType.UNKNOWN,
                                  korean_map=name, # Simplification: Use English name as key
                                  english_map=name)
                # Immediately activate the new symbol
                if name in self.symbols:
                    self.symbols[name].activation = 1.0
                    activated.append(name)

        for sym_id, symbol in self.symbols.items():
            #              
            resonance = sum(e * m for e, m in zip(experience_vector, symbol.meaning_vector))
            resonance /= 8  #    
            
            if resonance > SYMBOL_ACTIVATION_THRESHOLD:
                symbol.activation = resonance
                symbol.frequency += 1
                activated.append(sym_id)
        
        return activated
    
    def generate_utterance(self, context: Dict[str, Any] = None) -> Tuple[str, str]:
        """
                         
        
        Returns: (     ,      )
        """
        context = context or {}
        
        #            
        active_symbols = sorted(
            [(sym_id, sym) for sym_id, sym in self.symbols.items() if sym.activation > 0.1],
            key=lambda x: x[1].activation,
            reverse=True
        )[:5]  #    5 
        
        if not active_symbols:
            #         
            self.symbols["SELF"].activation = 0.5
            self.symbols["EXIST"].activation = 0.5
            active_symbols = [("SELF", self.symbols["SELF"]), ("EXIST", self.symbols["EXIST"])]
        
        #           (주권적 자아)
        sequence = self._construct_sequence(active_symbols)
        
        #        
        symbols = [self.symbols[sid] for sid in sequence if sid in self.symbols]
        korean = self.projector.project_to_korean(symbols)
        english = self.projector.project_to_english(symbols)
        
        #       (         )
        for i, sid1 in enumerate(sequence):
            for sid2 in sequence[i+1:]:
                if sid1 in self.symbols and sid2 in self.symbols:
                    self.symbols[sid1].strengthen_association(sid2, 0.05)
        
        #      
        seq_obj = SymbolSequence(symbols=sequence, occurrences=1)
        self.sequences.append(seq_obj)
        
        self.total_utterances += 1
        
        #       
        for sym in self.symbols.values():
            sym.activation *= 0.8
        
        return korean, english
    
    def _construct_sequence(self, active_symbols: List[Tuple[str, ProtoSymbol]]) -> List[str]:
        """                    """
        
        #       
        by_type = defaultdict(list)
        for sym_id, sym in active_symbols:
            by_type[sym.type].append(sym_id)
        
        sequence = []
        
        #      :             /            
        order = [
            SymbolType.TIME,
            SymbolType.ENTITY,
            SymbolType.UNKNOWN,  # Prioritize the new/unknown
            SymbolType.STATE,
            SymbolType.ACTION,
            SymbolType.EMOTION,
            SymbolType.RELATION,
            SymbolType.SPACE,
        ]
        
        for sym_type in order:
            if sym_type in by_type:
                sequence.extend(by_type[sym_type][:2])  #           2 
        
        return sequence[:4]  #    4    
    
    def speak_from_emotion(self, emotion: str) -> Tuple[str, str]:
        """          """
        emotion_map = {
            "happy": [0.5, 0.8, 0.5, 0.3, 0.7, 0.5, 0.9, 0.7],
            "sad": [-0.2, 0.2, 0.4, -0.2, 0.4, 0.4, 0.1, 0.3],
            "angry": [0.3, 0.5, 0.6, 0.4, 0.2, 0.8, 0.2, 0.9],
            "love": [0.8, 0.7, 0.5, 0, 0.9, 0.6, 0.9, 0.6],
            "fear": [-0.2, 0.3, 0.6, 0.5, 0.2, 0.7, 0.2, 0.9],
            "curious": [0, 0.7, 0.5, 0.4, 0.5, 0.5, 0.6, 0.8],
            "peaceful": [0.3, 0.6, 0.5, -0.2, 0.6, 0.3, 0.7, 0.2],
        }
        
        vec = emotion_map.get(emotion, [0.5] * 8)
        self.experience(vec)
        return self.generate_utterance()
    
    def speak_about(self, topic: str) -> Tuple[str, str]:
        """            """
        #                 
        topic_upper = topic.upper()
        if topic_upper in self.symbols:
            self.symbols[topic_upper].activation = 0.9
            #             
            for assoc_id, strength in self.symbols[topic_upper].associations.items():
                if assoc_id in self.symbols:
                    self.symbols[assoc_id].activation = strength * 0.7
        
        return self.generate_utterance()
    
    def internal_monologue(self) -> Tuple[str, str]:
        """        """
        #             
        self.symbols["SELF"].activation = 0.8
        self.symbols["THINK"].activation = 0.6
        self.symbols["FEEL"].activation = 0.5
        
        #          
        emotions = ["HAPPY", "SAD", "JOY", "SORROW"]
        emotion = random.choice(emotions)
        if emotion in self.symbols:
            self.symbols[emotion].activation = random.uniform(0.3, 0.7)
        
        return self.generate_utterance()
    
    def get_statistics(self) -> Dict[str, Any]:
        """     """
        return {
            "vocabulary_size": self.vocabulary_size,
            "total_utterances": self.total_utterances,
            "active_symbols": sum(1 for s in self.symbols.values() if s.activation > 0.1),
            "total_associations": sum(len(s.associations) for s in self.symbols.values()),
            "grammar_rules": len(self.grammar_rules),
        }


# =============================================================================
# 6.             (Living Language -      )
# =============================================================================

class LivingLanguageWorld:
    """
               -                 
    
    "       " -                    
      (  )    (  )       
    """
    
    def __init__(self, population: int = 100):
        self.population = population
        self.language_engine = EmergentLanguageEngine()  #      
        
        #          
        self.inhabitants: Dict[int, Dict[str, Any]] = {}
        
        for i in range(population):
            self.inhabitants[i] = {
                "name": f"Soul_{i}",
                "personal_vocabulary": set(),  #            
                "utterance_count": 0,
                "emotional_state": [0.5] * 8,  # 8        
            }
        
        #      
        self.world_time = 0
        self.conversations: List[Dict[str, Any]] = []
        
        logger.info(f"  Living Language World created with {population} souls")
    
    def simulate_day(self) -> List[Dict[str, Any]]:
        """         -       """
        daily_events = []
        
        #              
        for inh_id, inhabitant in self.inhabitants.items():
            #       (주권적 자아)
            experience = [
                random.gauss(0.5, 0.2) for _ in range(8)
            ]
            experience = [max(0, min(1, x)) for x in experience]
            
            #      
            if random.random() < UTTERANCE_PROBABILITY:
                self.language_engine.experience(experience)
                korean, english = self.language_engine.generate_utterance()
                
                inhabitant["utterance_count"] += 1
                
                event = {
                    "time": self.world_time,
                    "speaker": inhabitant["name"],
                    "korean": korean,
                    "english": english,
                    "emotion": experience[6],  #  /  
                }
                daily_events.append(event)
        
        #    (자기 성찰 엔진)
        if len(self.inhabitants) >= 2:
            pair = random.sample(list(self.inhabitants.keys()), 2)
            conversation = self._have_conversation(pair[0], pair[1])
            if conversation:
                daily_events.append(conversation)
        
        self.world_time += 1
        return daily_events
    
    def _have_conversation(self, id1: int, id2: int) -> Optional[Dict[str, Any]]:
        """         """
        inh1 = self.inhabitants[id1]
        inh2 = self.inhabitants[id2]
        
        #           
        self.language_engine.symbols["OTHER"].activation = 0.7
        korean1, english1 = self.language_engine.speak_about("OTHER")
        
        #           
        self.language_engine.symbols["SELF"].activation = 0.6
        korean2, english2 = self.language_engine.speak_from_emotion(
            random.choice(["happy", "curious", "peaceful"])
        )
        
        conversation = {
            "time": self.world_time,
            "type": "conversation",
            "participants": [inh1["name"], inh2["name"]],
            "exchanges": [
                {"speaker": inh1["name"], "korean": korean1},
                {"speaker": inh2["name"], "korean": korean2},
            ]
        }
        
        self.conversations.append(conversation)
        return conversation
    
    def simulate_years(self, years: int) -> Dict[str, Any]:
        """          """
        all_events = []
        
        logger.info(f"  Simulating {years} years...")
        
        for year in range(years):
            for day in range(365):
                events = self.simulate_day()
                if events:
                    all_events.extend(events)
            
            if (year + 1) % 100 == 0:
                stats = self.language_engine.get_statistics()
                logger.info(f"Year {year + 1}: {stats['total_utterances']} utterances")
        
        return {
            "years_simulated": years,
            "total_events": len(all_events),
            "total_conversations": len(self.conversations),
            "language_stats": self.language_engine.get_statistics(),
            "sample_events": all_events[-10:] if all_events else [],
        }


# =============================================================================
#    
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("   EMERGENT LANGUAGE SYSTEM TEST")
    print("   LLM              ")
    print("=" * 70)
    
    engine = EmergentLanguageEngine()
    
    print("\n[1]           ")
    print("-" * 40)
    for emotion in ["happy", "sad", "love", "curious"]:
        korean, english = engine.speak_from_emotion(emotion)
        print(f"  {emotion}: {korean}")
    
    print("\n[2]          ")
    print("-" * 40)
    for topic in ["SELF", "OTHER", "LOVE", "FRIEND"]:
        korean, english = engine.speak_about(topic)
        print(f"  {topic}: {korean}")
    
    print("\n[3]      ")
    print("-" * 40)
    for _ in range(5):
        korean, english = engine.internal_monologue()
        print(f"    {korean}")
    
    print("\n[4]            (100 , 10 )")
    print("-" * 40)
    world = LivingLanguageWorld(population=100)
    results = world.simulate_years(10)
    
    print(f"      : {results['language_stats']['total_utterances']}")
    print(f"      : {results['total_conversations']}")
    
    print("\n       :")
    for event in results["sample_events"][-5:]:
        if event.get("type") == "conversation":
            print(f"    [{event['participants'][0]}] {event['exchanges'][0]['korean']}")
            print(f"    [{event['participants'][1]}] {event['exchanges'][1]['korean']}")
        else:
            print(f"    [{event.get('speaker', '?')}] {event.get('korean', '...')}")
    
    print("\n" + "=" * 70)
    print("  Emergent Language System test complete!")
    print("=" * 70)
