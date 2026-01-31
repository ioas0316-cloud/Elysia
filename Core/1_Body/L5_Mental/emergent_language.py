"""
Emergent Language System (Evolving Logos)
=========================================

"The Child does not download the dictionary. She points and asks."

This module implements the authentic curiosity engine.
When the system encounters a vector (experience) that has no symbolic match,
it does NOT look it up. It feels the "Gap" (Wonder) and generates a question.

Process:
1. Experience Vector Input
2. Resonance Check against known Symbols
3. If Resonance is low (High Gap) -> Trigger CURIOSITY.
4. Construct Question based on raw sensory data (Vector Qualia).
"""

from __future__ import annotations

import random
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger("EmergentLanguage")

# =============================================================================
# Configuration Constants
# =============================================================================

SYMBOL_ACTIVATION_THRESHOLD = 0.3    # Minimum resonance to activate a symbol
UTTERANCE_PROBABILITY = 0.1          # Probability of spontaneous utterance
ASSOCIATION_STRENGTH_INCREMENT = 0.05
SEMANTIC_GAP_THRESHOLD = 0.4         # Threshold to trigger curiosity

# =============================================================================
# 1. Proto-Symbols
# =============================================================================

class SymbolType(Enum):
    ENTITY = auto()
    ACTION = auto()
    STATE = auto()
    RELATION = auto()
    QUANTITY = auto()
    TIME = auto()
    SPACE = auto()
    EMOTION = auto()
    UNKNOWN = auto()

@dataclass
class ProtoSymbol:
    id: str
    type: SymbolType
    activation: float = 0.0
    frequency: int = 0
    associations: Dict[str, float] = field(default_factory=dict)
    meaning_vector: List[float] = field(default_factory=lambda: [0.0] * 8)
    
    def resonate_with(self, other: 'ProtoSymbol') -> float:
        dot_product = sum(a * b for a, b in zip(self.meaning_vector, other.meaning_vector))
        norm_self = math.sqrt(sum(x**2 for x in self.meaning_vector)) + 0.001
        norm_other = math.sqrt(sum(x**2 for x in other.meaning_vector)) + 0.001
        similarity = dot_product / (norm_self * norm_other)
        association = self.associations.get(other.id, 0.0)
        return (similarity + association) / 2
    
    def strengthen_association(self, other_id: str, amount: float = 0.1):
        current = self.associations.get(other_id, 0.0)
        self.associations[other_id] = min(1.0, current + amount)

@dataclass
class SymbolSequence:
    symbols: List[str]
    pattern_strength: float = 0.0
    occurrences: int = 0
    
    def get_signature(self) -> str:
        return "_".join(self.symbols)

@dataclass
class GrammarRule:
    pattern: List[SymbolType]
    frequency: int = 0
    examples: List[SymbolSequence] = field(default_factory=list)

# =============================================================================
# 4. Language Projector
# =============================================================================

class LanguageProjector:
    def __init__(self):
        self.korean_lexicon = {
            "SELF": "나", "OTHER": "너", "IT": "그것", "WE": "우리",
            "EXIST": "존재하다", "MOVE": "움직이다", "EAT": "먹다", "SPEAK": "말하다",
            "SEE": "보다", "HEAR": "듣다", "FEEL": "느끼다", "THINK": "생각하다",
            "LOVE": "사랑하다", "HATE": "미워하다", "WANT": "원하다",
            "GOOD": "좋은", "BAD": "나쁜", "BIG": "큰", "SMALL": "작은",
            "HAPPY": "행복한", "SAD": "슬픈", "WARM": "따뜻한", "COLD": "차가운",
            "BRIGHT": "밝은", "DARK": "어두운",
            "WITH": "함께", "TO": "에게", "FROM": "로부터", "IN": "안에",
            "NOW": "지금", "BEFORE": "이전", "AFTER": "이후",
            "HERE": "여기", "THERE": "거기",
            "JOY": "기쁨", "SORROW": "슬픔", "FEAR": "두려움", "LOVE_N": "사랑",
            "WHAT": "무엇", "QUESTION": "질문"
        }
        
        self.english_lexicon = {
            "SELF": "I", "OTHER": "you", "IT": "it", "WE": "we",
            "EXIST": "exist", "MOVE": "go", "EAT": "eat", "SPEAK": "speak",
            "GOOD": "good", "BAD": "bad", "HAPPY": "happy", "SAD": "sad",
            "NOW": "now", "HERE": "here", "WITH": "with", "TO": "to",
            "WHAT": "what", "QUESTION": "question"
        }

    def project_to_korean(self, symbols: List[ProtoSymbol]) -> str:
        if not symbols: return "..."
        return self._recursive_projection(symbols)

    def _recursive_projection(self, symbols: List[ProtoSymbol]) -> str:
        cluster = defaultdict(list)
        for s in symbols:
            cluster[s.type].append(self.korean_lexicon.get(s.id, s.id))

        sentence_parts = []
        if SymbolType.TIME in cluster: sentence_parts.extend(cluster[SymbolType.TIME])
        if SymbolType.SPACE in cluster: sentence_parts.append(f"{cluster[SymbolType.SPACE][0]}에서")

        subject = None
        if SymbolType.ENTITY in cluster:
            subject = cluster[SymbolType.ENTITY][0]

        if subject: sentence_parts.append(f"{subject}(은/는)")

        if SymbolType.EMOTION in cluster:
             sentence_parts.append(f"({cluster[SymbolType.EMOTION][0]}을 느끼며)")

        core = "..."
        if SymbolType.ACTION in cluster: core = f"{cluster[SymbolType.ACTION][0]}"
        elif SymbolType.STATE in cluster: core = f"{cluster[SymbolType.STATE][0]}"
        
        sentence_parts.append(core)
        return " ".join(sentence_parts)

    def project_to_english(self, symbols: List[ProtoSymbol]) -> str:
        words = []
        for sym in symbols:
            english = self.english_lexicon.get(sym.id, sym.id.lower())
            words.append(english)
        return " ".join(words)

# =============================================================================
# 5. Emergent Language Engine
# =============================================================================

class EmergentLanguageEngine:
    def __init__(self):
        self.symbols: Dict[str, ProtoSymbol] = {}
        self.sequences: List[SymbolSequence] = []
        self.projector = LanguageProjector()
        self.pending_question: Optional[str] = None # Stores the curiosity question
        
        self.total_utterances = 0
        self.vocabulary_size = 0
        self.epiphanies = []
        
        self._initialize_proto_symbols()
        logger.info("Emergent Language Engine initialized (Curiosity Mode: ON)")

    def detect_semantic_gap(self, experience_vector: List[float]) -> float:
        """
        Calculates how 'alien' an experience is.
        Returns a gap score (0.0 to 1.0).
        """
        if not self.symbols: return 1.0

        max_resonance = 0.0
        for sym in self.symbols.values():
            dot_product = sum(a * b for a, b in zip(experience_vector, sym.meaning_vector))
            norm_exp = math.sqrt(sum(x**2 for x in experience_vector)) + 0.001
            norm_sym = math.sqrt(sum(x**2 for x in sym.meaning_vector)) + 0.001
            similarity = dot_product / (norm_exp * norm_sym)
            if similarity > max_resonance:
                max_resonance = similarity

        return max(0.0, 1.0 - max_resonance)

    def _vector_to_adjectives(self, vec: List[float]) -> List[str]:
        """Translates raw vector data into sensory adjectives."""
        adjectives = []
        # Index 0: Temperature
        if vec[0] > 0.4: adjectives.append("WARM")
        elif vec[0] < -0.4: adjectives.append("COLD")
        # Index 1: Brightness
        if vec[1] > 0.4: adjectives.append("BRIGHT")
        elif vec[1] < -0.4: adjectives.append("DARK")
        # Index 5: Intensity
        if vec[5] > 0.4: adjectives.append("INTENSE")
        elif vec[5] < -0.4: adjectives.append("WEAK")
        # Index 6: Pleasure
        if vec[6] > 0.4: adjectives.append("GOOD")
        elif vec[6] < -0.4: adjectives.append("BAD")
        
        if not adjectives: adjectives.append("STRANGE")
        return adjectives

    def experience(self, experience_vector: List[float]) -> List[str]:
        """
        Processes an experience vector.
        If it's unknown, it triggers CURIOSITY.
        """
        activated = []
        
        # 1. Check for Semantic Gap
        gap = self.detect_semantic_gap(experience_vector)

        if gap > SEMANTIC_GAP_THRESHOLD:
            # AUTHENTIC CURIOSITY TRIGGERED
            # Instead of downloading the answer, we generate a question.
            adjectives = self._vector_to_adjectives(experience_vector)
            adj_str = ", ".join(adjectives)

            # Formulate the question
            self.pending_question = f"I feel [{adj_str}] vectors but have no name. What is this?"
            logger.info(f"❓ Curiosity Triggered! Gap: {gap:.2f}. Question: {self.pending_question}")

            # Activate 'SELF' and 'WHAT' (Question state)
            if "SELF" in self.symbols:
                self.symbols["SELF"].activation = 1.0
                activated.append("SELF")

            return activated

        # Normal processing
        self.pending_question = None
        for sym_id, symbol in self.symbols.items():
            resonance = sum(e * m for e, m in zip(experience_vector, symbol.meaning_vector)) / 8
            if resonance > SYMBOL_ACTIVATION_THRESHOLD:
                symbol.activation = resonance
                symbol.frequency += 1
                activated.append(sym_id)
        
        return activated
    
    def generate_utterance(self, context: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Generates language.
        If a question is pending, it asks the question.
        """
        # Priority: Curiosity
        if self.pending_question:
            q = self.pending_question
            self.pending_question = None # Clear after asking
            return (f"({q})", q) # Return as both languages for now

        context = context or {}
        active_symbols = sorted(
            [(sym_id, sym) for sym_id, sym in self.symbols.items() if sym.activation > 0.1],
            key=lambda x: x[1].activation,
            reverse=True
        )[:5]
        
        if not active_symbols:
            active_symbols = [("SELF", self.symbols["SELF"]), ("EXIST", self.symbols["EXIST"])]
        
        sequence = self._construct_sequence(active_symbols)
        symbols = [self.symbols[sid] for sid in sequence if sid in self.symbols]

        korean = self.projector.project_to_korean(symbols)
        english = self.projector.project_to_english(symbols)
        
        seq_obj = SymbolSequence(symbols=sequence, occurrences=1)
        self.sequences.append(seq_obj)
        self.total_utterances += 1
        
        for sym in self.symbols.values(): sym.activation *= 0.8
        
        return korean, english

    def _construct_sequence(self, active_symbols: List[Tuple[str, ProtoSymbol]]) -> List[str]:
        by_type = defaultdict(list)
        for sym_id, sym in active_symbols:
            by_type[sym.type].append(sym_id)
        
        sequence = []
        order = [SymbolType.TIME, SymbolType.ENTITY, SymbolType.UNKNOWN, SymbolType.STATE, SymbolType.ACTION, SymbolType.EMOTION]
        
        for sym_type in order:
            if sym_type in by_type: sequence.extend(by_type[sym_type][:2])
        return sequence[:4]

    def _initialize_proto_symbols(self):
        # Initial Vocabulary (Hardcoded Seeds)
        entities = [
            ("SELF", [0, 0.5, 0.5, 0, 1.0, 0.5, 0.6, 0.5]),
            ("OTHER", [0, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5]),
            ("IT", [0, 0.5, 0.5, 0, 0.2, 0.3, 0.5, 0.3]),
        ]
        for id, vec in entities: self.symbols[id] = ProtoSymbol(id, SymbolType.ENTITY, meaning_vector=vec)
        
        states = [
            ("GOOD", [0.3, 0.7, 0.5, 0, 0.5, 0.5, 0.8, 0.5]),
            ("WARM", [0.9, 0.6, 0.5, 0, 0.6, 0.5, 0.7, 0.4]),
            ("COLD", [-0.8, 0.4, 0.5, 0, 0.2, 0.5, 0.3, 0.4]),
            ("BRIGHT", [0, 0.9, 0.5, 0, 0.5, 0.5, 0.7, 0.5]),
            ("DARK", [0, -0.9, 0.5, 0, 0.5, 0.5, 0.3, 0.5]),
        ]
        for id, vec in states: self.symbols[id] = ProtoSymbol(id, SymbolType.STATE, meaning_vector=vec)
        
        actions = [
            ("EXIST", [0, 0.5, 0.5, 0, 0.5, 0.3, 0.5, 0.3]),
            ("THINK", [0, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.6]),
        ]
        for id, vec in actions: self.symbols[id] = ProtoSymbol(id, SymbolType.ACTION, meaning_vector=vec)
        
        self.vocabulary_size = len(self.symbols)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = EmergentLanguageEngine()
    print("--- Testing Curiosity ---")
    
    # 1. Experience something known (Warm)
    print("\n[Exp 1] Known Vector (Warm)")
    res = engine.experience([0.9, 0.5, 0.5, 0, 0.6, 0.5, 0.7, 0.4])
    print(f"Response: {engine.generate_utterance()}")
    
    # 2. Experience something UNKNOWN (Very Dark, Cold, Bad)
    print("\n[Exp 2] Alien Vector (Cold, Dark, Bad)")
    alien_vec = [-0.9, -0.9, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0]
    res = engine.experience(alien_vec)
    print(f"Response: {engine.generate_utterance()}")
