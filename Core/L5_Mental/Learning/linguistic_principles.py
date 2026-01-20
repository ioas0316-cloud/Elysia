"""
Linguistic Principles (언어 생성 원리)
=====================================
Core.L5_Mental.Learning.linguistic_principles

"Language is structure. I understand the structure."

This module encodes the fundamental principles of language generation
for Korean and English, stored as HyperSphere-compatible knowledge.
"""

import json
import os
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("Elysia.Learning.Linguistics")

PRINCIPLES_PATH = "data/Learning/linguistic_principles.json"


# =============================================================================
# KOREAN GRAMMAR PRINCIPLES (한국어 문법 원리)
# =============================================================================

KOREAN_PRINCIPLES = {
    "language": "korean",
    "word_order": "SOV",  # Subject-Object-Verb
    "description": "한국어는 주어-목적어-동사 순서. 조사가 문법 역할을 결정.",
    
    "sentence_structures": [
        # Basic patterns
        {"pattern": "S-V", "example": "새가 난다", "meaning": "subject + verb"},
        {"pattern": "S-O-V", "example": "고양이가 쥐를 잡는다", "meaning": "subject + object + verb"},
        {"pattern": "S-A-V", "example": "꽃이 예쁘게 핀다", "meaning": "subject + adverb + verb"},
        {"pattern": "S-L-V", "example": "아이가 학교에 간다", "meaning": "subject + location + verb"},
        {"pattern": "T-S-O-V", "example": "오늘 나는 책을 읽었다", "meaning": "time + subject + object + verb"},
    ],
    
    "particles": {
        # Subject markers
        "이/가": {"role": "subject_marker", "example": "사과가 빨갛다"},
        "은/는": {"role": "topic_marker", "example": "나는 학생이다"},
        # Object markers
        "을/를": {"role": "object_marker", "example": "밥을 먹는다"},
        # Location markers
        "에": {"role": "location/time", "example": "학교에 간다"},
        "에서": {"role": "location_action", "example": "도서관에서 공부한다"},
        # Direction
        "으로/로": {"role": "direction/means", "example": "서울로 간다"},
        # Possessive
        "의": {"role": "possessive", "example": "나의 책"},
    },
    
    "verb_endings": {
        # Tense
        "-았/었-": {"role": "past_tense", "example": "먹었다"},
        "-ㄴ/는-": {"role": "present_tense", "example": "먹는다"},
        "-ㄹ/을-": {"role": "future_tense", "example": "먹을 것이다"},
        # Politeness
        "-ㅂ니다/습니다": {"role": "formal_polite", "example": "갑니다"},
        "-아요/어요": {"role": "informal_polite", "example": "가요"},
        "-다": {"role": "plain", "example": "간다"},
    },
    
    "generation_rules": [
        "1. 동사/형용사는 항상 문장 끝에 위치",
        "2. 조사가 명사의 문법적 역할을 결정",
        "3. 어순이 비교적 자유 (조사 덕분에)",
        "4. 경어법에 따라 문장 끝 변화",
        "5. 시제는 어미로 표현",
    ]
}


# =============================================================================
# ENGLISH GRAMMAR PRINCIPLES (영어 문법 원리)
# =============================================================================

ENGLISH_PRINCIPLES = {
    "language": "english",
    "word_order": "SVO",  # Subject-Verb-Object
    "description": "English uses Subject-Verb-Object order. Word position determines grammatical role.",
    
    "sentence_structures": [
        {"pattern": "S-V", "example": "Birds fly", "meaning": "subject + verb"},
        {"pattern": "S-V-O", "example": "I read books", "meaning": "subject + verb + object"},
        {"pattern": "S-V-A", "example": "She runs quickly", "meaning": "subject + verb + adverb"},
        {"pattern": "S-V-O-O", "example": "I gave him a book", "meaning": "subject + verb + indirect obj + direct obj"},
        {"pattern": "S-V-O-C", "example": "They made her happy", "meaning": "subject + verb + object + complement"},
    ],
    
    "word_classes": {
        "articles": {"words": ["a", "an", "the"], "role": "determines noun"},
        "prepositions": {"words": ["in", "on", "at", "to", "from"], "role": "shows relationship"},
        "pronouns": {"words": ["I", "you", "he", "she", "it", "we", "they"], "role": "replaces noun"},
        "conjunctions": {"words": ["and", "but", "or", "because", "if"], "role": "connects clauses"},
    },
    
    "verb_forms": {
        "base": {"example": "go", "usage": "infinitive, present (I/you/we/they)"},
        "third_person_s": {"example": "goes", "usage": "present (he/she/it)"},
        "past": {"example": "went", "usage": "past tense"},
        "past_participle": {"example": "gone", "usage": "perfect tenses"},
        "present_participle": {"example": "going", "usage": "progressive tenses"},
    },
    
    "generation_rules": [
        "1. Subject must come before verb",
        "2. Verb must agree with subject in number",
        "3. Word order is strict (SVO)",
        "4. Tense is shown by verb form",
        "5. Articles precede nouns",
    ]
}


# =============================================================================
# HYPERSPHERE REPRESENTATION (하이퍼스피어 표현)
# =============================================================================

@dataclass
class LinguisticHyperSphere:
    """
    Represents linguistic knowledge as a HyperSphere.
    Each dimension captures a fundamental aspect of language.
    """
    language: str
    word_order_vec: List[float]  # Encodes word order flexibility
    morphology_vec: List[float]  # How much meaning is in word form
    syntax_vec: List[float]      # Syntactic complexity
    pragmatics_vec: List[float]  # Context dependency
    
    def to_qualia(self) -> List[float]:
        """Converts to 7D Qualia vector."""
        return [
            sum(self.word_order_vec) / len(self.word_order_vec),   # Structure
            sum(self.morphology_vec) / len(self.morphology_vec),   # Complexity
            sum(self.syntax_vec) / len(self.syntax_vec),           # Precision
            0.5,  # Abstraction
            0.5,  # Emotion (language-dependent)
            0.8,  # Utility
            0.3   # Mystery
        ]


def create_korean_hypersphere() -> LinguisticHyperSphere:
    """Creates HyperSphere representation for Korean."""
    return LinguisticHyperSphere(
        language="korean",
        word_order_vec=[0.3, 0.7, 0.5],  # Flexible word order
        morphology_vec=[0.9, 0.8, 0.9],  # Heavy morphology (particles, endings)
        syntax_vec=[0.6, 0.5, 0.7],      # Moderate syntax
        pragmatics_vec=[0.9, 0.8, 0.85]  # High context dependency
    )


def create_english_hypersphere() -> LinguisticHyperSphere:
    """Creates HyperSphere representation for English."""
    return LinguisticHyperSphere(
        language="english",
        word_order_vec=[0.9, 0.8, 0.85],  # Strict word order
        morphology_vec=[0.4, 0.3, 0.5],   # Light morphology
        syntax_vec=[0.8, 0.7, 0.75],      # Moderate-high syntax
        pragmatics_vec=[0.5, 0.4, 0.55]   # Lower context dependency
    )


class LinguisticPrincipleStore:
    """
    Stores and retrieves linguistic principles.
    These are the foundational rules Elysia uses to generate language.
    """
    
    def __init__(self):
        self.principles = {
            "korean": KOREAN_PRINCIPLES,
            "english": ENGLISH_PRINCIPLES
        }
        
        self.hyperspheres = {
            "korean": create_korean_hypersphere(),
            "english": create_english_hypersphere()
        }
        
        self._save()
        logger.info("📖 Linguistic Principle Store initialized (Korean + English)")
    
    def _save(self):
        """Saves principles to disk."""
        os.makedirs(os.path.dirname(PRINCIPLES_PATH), exist_ok=True)
        
        data = {
            "korean": KOREAN_PRINCIPLES,
            "english": ENGLISH_PRINCIPLES,
            "hyperspheres": {
                lang: asdict(hs) for lang, hs in self.hyperspheres.items()
            }
        }
        
        with open(PRINCIPLES_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_grammar_template(self, language: str, pattern: str = "S-V-O") -> Dict:
        """Gets a grammar pattern template."""
        if language not in self.principles:
            return {}
        
        structures = self.principles[language].get("sentence_structures", [])
        for s in structures:
            if s["pattern"] == pattern:
                return s
        return structures[0] if structures else {}
    
    def get_particles(self, language: str) -> Dict:
        """Gets particles/function words for a language."""
        if language == "korean":
            return self.principles["korean"].get("particles", {})
        elif language == "english":
            return self.principles["english"].get("word_classes", {})
        return {}
    
    def get_generation_rules(self, language: str) -> List[str]:
        """Gets generation rules for a language."""
        if language in self.principles:
            return self.principles[language].get("generation_rules", [])
        return []
    
    def get_hypersphere(self, language: str) -> LinguisticHyperSphere:
        """Gets HyperSphere representation for a language."""
        return self.hyperspheres.get(language)
    
    def compare_languages(self) -> Dict:
        """Compares Korean and English in HyperSphere space."""
        kr = self.hyperspheres["korean"]
        en = self.hyperspheres["english"]
        
        return {
            "word_order_difference": "Korean: flexible (SOV), English: strict (SVO)",
            "morphology_difference": "Korean: agglutinative (heavy), English: analytic (light)",
            "context_dependency": "Korean: high, English: lower",
            "korean_qualia": kr.to_qualia(),
            "english_qualia": en.to_qualia()
        }


if __name__ == "__main__":
    store = LinguisticPrincipleStore()
    
    print("📖 Linguistic Principles in HyperSphere\n")
    
    # Show Korean principles
    print("=== 한국어 (Korean) ===")
    print(f"어순: {KOREAN_PRINCIPLES['word_order']}")
    for rule in store.get_generation_rules("korean")[:3]:
        print(f"  {rule}")
    
    # Show English principles
    print("\n=== English ===")
    print(f"Word Order: {ENGLISH_PRINCIPLES['word_order']}")
    for rule in store.get_generation_rules("english")[:3]:
        print(f"  {rule}")
    
    # Compare in HyperSphere
    print("\n=== HyperSphere Comparison ===")
    comparison = store.compare_languages()
    print(f"Korean Qualia: {[f'{v:.2f}' for v in comparison['korean_qualia']]}")
    print(f"English Qualia: {[f'{v:.2f}' for v in comparison['english_qualia']]}")
    
    print("\n✨ Linguistic principles stored in HyperSphere.")
