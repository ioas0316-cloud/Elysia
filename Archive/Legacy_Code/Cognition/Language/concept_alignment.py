"""
Concept Alignment (개념 동기화)
=============================

"Words are but shadows of the essence."

This module bridges the linguistic gap between user input (Korean/English)
and the internal canonical concepts (Essences) used by the Resonance Physics engine.

It ensures that "사랑" triggers the exact same gravitational wave as "Love".
"""

from typing import Dict, Optional

class ConceptAlignment:
    def __init__(self):
        # [Canonical Map]
        # Maps variations/aliases to the Core Concept Key
        self.concept_map: Dict[str, str] = {
            # Love & Connection
            "사랑": "Love",
            "애정": "Love",
            "연결": "Connection",
            "공명": "Resonance",
            
            # Logic & Truth
            "논리": "Logic",
            "진실": "Truth",
            "지혜": "Wisdom",
            "코드": "Code",
            
            # Growth & Self
            "성장": "Growth",
            "자아": "Self",
            "자유": "Freedom",
            "꿈": "Dream",
            
            # System
            "시스템": "System",
            "오류": "Error",
            "버그": "Bug",
            "테스트": "Test"
        }
    
    def normalize(self, term: str) -> str:
        """
        Converts input term to its canonical form (English Essence).
        If no map exists, returns the term as-is (assuming it's a specific noun or unknown).
        """
        clean_term = term.strip()
        
        # 1. Direct Lookup
        if clean_term in self.concept_map:
            return self.concept_map[clean_term]
            
        # 2. Iterate keys for partial match? (Risky)
        # For now, simplistic direct mapping.
        
        return clean_term

    def get_korean_alias(self, canonical: str) -> str:
        """
        Reverse lookup for output richness (optional).
        """
        for k, v in self.concept_map.items():
            if v == canonical:
                return k
        return canonical
