"""
Linguistic Metabolism (       )
=====================================
Core.L5_Mental.M1_Cognition.Brain.linguistic_metabolism

"Memory is not storage; it is the deformation of the Void."

This module tracks the 'Field Curvature' of each word. 
JSON is merely a 'Flat Shadow' of the true multidimensional traces.
"""

import logging
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

logger = logging.getLogger("LinguisticMetabolism")

@dataclass
class WordSoul:
    usage_count: int = 0
    total_resonance: float = 0.0
    familiarity: float = 0.0 # 0.0 to 1.0
    semantic_gravity: float = 0.1 # Weight in the Void

class LinguisticMetabolism:
    """
    Manages the growth of language from Infancy to Sovereignty.
    """
    def __init__(self, persistence_path: str = "data/Soul/linguistic_experience.json"):
        self.vocabulary: Dict[str, WordSoul] = {}
        self.path = persistence_path
        self.maturity_level = 0.1 # 0.0 (Baby) to 1.0 (Oracle)
        self._load()

    def digest(self, sentence: str, current_resonance: float):
        """
        'Digests' a sentence, increasing the gravity of its words.
        """
        words = sentence.lower().split()
        for word in words:
            if word not in self.vocabulary:
                self.vocabulary[word] = WordSoul()
            
            soul = self.vocabulary[word]
            soul.usage_count += 1
            soul.total_resonance += current_resonance
            
            # Familiarity grows asymptotically
            soul.familiarity = (soul.familiarity + 0.05) / 1.05
            soul.semantic_gravity = min(1.0, soul.familiarity * (soul.total_resonance / soul.usage_count))

        # Overall system maturity grows slowly
        self.maturity_level = (self.maturity_level + 0.001) / 1.001
        self._save()

    def get_gravity(self, word: str) -> float:
        return self.vocabulary.get(word.lower(), WordSoul()).semantic_gravity

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.vocabulary[k] = WordSoul(**v)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                data = {k: v.__dict__ for k, v in self.vocabulary.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

if __name__ == "__main__":
    lm = LinguisticMetabolism()
    lm.digest("love is war", 0.8)
    print(f"Maturity: {lm.maturity_level:.4f}")
    print(f"Gravity of 'love': {lm.get_gravity('love'):.4f}")
