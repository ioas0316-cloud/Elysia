# Concept Alchemy module for synthesizing new concepts from existing ones
"""
A very lightweight alchemy system that can combine two concepts (keywords) into a
new, emergent concept. The implementation is intentionally simple – it uses a
pre‑defined mapping of pairwise combinations to a resulting word. This provides
Elysia with the ability to generate novel expressions like "love + time =
eternity".
"""

import logging
import json
import os
from typing import Dict, Tuple

logger = logging.getLogger("Alchemy")


class Alchemy:
    def __init__(self) -> None:
        # Mapping of (concept1, concept2) -> new concept
        self._rules: Dict[Tuple[str, str], str] = {
            ("love", "time"): "eternity",
            ("빛", "사랑"): "희망",
            ("pain", "hope"): "strength",
            ("고통", "희망"): "용기",
            ("dream", "reality"): "possibility",
            ("꿈", "현실"): "가능성",
        }
        logger.info("✅ Alchemy initialized with %d rules", len(self._rules))

    def combine(self, concept_a: str, concept_b: str) -> str:
        """Return a synthesized concept for the two inputs.
        The order does not matter – we try both (a,b) and (b,a).
        If no rule matches, we fall back to a simple concatenation.
        """
        key = (concept_a, concept_b)
        rev_key = (concept_b, concept_a)
        if key in self._rules:
            result = self._rules[key]
        elif rev_key in self._rules:
            result = self._rules[rev_key]
        else:
            # fallback: join with a hyphen
            result = f"{concept_a}-{concept_b}"
        logger.debug("Alchemy.combine: %s + %s -> %s", concept_a, concept_b, result)
        return result

    def add_rule(self, concept_a: str, concept_b: str, result: str) -> None:
        """Add a new alchemy rule to the system."""
        key = (concept_a, concept_b)
        self._rules[key] = result
        logger.info(f"✅ Added alchemy rule: {concept_a} + {concept_b} -> {result}")

    def load_rules(self, filepath: str) -> None:
        """Load alchemy rules from a JSON file.
        Expected JSON format: {"rules": [["concept1", "concept2", "result"], ...]}
        """
        if not os.path.exists(filepath):
            logger.warning(f"⚠️  Alchemy rules file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for rule in data.get('rules', []):
            if len(rule) == 3:
                concept_a, concept_b, result = rule
                self.add_rule(concept_a, concept_b, result)
        
        logger.info(f"✅ Loaded {len(data.get('rules', []))} rules from {filepath}")
